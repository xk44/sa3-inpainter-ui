"""RES4LYF exponential Runge-Kutta samplers for SA3 rectified flow.

Ports the pure-math core (phi functions + Butcher tableaux) from
ClownsharkBatwing/RES4LYF, adapted to SA3's rectified flow velocity
interface. No ComfyUI dependency.

SA3 RF convention:
  model(x, t, **extra_args) -> velocity v
  denoised = x - t * v
  t in [1, 0], decreasing (1=noise, 0=clean)

Exponential integrator formulation:
  h = log(t_curr / t_next) > 0
  eps_i = denoised_i - x_0   (exponential residual)
  x_stage = x_0 * (t_next/t_curr)^c_i + h * sum_j(a[i][j] * eps_j)
  x_next  = x_0 * (t_next/t_curr)   + h * sum_j(b[j]    * eps_j)
"""
import math
import torch
from tqdm import tqdm


# -------- phi functions --------

def _phi(j, z):
    """phi_j(z) via Taylor series. Stable for all z including near zero.

    phi_j(z) = sum_{k=0}^inf z^k / (k+j)!
    phi_0(z) = exp(z)
    phi_1(z) = (exp(z) - 1) / z
    phi_2(z) = (exp(z) - 1 - z) / z^2
    """
    if j == 0:
        return math.exp(z)
    if abs(z) < 1e-8:
        return 1.0 / math.factorial(j)
    result = 0.0
    term = 1.0 / math.factorial(j)
    for k in range(200):
        result += term
        term *= z / (k + j + 1)
        if abs(term) < 1e-16 * max(abs(result), 1e-30):
            break
    return result


class _PhiCache:
    """Cached phi evaluations for a step with given h and c-nodes."""
    def __init__(self, h, c_nodes):
        self.h = h
        self.c = c_nodes
        self._cache = {}

    def __call__(self, j, i=-1):
        """phi_j(-h * c_i). i=-1 -> c=1 (full step). i>=1 -> c[i-1]."""
        key = (j, i)
        if key not in self._cache:
            c = 1.0 if i < 0 else self.c[i - 1]
            if c == 0.0 and j > 0:
                val = 0.0
            else:
                val = _phi(j, -self.h * c)
            self._cache[key] = val
        return self._cache[key]


# -------- RK coefficient builders --------

def _gamma(c2, c3):
    """Order-3 consistency condition coefficient."""
    return (3 * c3**3 - 2 * c3) / (c2 * (2 - 3 * c2))


def _coefficients_res_2s(h, c2=0.5):
    c = [0.0, c2]
    p = _PhiCache(h, c)
    a = [[0.0, 0.0],
         [c2 * p(1, 2), 0.0]]
    b2 = p(2) / c2
    b1 = p(1) - b2
    return a, [b1, b2], c


def _coefficients_res_3s(h, c2=0.5, c3=1.0):
    c = [0.0, c2, c3]
    p = _PhiCache(h, c)
    g = _gamma(c2, c3)
    a3_2 = g * c2 * p(2, 2) + (c3**2 / c2) * p(2, 3)
    a = [[0.0, 0.0, 0.0],
         [c2 * p(1, 2), 0.0, 0.0],
         [c3 * p(1, 3) - a3_2, a3_2, 0.0]]
    b3 = p(2) / (g * c2 + c3)
    b2 = g * b3
    b1 = p(1) - b2 - b3
    return a, [b1, b2, b3], c


def _coefficients_res_2s_stable(h, c2=0.5):
    """res_2s with improved stability for large steps."""
    c = [0.0, c2]
    p = _PhiCache(h, c)
    a = [[0.0, 0.0],
         [c2 * p(1, 2), 0.0]]
    b2 = (1.0 / (2.0 * c2)) * p(2)
    b1 = p(1) - b2
    return a, [b1, b2], c


def _coefficients_dpmpp_2s(h, c2=0.5):
    return _coefficients_res_2s(h, c2)


def _coefficients_dpmpp_3s(h, c2=1.0/3, c3=2.0/3):
    c = [0.0, c2, c3]
    p = _PhiCache(h, c)
    a3_2 = (c3 / c2) * (p(2, 3) - p(2, 2))
    a = [[0.0, 0.0, 0.0],
         [c2 * p(1, 2), 0.0, 0.0],
         [c3 * p(1, 3) - a3_2, a3_2, 0.0]]
    b3 = 1.5 * p(2)
    b1 = p(1) - b3
    return a, [b1, 0.0, b3], c


def _coefficients_res_5s(h):
    """Hochbruck-Ostermann 5-stage exponential RK (order 4)."""
    c2, c3, c4, c5 = 0.5, 0.5, 1.0, 0.5
    c = [0.0, c2, c3, c4, c5]
    p = _PhiCache(h, c)

    a3_2 = c3 / (2.0 * c2) * p(2, 3)
    a4_2 = (c4**2 * (c3 - c2)) / (c2 * (c4 - c2) * (c4 - c3)) * p(2, 4) if abs(c4 - c2) > 1e-10 and abs(c4 - c3) > 1e-10 else 0.0
    a4_3 = (c4**2 * (c4 - c2 - c2 * a4_2 / (c4**2 * p(2, 4) + 1e-30))) if abs(c3) > 1e-10 else 0.0
    # Simplified: use the standard Hochbruck-Ostermann tableau
    # With c2=c3=c5=0.5, c4=1.0, the coefficients simplify
    a4_3_corrected = (c4 * p(1, 4) - a4_2 * p(1, 2) - c2 * p(1, 2)) if abs(c3) > 1e-10 else 0.0
    # For the HO method, use the published tableau directly
    a5_4 = 0.5 * p(2, 5)
    a5_2 = 0.5 * p(2, 5) - a5_4
    a5_3 = 0.0

    # Recompute using the standard HO formulas
    p1 = p(1)     # phi_1(-h)
    p2 = p(2)     # phi_2(-h)
    p3 = p(3)     # phi_3(-h)

    a = [[0]*5 for _ in range(5)]
    a[1][0] = c2 * p(1, 2)
    a[2][0] = c3 * p(1, 3) - a3_2
    a[2][1] = a3_2
    # For stages 3 and 4, use the specific HO formulas
    a[3][0] = c4 * p(1, 4) - 2 * a3_2
    a[3][1] = a3_2
    a[3][2] = a3_2
    a[4][0] = c5 * p(1, 5) - (p2 - 2*p3) - a3_2
    a[4][1] = 0.5 * (p2 - 2*p3)
    a[4][2] = 0.5 * (p2 - 2*p3)
    a[4][3] = p3 - 0.5 * p2 + a3_2

    b5 = -p3 + p2 - 0.5 * p(2)
    b = [p1 - 3*p2 + 4*p3,
         0.0,
         0.0,
         -p2 + 4*p3 - 4*p3,
         4*p2 - 8*p3]
    # The published HO b-weights
    b = [p1 - 3*p2 + 4*p3,
         0.0,
         0.0,
         4*p3 - p2,
         4*p2 - 8*p3]
    # Ensure they sum correctly: sum(b) = phi(1)
    b[0] = p1 - b[1] - b[2] - b[3] - b[4]

    return a, b, c


SAMPLER_REGISTRY = {
    "res_2s":        _coefficients_res_2s,
    "res_2s_stable": _coefficients_res_2s_stable,
    "res_3s":        _coefficients_res_3s,
    "res_5s":        _coefficients_res_5s,
    "dpmpp_2s":      _coefficients_dpmpp_2s,
    "dpmpp_3s":      _coefficients_dpmpp_3s,
}

SAMPLER_NAMES = set(SAMPLER_REGISTRY.keys())


def get_coefficients(rk_type, h):
    fn = SAMPLER_REGISTRY.get(rk_type)
    if fn is None:
        raise ValueError(f"Unknown RK type: {rk_type}. Available: {sorted(SAMPLER_NAMES)}")
    return fn(h)


# -------- SA3-compatible sampler loop --------

@torch.no_grad()
def sample_res_rk(model, x, sigmas, rk_type="res_2s", callback=None, disable_tqdm=False, **extra_args):
    """Exponential RK sampler matching SA3's sampler signature.

    Args:
        model: SA3 velocity model, callable as model(x, t, **extra_args) -> v
        x: (B, C, T) initial noisy latent
        sigmas: (steps+1,) or (B, steps+1) RF timestep schedule [t_max, ..., 0]
        rk_type: which RK method to use
        callback: optional per-step callback
        **extra_args: passed to model (conditioning, CFG, etc.)
    """
    t = sigmas.to(x.device)
    per_element = t.dim() == 2
    num_steps = t.shape[-1] - 1
    ones = x.new_ones([x.shape[0]])

    for i in tqdm(range(num_steps), disable=disable_tqdm):
        if per_element:
            t_curr = t[:, i].to(x.dtype)
            t_next = t[:, i + 1].to(x.dtype)
        else:
            t_curr = (t[i] * ones).to(x.dtype)
            t_next = (t[i + 1] * ones).to(x.dtype)

        dt_abs = (t_curr - t_next).abs()
        if dt_abs.max() < 1e-10:
            continue

        t_curr_safe = t_curr.clamp(min=1e-10)
        t_next_safe = t_next.clamp(min=1e-10)
        ratio = t_next_safe / t_curr_safe       # (B,) < 1
        h = torch.log(t_curr_safe / t_next_safe) # (B,) > 0

        h_scalar = h[0].item()
        a, b, c_nodes = get_coefficients(rk_type, h_scalar)
        stages = len(c_nodes)

        x_0 = x
        eps = [None] * stages

        for s in range(stages):
            cs = c_nodes[s]
            if s == 0:
                x_s = x_0
                t_s = t_curr
            else:
                scale = (ratio ** cs).unsqueeze(-1).unsqueeze(-1)  # (B,1,1)
                h_b = h.unsqueeze(-1).unsqueeze(-1)                # (B,1,1)
                x_s = x_0 * scale
                for j in range(s):
                    if a[s][j] != 0.0:
                        x_s = x_s + h_b * a[s][j] * eps[j]
                t_s = t_curr * (ratio ** cs)

            v = model(x_s, t_s, **extra_args)
            denoised = x_s - t_s.unsqueeze(-1).unsqueeze(-1) * v
            eps[s] = denoised - x_0

        # final update
        h_b = h.unsqueeze(-1).unsqueeze(-1)
        ratio_b = ratio.unsqueeze(-1).unsqueeze(-1)
        x = x_0 * ratio_b
        for j in range(stages):
            if b[j] != 0.0:
                x = x + h_b * b[j] * eps[j]

        if callback is not None:
            callback({'x': x, 't': t_curr, 'sigma': t_curr, 'i': i, 'denoised': denoised})

    return x


# -------- SA3 registration via monkey-patch --------

_registered = False

def register_samplers():
    """Monkey-patch SA3's sample_diffusion to support RES4LYF methods."""
    global _registered
    if _registered:
        return

    try:
        import stable_audio_3.inference.sampling as sa3_sampling
    except ImportError:
        print("[res4lyf] stable_audio_3 not found, skipping sampler registration")
        return

    _orig = sa3_sampling.sample_diffusion

    def _patched(*args, **kwargs):
        sampler_type = kwargs.get("sampler_type")
        if sampler_type in SAMPLER_NAMES:
            model = args[0] if len(args) > 0 else kwargs["model"]
            noise = args[1] if len(args) > 1 else kwargs["noise"]
            sigmas = kwargs.get("sigmas", args[2] if len(args) > 2 else None)
            callback = kwargs.get("callback")
            disable_tqdm = kwargs.get("disable_tqdm", False)
            fwd = {k: v for k, v in kwargs.items()
                   if k not in ("sampler_type", "sigmas", "callback",
                                "disable_tqdm", "diffusion_objective",
                                "model", "noise")}
            return sample_res_rk(model, noise, sigmas, rk_type=sampler_type,
                                 callback=callback, disable_tqdm=disable_tqdm, **fwd)
        return _orig(*args, **kwargs)

    sa3_sampling.sample_diffusion = _patched
    _registered = True
    print(f"[res4lyf] registered {len(SAMPLER_NAMES)} samplers: {', '.join(sorted(SAMPLER_NAMES))}")
