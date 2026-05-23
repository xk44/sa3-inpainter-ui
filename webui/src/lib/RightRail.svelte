<script>
import { session, apiGenerate, cancelGenerate, apiSwitchModel, apiTempo, apiRedetectBpm, apiMemtokInfo, apiMemtokSet, apiMemtokAction, apiEmbeddingsList, apiEmbeddingCheckpoints, apiApplyCheckpoint, apiTrainEmbedding, apiTrainStatus, apiTrainLora, apiLoraTrainStatus, apiPreEncode, apiPreEncodeStatus, apiCheckEncoded, apiDecodeSettings, apiSetDecodeSettings } from "./session.svelte.js";
import Panel from "./Panel.svelte";

let promptCharCount = $derived(session.prompt.length);

let ctaLabel = $derived.by(() => {
  if (!session.hasAudio) return "Generate";
  if (session.hasMask) return "Inpaint";
  if (session.noise > 0) return "Vary";
  return "Generate";
});

let ctaVisible = $derived.by(() => {
  if (!session.hasAudio) return true;
  if (session.hasMask) return true;
  if (session.noise > 0) return true;
  return false;
});

let loraLibrary = $state([]);
let loraDir = $state("");
let loraPickerOpen = $state(false);
let availableLoras = $derived(
  loraLibrary.filter(l => !session.loras.some(s => s.name === l.name))
);

async function refreshLoraLibrary() {
  try {
    const r = await fetch("/api/loras");
    const j = await r.json();
    loraLibrary = (j.files || []).map(f => typeof f === "string" ? { name: f } : f);
    loraDir = j.dir || "";
  } catch (e) { console.warn("lora list failed:", e); }
}
$effect(() => { refreshLoraLibrary(); });

function addLora(name) {
  session.loras = [...session.loras, { name, strength: 0.7, enabled: true }];
  loraPickerOpen = false;
}

function togglePicker() {
  refreshLoraLibrary();
  loraPickerOpen = !loraPickerOpen;
}

function removeLora(i) {
  session.loras = session.loras.filter((_, j) => j !== i);
}

function rerollSeed() {
  session.seed = Math.floor(Math.random() * 1000000);
}

function repeatSeed() {
  if (session.lastSeed != null) session.seed = session.lastSeed;
}

async function setPrecision(e) {
  const val = e.target.value;
  try {
    const r = await fetch("/api/precision", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ precision: val }),
    });
    if (r.ok) {
      const j = await r.json();
      session.precision = j.precision;
    }
  } catch (err) { console.error("precision switch failed:", err); }
}

async function setModel(e) {
  const name = e.target.value;
  if (name === session.model) return;
  try { await apiSwitchModel(name); } catch (err) { console.error(err); }
}

async function rerollGenerate() {
  rerollSeed();
  try { await apiGenerate(); } catch (e) { console.error(e); }
}

async function clickGenerate() {
  if (session.generating) { cancelGenerate(); return; }
  try { await apiGenerate(); } catch (e) { console.error(e); }
}

let targetBpm = $state("");
let tempoProcessing = $state(false);

let tempoFactor = $derived.by(() => {
  const target = parseFloat(targetBpm);
  if (!target || !session.bpm || target <= 0) return null;
  return target / session.bpm;
});

async function applyTempo() {
  if (tempoProcessing || !tempoFactor || Math.abs(tempoFactor - 1.0) < 0.001) return;
  const target = parseFloat(targetBpm);
  tempoProcessing = true;
  try {
    await apiTempo(tempoFactor, target > 0 ? target : null);
    targetBpm = "";
  } finally {
    tempoProcessing = false;
  }
}

function setTargetFromSlider(e) {
  const val = parseFloat(e.target.value);
  if (session.bpm) targetBpm = (session.bpm * val).toFixed(1);
}

let memtokSaveName = $state("");

// checkpoint editor
let editingEmbed = $state(null);
let checkpoints = $state([]);
let selectedCkpt = $state(null);

async function openCheckpointEditor(name) {
  if (editingEmbed === name) { editingEmbed = null; return; }
  const result = await apiEmbeddingCheckpoints(name);
  checkpoints = result.checkpoints || [];
  editingEmbed = name;
  selectedCkpt = checkpoints.length > 0 ? checkpoints[checkpoints.length - 1].file : null;
}

async function applySelectedCheckpoint() {
  if (!editingEmbed || !selectedCkpt) return;
  await apiApplyCheckpoint(editingEmbed, selectedCkpt);
  editingEmbed = null;
}

// embedding training
let trainFolder = $state("");
let trainName = $state("");
let trainTokens = $state(4);
let trainSteps = $state(500);
let trainBatch = $state(0);
let trainStatus = $state("idle");
let trainStep = $state(0);
let trainLoss = $state(0);
let trainPollTimer = null;

// lora training
let loraFolder = $state("");
let loraName = $state("");
let loraCaption = $state("");
let loraTrigger = $state("");
let loraRank = $state(16);
let loraSteps = $state(1000);
let loraAdapter = $state("dora-rows");
let loraCompile = $state(false);
let loraBatch = $state(0);
let loraStatus = $state("idle");
let loraStep = $state(0);
let loraLoss = $state(0);
let loraTrainBatch = $state(1);
let loraPollTimer = null;

// pre-encode
let preEncodeStatus = $state("idle");
let preEncodeBatch = $state(0);
let preEncodeTotal = $state(0);
let hasEncoded = $state(false);
let encodedLatents = $state(0);

async function browseLoraFolder() {
  try {
    const start = loraFolder || "~";
    const r = await fetch(`/api/browse_folder?start=${encodeURIComponent(start)}`);
    if (!r.ok) return;
    const j = await r.json();
    if (j.path) loraFolder = j.path;
  } catch {}
}

async function checkEncodedStatus() {
  if (!loraName) { hasEncoded = false; encodedLatents = 0; return; }
  const j = await apiCheckEncoded(loraName);
  hasEncoded = j.has_encoded;
  encodedLatents = j.latents;
}

async function startPreEncode() {
  if (!loraFolder || !loraName) return;
  const caption = loraTrigger || loraCaption || loraName;
  const result = await apiPreEncode(loraFolder, loraName, caption);
  if (result) {
    preEncodeStatus = "running";
    preEncodeBatch = 0;
    pollPreEncode();
  }
}

async function pollPreEncode() {
  const s = await apiPreEncodeStatus();
  preEncodeStatus = s.status;
  if (s.progress?.batch !== undefined) preEncodeBatch = s.progress.batch;
  if (s.progress?.total !== undefined) preEncodeTotal = s.progress.total;
  if (s.status === "running") {
    setTimeout(pollPreEncode, 3000);
  } else if (s.status === "done") {
    await checkEncodedStatus();
  }
}

async function startLoraTraining() {
  if (!loraFolder || !loraName) return;
  const caption = loraTrigger || loraCaption || loraName;
  const result = await apiTrainLora({
    folder: loraFolder, name: loraName, caption,
    rank: loraRank, adapter_type: loraAdapter, steps: loraSteps,
    batch_size: loraBatch, use_compile: loraCompile,
  });
  if (result) {
    session.modelLoaded = false;
    loraStatus = "running";
    loraStep = 0;
    loraLoss = 0;
    loraTrainBatch = result.batch_size || 1;
    pollLoraStatus();
  }
}

async function pollLoraStatus() {
  const s = await apiLoraTrainStatus();
  loraStatus = s.status;
  if (s.progress?.step !== undefined) loraStep = s.progress.step;
  if (s.progress?.loss !== undefined) loraLoss = s.progress.loss;
  if (s.result?.step !== undefined) loraStep = s.result.step;
  if (s.status === "running") {
    loraPollTimer = setTimeout(pollLoraStatus, 3000);
  } else if (s.status === "done" || s.status === "error") {
    if (s.status === "done") {
      loraStep = Math.ceil(loraSteps / loraTrainBatch);
      loraFolder = "";
      loraName = "";
      loraTrigger = "";
      loraCaption = "";
    }
    if (s.model_reloaded) session.modelLoaded = true;
  }
}

async function browseFolder() {
  try {
    const start = trainFolder || "~";
    const r = await fetch(`/api/browse_folder?start=${encodeURIComponent(start)}`);
    if (!r.ok) return;
    const j = await r.json();
    if (j.path) trainFolder = j.path;
  } catch {}
}

async function startTraining() {
  if (!trainFolder || !trainName) return;
  const result = await apiTrainEmbedding(trainFolder, trainName, trainTokens, trainSteps, trainBatch);
  if (result) {
    trainStatus = "running";
    pollTrainStatus();
  }
}

async function pollTrainStatus() {
  const s = await apiTrainStatus();
  trainStatus = s.status;
  if (s.progress) {
    trainStep = s.progress.step ?? trainStep;
    trainLoss = s.progress.loss ?? trainLoss;
  }
  if (s.result?.step !== undefined) trainStep = s.result.step;
  if (s.result?.final_loss !== undefined) trainLoss = s.result.final_loss;
  if (s.status === "running") {
    trainPollTimer = setTimeout(pollTrainStatus, 2000);
  } else if (s.status === "done") {
    trainStep = trainSteps;
    apiEmbeddingsList();
    trainFolder = "";
    trainName = "";
  }
}

// fetch memory token info + embeddings list when model is loaded
$effect(() => {
  if (session.modelLoaded) {
    apiMemtokInfo();
    apiEmbeddingsList();
    apiDecodeSettings();
  }
});
// check if pre-encoded data exists when lora name changes
$effect(() => {
  if (loraName) checkEncodedStatus();
});
</script>

<aside class="right-rail">

  <section class="prompt-section">
    <header class="section-header"><span>Prompt</span></header>
    <textarea class="prompt-input" bind:value={session.prompt} maxlength="500"></textarea>
    <div class="prompt-meta">
      <span class="text-muted">{promptCharCount} / 500</span>
      <button class="link" onclick={() => session.prompt = ""}>Clear</button>
    </div>
    <details class="neg-prompt-section">
      <summary class="neg-prompt-toggle">Negative prompt</summary>
      <textarea class="prompt-input neg-prompt" bind:value={session.negativePrompt} placeholder="what to avoid..."></textarea>
    </details>
  </section>

  <section class="cta-panel">
    {#if ctaVisible || session.generating}
      <button class="btn btn-primary btn-lg" onclick={clickGenerate}>
        {#if session.generating}
          <i class="bi bi-stop-circle"></i> Cancel
        {:else}
          <i class="bi bi-magic"></i> {ctaLabel}
        {/if}
      </button>
      {#if !session.generating && session.hasAudio}
        <button class="btn btn-ghost btn-square" onclick={rerollGenerate} title="Reroll (new seed)">
          <i class="bi bi-dice-5"></i>
        </button>
      {/if}
    {/if}
    {#if !ctaVisible && !session.generating}
      <div class="cta-hint">paint latents or raise A2A noise to generate</div>
    {/if}
  </section>

  {#if session.variants.length > 0}
    <div class="variant-nav">
      <button class="icon-btn" disabled={session.variantIndex <= 0}
        onclick={() => session.variantIndex--}>
        <i class="bi bi-chevron-left"></i>
      </button>
      <span class="variant-count">{session.variantLabel}</span>
      <button class="icon-btn" disabled={session.variantIndex >= session.variants.length - 1}
        onclick={() => session.variantIndex++}>
        <i class="bi bi-chevron-right"></i>
      </button>
    </div>
  {/if}

  <Panel title="Generation">
    {#snippet children()}
      <div class="form-row">
        <label>
          Model
          <span class="model-dot" class:ok={session.modelLoaded && !session.switchingModel}
                title={session.switchingModel ? "switching…" : session.modelLoaded ? "model loaded" : "model not loaded"}></span>
        </label>
        <select class="select" value={session.model} onchange={setModel}
                disabled={session.switchingModel || session.generating}>
          <option value="medium">Medium (ARC)</option>
          <option value="medium-base">Medium-base (RF)</option>
          <option value="small-music">Small Music</option>
          <option value="small-sfx">Small SFX</option>
        </select>
      </div>
      {#if session.backend === "cuda"}
        <div class="form-row">
          <label>Precision</label>
          <select class="select" value={session.precision} onchange={setPrecision}>
            <option value="fp16">fp16 (fast, less VRAM)</option>
            <option value="fp32">fp32 (higher quality)</option>
          </select>
        </div>
      {/if}
      <!-- Length: only matters when generating from scratch (no source loaded) -->
      <div class="form-row" class:disabled={session.hasAudio}>
        <label>Length</label>
        <div class="slider-row">
          <input type="range" min="5" max="380" step="1" bind:value={session.duration} class="slider"
                 disabled={session.hasAudio}>
          <span class="value">{session.duration}s</span>
        </div>
      </div>
      <div class="form-row">
        <label>Steps</label>
        <div class="slider-row">
          <input type="range" min="1" max="32" bind:value={session.steps} class="slider">
          <span class="value">{session.steps}</span>
        </div>
      </div>
      <div class="form-row">
        <label>Guidance</label>
        <div class="slider-row">
          <input type="range" min="1" max="10" step="0.1" bind:value={session.cfg} class="slider">
          <span class="value">{session.cfg.toFixed(1)}</span>
        </div>
      </div>
      <!-- A2A strength: always visible, greyed when inpainting (mask present) or no source.
           when inpainting, value displays as 0 (not applied) without losing the saved setting. -->
      <div class="form-row" class:disabled={!session.hasAudio || session.hasMask}>
        <label>A2A</label>
        <div class="slider-row">
          {#if session.hasMask}
            <input type="range" min="0" max="1" step="0.01" value="0" class="slider" disabled>
            <span class="value">0.00</span>
          {:else}
            <input type="range" min="0" max="1" step="0.01" bind:value={session.noise} class="slider"
                   disabled={!session.hasAudio}
                   onpointerdown={() => session.scrubbingNoise = true}
                   onpointerup={() => session.scrubbingNoise = false}>
            <span class="value">{session.noise.toFixed(2)}</span>
          {/if}
        </div>
      </div>
      <div class="form-row">
        <label>Seed</label>
        <div class="seed-row">
          <input type="text" bind:value={session.seed} class="seed-input">
          <button class="icon-btn" onclick={repeatSeed} title="Repeat last seed"
                  disabled={session.lastSeed == null}>
            <i class="bi bi-arrow-repeat"></i>
          </button>
          <button class="icon-btn" onclick={rerollSeed} title="Random seed">
            <i class="bi bi-dice-5"></i>
          </button>
        </div>
      </div>
    {/snippet}
  </Panel>

  {#if session.advancedMode}
  <Panel title="Transform" defaultOpen={false}>
    {#snippet children()}
      <div class="form-row" class:disabled={!session.hasAudio}>
        <label>BPM</label>
        <div class="tempo-row">
          <span class="bpm-detected" title="{session.bpmSource === 'tag' ? 'From file metadata' : 'Detected'} BPM">{session.bpm ? session.bpm.toFixed(1) : "—"}</span>
          <button class="btn-icon bpm-refresh" title="Re-detect BPM (ignore metadata)" onclick={() => apiRedetectBpm()}
                  disabled={!session.hasAudio || tempoProcessing}><i class="bi bi-arrow-clockwise"></i></button>
          <i class="bi bi-arrow-right bpm-arrow"></i>
          <input type="number" class="bpm-input" bind:value={targetBpm}
                 placeholder={session.bpm ? session.bpm.toFixed(0) : "—"}
                 disabled={!session.hasAudio || tempoProcessing}
                 min="20" max="999" step="0.1">
          <button class="btn btn-sm" onclick={applyTempo}
                  disabled={!session.hasAudio || tempoProcessing || !tempoFactor || Math.abs(tempoFactor - 1.0) < 0.001}>
            {tempoProcessing ? "…" : "Apply"}
          </button>
        </div>
      </div>
      {#if session.hasAudio && session.bpm}
        <div class="form-row">
          <label>Ratio</label>
          <div class="slider-row">
            <input type="range" min="0.25" max="4.0" step="0.01" value={tempoFactor || 1.0}
                   oninput={setTargetFromSlider} class="slider"
                   disabled={tempoProcessing}>
            <span class="value">{tempoFactor ? tempoFactor.toFixed(2) + "×" : "1.00×"}</span>
          </div>
        </div>
      {/if}
    {/snippet}
  </Panel>

  <Panel title="Advanced" defaultOpen={false}>
    {#snippet children()}
      <div class="form-row">
        <label>Sampler</label>
        <select class="select" bind:value={session.samplerType}>
          <option value="">Pingpong (default)</option>
          <optgroup label="SA3 Built-in">
            <option value="euler">Euler</option>
            <option value="rk4">RK4</option>
            <option value="dpmpp">DPM++ (flow)</option>
            <option value="pingpong">Pingpong</option>
          </optgroup>
          <optgroup label="RES4LYF (Exponential)">
            <option value="res_2s">RES 2s</option>
            <option value="res_2s_stable">RES 2s Stable</option>
            <option value="res_3s">RES 3s</option>
            <option value="res_5s">RES 5s (HO4)</option>
            <option value="dpmpp_2s">DPM++ 2s (exp)</option>
            <option value="dpmpp_3s">DPM++ 3s (exp)</option>
          </optgroup>
        </select>
      </div>
      <div class="form-row">
        <label>APG</label>
        <div class="slider-row">
          <input type="range" min="0" max="2" step="0.05" bind:value={session.apgScale} class="slider">
          <span class="value">{session.apgScale.toFixed(2)}</span>
        </div>
      </div>
      <div class="form-row">
        <label>Schedule</label>
        <select class="select" bind:value={session.distShiftType}>
          <option value="default">Default</option>
          <option value="logsnr">LogSNR</option>
          <option value="flux">Flux</option>
          <option value="full">Full</option>
          <option value="none">None (linear)</option>
        </select>
      </div>
      {#if session.distShiftType === "logsnr"}
        <div class="form-row">
          <label>Anchor</label>
          <div class="slider-row">
            <input type="range" min="-10" max="0" step="0.1" bind:value={session.distShiftAnchorLogsnr} class="slider">
            <span class="value">{session.distShiftAnchorLogsnr.toFixed(1)}</span>
          </div>
        </div>
        <div class="form-row">
          <label>Rate</label>
          <div class="slider-row">
            <input type="range" min="0" max="3" step="0.1" bind:value={session.distShiftRate} class="slider">
            <span class="value">{session.distShiftRate.toFixed(1)}</span>
          </div>
        </div>
        <div class="form-row">
          <label>SNR end</label>
          <div class="slider-row">
            <input type="range" min="-2" max="5" step="0.1" bind:value={session.distShiftLogsnrEnd} class="slider">
            <span class="value">{session.distShiftLogsnrEnd.toFixed(1)}</span>
          </div>
        </div>
      {:else if session.distShiftType === "flux"}
        <div class="form-row">
          <label>α min</label>
          <div class="slider-row">
            <input type="range" min="0.1" max="20" step="0.1" bind:value={session.distShiftAlphaMin} class="slider">
            <span class="value">{session.distShiftAlphaMin.toFixed(1)}</span>
          </div>
        </div>
        <div class="form-row">
          <label>α max</label>
          <div class="slider-row">
            <input type="range" min="0.1" max="20" step="0.1" bind:value={session.distShiftAlphaMax} class="slider">
            <span class="value">{session.distShiftAlphaMax.toFixed(1)}</span>
          </div>
        </div>
      {:else if session.distShiftType === "full"}
        <div class="form-row">
          <label>Base</label>
          <div class="slider-row">
            <input type="range" min="0" max="2" step="0.05" bind:value={session.distShiftBaseShift} class="slider">
            <span class="value">{session.distShiftBaseShift.toFixed(2)}</span>
          </div>
        </div>
        <div class="form-row">
          <label>Max</label>
          <div class="slider-row">
            <input type="range" min="0" max="3" step="0.05" bind:value={session.distShiftMaxShift} class="slider">
            <span class="value">{session.distShiftMaxShift.toFixed(2)}</span>
          </div>
        </div>
      {/if}
      <div class="form-row">
        <label>φ rescale</label>
        <div class="slider-row">
          <input type="range" min="0" max="1" step="0.05" bind:value={session.scalePhi} class="slider">
          <span class="value">{session.scalePhi.toFixed(2)}</span>
        </div>
      </div>
      <div class="form-row">
        <label>CFG from</label>
        <div class="slider-row">
          <input type="range" min="0" max="1" step="0.05" bind:value={session.cfgIntervalStart} class="slider">
          <span class="value">{session.cfgIntervalStart.toFixed(2)}</span>
        </div>
      </div>
      <div class="form-row">
        <label>CFG to</label>
        <div class="slider-row">
          <input type="range" min="0" max="1" step="0.05" bind:value={session.cfgIntervalEnd} class="slider">
          <span class="value">{session.cfgIntervalEnd.toFixed(2)}</span>
        </div>
      </div>
      <div class="form-row">
        <label>CFG clip</label>
        <div class="slider-row">
          <input type="range" min="0" max="100" step="1" bind:value={session.cfgNormThreshold} class="slider">
          <span class="value">{session.cfgNormThreshold || "off"}</span>
        </div>
      </div>
      <div class="form-row">
        <label>Exit layer</label>
        <div class="slider-row">
          <input type="range" min="0" max="24" step="1" bind:value={session.exitLayerIx} class="slider">
          <span class="value">{session.exitLayerIx || "all"}</span>
        </div>
      </div>
      <div class="form-row">
        <label>Padding</label>
        <div class="slider-row">
          <input type="range" min="0" max="20" step="0.5" bind:value={session.durationPaddingSec} class="slider">
          <span class="value">{session.durationPaddingSec.toFixed(1)}s</span>
        </div>
      </div>
      <div class="subsection-label">Performance</div>
      <div class="form-row">
        <label>KV Cache</label>
        <label class="toggle-sm">
          <input type="checkbox" bind:checked={session.kvCache}>
          <span>{session.kvCache ? "On" : "Off"}</span>
        </label>
      </div>
      <div class="form-row">
        <label>ToMe Ratio</label>
        <div class="slider-row">
          <input type="range" min="0" max="0.5" step="0.05" bind:value={session.tomeRatio} class="slider">
          <span class="value">{session.tomeRatio.toFixed(2)}</span>
        </div>
      </div>

      <div class="subsection-label">VAE Decode</div>
      <div class="form-row">
        <label>FP32 Decode</label>
        <label class="toggle-sm">
          <input type="checkbox" bind:checked={session.decodeFp32}
                 onchange={() => apiSetDecodeSettings({ decode_fp32: session.decodeFp32 })}>
          <span>{session.decodeFp32 ? "On" : "Off"}</span>
        </label>
      </div>
      <div class="form-row">
        <label>Overlap</label>
        <div class="slider-row">
          <input type="range" min="0" max="128" step="8" bind:value={session.decodeOverlap} class="slider"
                 onchange={() => apiSetDecodeSettings({ decode_overlap: session.decodeOverlap })}>
          <span class="value">{session.decodeOverlap}</span>
        </div>
      </div>

      {#if session.memtokAvailable}
        <div class="subsection-label">Memory Tokens</div>
        <div class="form-row">
          <label>Strength</label>
          <div class="slider-row">
            <input type="range" min="0" max="3" step="0.05" value={session.memtokStrength}
                   oninput={(e) => apiMemtokSet(parseFloat(e.target.value))} class="slider">
            <span class="value">{session.memtokStrength.toFixed(2)}</span>
          </div>
        </div>
        <div class="form-row memtok-actions">
          <input type="text" class="bpm-input" bind:value={memtokSaveName} placeholder="preset name">
          <button class="btn btn-sm" onclick={() => { apiMemtokAction("save", memtokSaveName || "custom"); memtokSaveName = ""; }}
                  disabled={!memtokSaveName}>Save</button>
          <button class="btn btn-sm" onclick={() => apiMemtokAction("reset")}>Reset</button>
        </div>
        {#if session.memtokPresets.length > 0}
          <div class="form-row">
            <label>Presets</label>
            <div class="memtok-presets">
              {#each session.memtokPresets as preset}
                <button class="btn btn-sm" onclick={() => apiMemtokAction("load", preset)}>{preset}</button>
              {/each}
            </div>
          </div>
        {/if}
      {/if}
    {/snippet}
  </Panel>

  <Panel title="Embeddings" defaultOpen={false}>
    {#snippet children()}
      {#if session.embeddings.length > 0}
        <div class="embed-hint">Use <code>&lt;name&gt;</code> or <code>&lt;name:0.8&gt;</code> for strength</div>
        <div class="embed-list">
          {#each session.embeddings as emb}
            <div class="embed-item">
              <code>&lt;{emb.name}&gt;</code>
              <div class="embed-actions">
                {#if emb.has_checkpoints}
                  <button class="btn-icon" title="Edit checkpoint"
                          onclick={() => openCheckpointEditor(emb.name)}>
                    <i class="bi bi-pencil"></i>
                  </button>
                {/if}
                <button class="btn-icon" title="Insert into prompt"
                        onclick={() => { session.prompt += (session.prompt ? " " : "") + `<${emb.name}>`; }}>
                  <i class="bi bi-plus-circle"></i>
                </button>
              </div>
            </div>
            {#if editingEmbed === emb.name}
              <div class="ckpt-editor">
                <select class="select" bind:value={selectedCkpt}>
                  {#each checkpoints as ckpt}
                    <option value={ckpt.file}>{ckpt.step} — {ckpt.loss.toFixed(4)}</option>
                  {/each}
                </select>
                <button class="btn btn-sm" onclick={applySelectedCheckpoint}>Apply</button>
              </div>
            {/if}
          {/each}
        </div>
      {/if}
      <div class="subsection-label">Train New</div>
      <div class="form-row">
        <label>Folder</label>
        <div class="folder-row">
          <input type="text" class="input-full" bind:value={trainFolder}
                 placeholder="~/Music/artist-name" disabled={trainStatus === "running"}>
          <button class="btn btn-sm" onclick={browseFolder}
                  disabled={trainStatus === "running"} title="Browse…">
            <i class="bi bi-folder2-open"></i>
          </button>
        </div>
      </div>
      <div class="form-row">
        <label>Name</label>
        <input type="text" class="bpm-input" bind:value={trainName}
               placeholder="trigger-name" disabled={trainStatus === "running"}
               style="width: 100%">
      </div>
      <div class="form-row">
        <label>Tokens</label>
        <div class="slider-row">
          <input type="range" min="1" max="16" step="1" bind:value={trainTokens} class="slider"
                 disabled={trainStatus === "running"}>
          <span class="value">{trainTokens}</span>
        </div>
      </div>
      <div class="form-row">
        <label>Steps</label>
        <div class="slider-row">
          <input type="range" min="100" max="2000" step="50" bind:value={trainSteps} class="slider"
                 disabled={trainStatus === "running"}>
          <span class="value">{trainSteps}</span>
        </div>
      </div>
      <div class="form-row">
        <label>Batch</label>
        <select class="select" bind:value={trainBatch} disabled={trainStatus === "running"}>
          <option value={0}>Auto</option>
          <option value={1}>1</option>
          <option value={2}>2</option>
          <option value={4}>4</option>
          <option value={8}>8</option>
        </select>
      </div>
      <button class="btn btn-sm train-btn"
              onclick={startTraining}
              disabled={!trainFolder || !trainName || trainStatus === "running"}>
        {#if trainStatus === "running"}
          <i class="bi bi-hourglass-split"></i> Training…
        {:else}
          <i class="bi bi-lightning"></i> Train Embedding
        {/if}
      </button>
      {#if trainStatus === "running" || trainStatus === "done"}
        <div class="train-progress">
          <div class="train-progress-bar">
            <div class="train-progress-fill" style="width: {Math.min(100, trainStep / trainSteps * 100)}%"></div>
          </div>
          <span class="train-progress-text">{trainStep}/{trainSteps} · loss {trainLoss.toFixed(4)}</span>
        </div>
      {/if}
    {/snippet}
  </Panel>
  {/if}

  <section class="loras-section">
    <header class="loras-header">
      <span>LoRAs</span>
      <button class="icon-btn" onclick={togglePicker} title="Add LoRA">
        <i class="bi bi-plus"></i>
      </button>
    </header>
    {#if loraPickerOpen}
      <div class="lora-picker">
        {#if availableLoras.length}
          {#each availableLoras as lora}
            <button class="picker-item" onclick={() => addLora(lora.name)}>
              {lora.name}
              {#if lora.loss != null}<span class="lora-loss">loss {lora.loss}</span>{/if}
            </button>
          {/each}
        {:else}
          <span class="picker-empty">no loras in {loraDir || "library"}</span>
        {/if}
      </div>
    {/if}
    <div class="loras-box">
      {#each session.loras as lora, i}
        <div class="lora-card">
          <div class="lora-head">
            <span class="lora-name">{lora.name}</span>
            <button class="icon-btn" onclick={() => removeLora(i)}>
              <i class="bi bi-x"></i>
            </button>
          </div>
          <div class="slider-row">
            <input type="range" min="0" max="1" step="0.01" bind:value={lora.strength} class="slider">
            <span class="value">{lora.strength.toFixed(2)}</span>
          </div>
        </div>
      {:else}
        <div class="lora-empty">drop a .safetensors here or click + to add</div>
      {/each}
    </div>
    <div class="subsection-label" style="padding: 0 var(--gap-2);">Train LoRA</div>
    <div style="padding: 0 var(--gap-2);">
      <div class="form-row">
        <label>Folder</label>
        <div class="folder-row">
          <input type="text" class="input-full" bind:value={loraFolder}
                 placeholder="~/Music/artist-name" disabled={loraStatus === "running"}>
          <button class="btn btn-sm" onclick={browseLoraFolder}
                  disabled={loraStatus === "running"} title="Browse…">
            <i class="bi bi-folder2-open"></i>
          </button>
        </div>
      </div>
      <div class="form-row">
        <label>Name</label>
        <input type="text" class="bpm-input" bind:value={loraName}
               placeholder="lora-name" disabled={loraStatus === "running"} style="width: 100%">
      </div>
      <div class="form-row">
        <label>Trigger</label>
        <input type="text" class="input-full" bind:value={loraTrigger}
               placeholder="single word (overrides caption)" disabled={loraStatus === "running"}>
      </div>
      <div class="form-row">
        <label>Caption</label>
        <input type="text" class="input-full" bind:value={loraCaption}
               placeholder="style description (optional)" disabled={loraStatus === "running" || !!loraTrigger}>
      </div>
      <div class="form-row">
        <label>Rank</label>
        <select class="select" bind:value={loraRank} disabled={loraStatus === "running"}>
          <option value={4}>4</option>
          <option value={8}>8</option>
          <option value={16}>16</option>
          <option value={32}>32</option>
        </select>
      </div>
      <div class="form-row">
        <label>Adapter</label>
        <select class="select" bind:value={loraAdapter} disabled={loraStatus === "running"}>
          <option value="dora-rows">DoRA (recommended)</option>
          <option value="lora">LoRA</option>
          <option value="lora-xs">LoRA-XS (low VRAM)</option>
          <option value="bora">BoRA</option>
        </select>
      </div>
      <div class="form-row">
        <label>Total steps</label>
        <div class="slider-row">
          <input type="range" min="500" max="10000" step="500" bind:value={loraSteps} class="slider"
                 disabled={loraStatus === "running"}>
          <span class="value">{loraSteps}</span>
        </div>
      </div>
      <div class="form-row">
        <label>Batch</label>
        <select class="select" bind:value={loraBatch} disabled={loraStatus === "running"}>
          <option value={0}>Auto</option>
          <option value={1}>1</option>
          <option value={2}>2</option>
          <option value={4}>4</option>
          <option value={8}>8</option>
        </select>
      </div>
      <div class="form-row">
        <label>Compile</label>
        <label class="toggle-sm">
          <input type="checkbox" bind:checked={loraCompile}
                 disabled={loraStatus === "running"}>
          <span>{loraCompile ? "On" : "Off"}</span>
        </label>
      </div>
      <div class="form-row">
        <label>Pre-encode</label>
        <div class="pre-encode-row">
          {#if hasEncoded}
            <span class="encoded-badge">{encodedLatents} latents cached</span>
          {:else if preEncodeStatus === "running"}
            <span class="text-muted" style="font-size:11px">Encoding {preEncodeBatch}/{preEncodeTotal}…</span>
          {:else}
            <span class="text-muted" style="font-size:11px">Not cached</span>
          {/if}
          <button class="btn btn-sm" onclick={startPreEncode}
                  disabled={!loraFolder || !loraName || preEncodeStatus === "running" || loraStatus === "running"}>
            {preEncodeStatus === "running" ? "…" : hasEncoded ? "Re-encode" : "Encode"}
          </button>
        </div>
      </div>
      <button class="btn btn-sm train-btn"
              onclick={startLoraTraining}
              disabled={!loraFolder || !loraName || loraStatus === "running"}>
        {#if loraStatus === "running"}
          <i class="bi bi-hourglass-split"></i> Training LoRA…
        {:else}
          <i class="bi bi-lightning"></i> Train LoRA
        {/if}
      </button>
      {#if loraStatus === "running" || loraStatus === "done"}
        <div class="train-progress">
          <div class="train-progress-bar">
            <div class="train-progress-fill" style="width: {Math.min(100, loraStep * loraTrainBatch / loraSteps * 100)}%"></div>
          </div>
          <span class="train-progress-text">{loraStep * loraTrainBatch}/{loraSteps} · loss {loraLoss.toFixed(4)}</span>
        </div>
      {/if}
    </div>
  </section>

</aside>

<style>
.right-rail {
  background: var(--bg-lighter);
  border-left: 1px solid var(--border-color);
  display: flex;
  flex-direction: column;
  overflow-y: auto;
}
.form-row {
  display: grid;
  grid-template-columns: 70px 1fr;
  align-items: center;
  gap: var(--gap-3);
}
.form-row label {
  font-size: 11px;
  color: var(--text-secondary);
  letter-spacing: 0.04em;
  text-transform: uppercase;
}
.form-row.disabled label,
.form-row.disabled .value {
  color: var(--text-muted);
}
.model-dot {
  display: inline-block;
  margin-left: 6px;
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--error-red);
  vertical-align: middle;
}
.model-dot.ok { background: var(--success-green); }
.form-row.disabled .slider { opacity: 0.4; pointer-events: none; }
.prompt-section {
  padding: 0 var(--gap-4) var(--gap-2);
  display: flex;
  flex-direction: column;
  gap: var(--gap-2);
}
.section-header {
  padding: var(--gap-3) 0;
  font-size: 12px;
  font-weight: 500;
  color: var(--text-primary);
}
.prompt-input {
  background: var(--code-block);
  border: 1px solid var(--border-color);
  color: var(--text-primary);
  padding: var(--gap-3);
  resize: vertical;
  min-height: 64px;
  font-size: 13px;
  width: 100%;
}
.prompt-input:focus { outline: 1px solid var(--accent-blue); border-color: transparent; }
.prompt-meta { display: flex; justify-content: space-between; align-items: center; font-size: 11px; }
.link { color: var(--accent-blue); font-size: 11px; }
.link:hover { color: var(--text-primary); }

.slider-row { display: flex; align-items: center; gap: var(--gap-3); }
.slider-row .value {
  font-variant-numeric: tabular-nums;
  font-size: 11px;
  color: var(--text-primary);
  min-width: 32px;
  text-align: right;
}
.select, .seed-input {
  background: var(--code-block);
  border: 1px solid var(--border-color);
  color: var(--text-primary);
  padding: 6px var(--gap-2);
  font-size: 12px;
  appearance: none;
  width: 100%;
}
.seed-input {
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  font-variant-numeric: tabular-nums;
  flex: 1;
  padding: 4px var(--gap-2);
}
.seed-row { display: flex; align-items: center; gap: var(--gap-1); }

.loras-section {
  padding: 0 var(--gap-4) var(--gap-2);
  display: flex;
  flex-direction: column;
  gap: var(--gap-2);
}
.loras-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: var(--gap-3) 0;
  font-size: 12px;
  font-weight: 500;
  color: var(--text-primary);
}
.loras-box {
  border: 1px dashed var(--border-color);
  border-radius: 4px;
  padding: var(--gap-2);
  display: flex;
  flex-direction: column;
  gap: var(--gap-2);
  min-height: 56px;
}
.lora-empty {
  color: var(--text-muted);
  font-size: 11px;
  text-align: center;
  padding: var(--gap-3);
  font-style: italic;
}
.lora-card {
  background: var(--code-block);
  border: 1px solid var(--border-color);
  border-radius: 3px;
  padding: var(--gap-2) var(--gap-3);
  display: flex;
  flex-direction: column;
  gap: var(--gap-2);
}
.lora-head { display: flex; justify-content: space-between; align-items: center; }
.lora-name { font-size: 12px; color: var(--text-primary); }

.lora-picker {
  background: var(--code-block);
  border: 1px solid var(--border-color);
  max-height: 200px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
}
.picker-item {
  text-align: left;
  padding: 6px var(--gap-3);
  font-size: 12px;
  color: var(--text-primary);
  background: transparent;
  border: 0;
}
.picker-item:hover { background: var(--code-highlight); color: var(--accent-blue); }
.lora-loss { margin-left: auto; font-size: 10px; color: var(--text-muted); font-variant-numeric: tabular-nums; }
.picker-empty {
  padding: 6px var(--gap-3);
  font-size: 11px;
  color: var(--text-muted);
  font-style: italic;
}

.cta-panel {
  display: flex;
  gap: var(--gap-2);
  padding: 0 var(--gap-4) var(--gap-4);
  align-items: stretch;
}
.cta-panel .btn-primary { height: 36px; padding: 0 var(--gap-3); border-radius: 4px; border: 0; }
.cta-panel .btn-square { border-radius: 4px; }
.cta-hint { padding: 0 var(--gap-4) var(--gap-4); font-size: 11px; color: var(--text-muted); font-style: italic; text-align: center; }
.variant-nav {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: var(--gap-3);
  padding: var(--gap-3);
  border-bottom: 1px solid var(--border-color);
}
.variant-count {
  font-size: 12px;
  font-variant-numeric: tabular-nums;
  color: var(--text-secondary);
  min-width: 48px;
  text-align: center;
}
.neg-prompt-section {
  margin-top: var(--gap-1);
}
.neg-prompt-toggle {
  font-size: 11px;
  color: var(--text-muted);
  cursor: pointer;
  user-select: none;
  padding: var(--gap-1) 0;
  letter-spacing: 0.04em;
}
.neg-prompt-toggle:hover { color: var(--text-secondary); }
.neg-prompt { margin-top: var(--gap-2); min-height: 48px; }

.tempo-row {
  display: flex;
  align-items: center;
  gap: var(--gap-2);
}
.bpm-detected {
  font-size: 12px;
  font-variant-numeric: tabular-nums;
  color: var(--text-secondary);
  min-width: 36px;
}
.bpm-refresh {
  background: none;
  border: none;
  color: var(--text-muted);
  cursor: pointer;
  padding: 0 2px;
  font-size: 11px;
  line-height: 1;
}
.bpm-refresh:hover { color: var(--accent-blue); }
.bpm-refresh[disabled] { opacity: 0.3; cursor: default; }
.bpm-arrow {
  font-size: 10px;
  color: var(--text-muted);
}
.bpm-input {
  background: var(--code-block);
  border: 1px solid var(--border-color);
  color: var(--text-primary);
  padding: 3px 6px;
  font-size: 12px;
  font-variant-numeric: tabular-nums;
  width: 60px;
  text-align: center;
}
.bpm-input:focus { outline: 1px solid var(--accent-blue); border-color: transparent; }
.btn-sm {
  font-size: 11px;
  padding: 2px 8px;
  border: 1px solid var(--border-color);
  background: var(--code-block);
  color: var(--text-primary);
  border-radius: 3px;
  white-space: nowrap;
}
.btn-sm:hover:not(:disabled) { background: var(--code-highlight); }
.btn-sm:disabled { opacity: 0.4; }
.subsection-label {
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-muted);
  margin-top: var(--gap-3);
  padding-bottom: var(--gap-1);
  border-bottom: 1px solid var(--border-color);
}
.toggle-sm {
  display: flex;
  align-items: center;
  gap: var(--gap-2);
  cursor: pointer;
  font-size: 11px;
  color: var(--text-secondary);
}
.toggle-sm input { margin: 0; }
.memtok-actions {
  display: flex;
  align-items: center;
  gap: var(--gap-2);
}
.memtok-presets {
  display: flex;
  flex-wrap: wrap;
  gap: var(--gap-1);
}
.embed-hint {
  font-size: 11px;
  color: var(--text-muted);
  padding: 0 var(--gap-2);
}
.embed-hint code { color: var(--accent-blue); }
.embed-list { display: flex; flex-direction: column; gap: var(--gap-1); padding: var(--gap-2); }
.embed-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 12px;
}
.embed-item code { color: var(--accent-blue); font-size: 11px; }
.embed-actions { display: flex; gap: var(--gap-1); align-items: center; }
.ckpt-editor {
  display: flex;
  gap: var(--gap-2);
  align-items: center;
  padding-left: var(--gap-3);
}
.ckpt-editor .select {
  flex: 1;
  background: var(--code-block);
  border: 1px solid var(--border-color);
  color: var(--text-primary);
  padding: 2px 4px;
  font-size: 11px;
  min-width: 0;
}
.input-full {
  background: var(--code-block);
  border: 1px solid var(--border-color);
  color: var(--text-primary);
  padding: 3px 6px;
  font-size: 12px;
  width: 100%;
}
.input-full:focus { outline: 1px solid var(--accent-blue); border-color: transparent; }
.folder-row {
  display: flex;
  gap: var(--gap-1);
  align-items: center;
  width: 100%;
}
.folder-row .input-full { flex: 1; min-width: 0; }
.train-btn {
  width: 100%;
  margin-top: var(--gap-2);
  display: flex;
  align-items: center;
  justify-content: center;
  gap: var(--gap-2);
}
.train-progress {
  margin-top: var(--gap-2);
  display: flex;
  flex-direction: column;
  gap: 3px;
}
.train-progress-bar {
  height: 4px;
  background: var(--code-block);
  border-radius: 2px;
  overflow: hidden;
}
.train-progress-fill {
  height: 100%;
  background: var(--accent-blue);
  transition: width 0.3s ease;
}
.train-progress-text {
  font-size: 10px;
  color: var(--text-secondary);
  font-variant-numeric: tabular-nums;
}
.pre-encode-row {
  display: flex;
  align-items: center;
  gap: var(--gap-2);
  justify-content: space-between;
}
.encoded-badge {
  font-size: 10px;
  color: var(--success-green);
  font-variant-numeric: tabular-nums;
}
</style>
