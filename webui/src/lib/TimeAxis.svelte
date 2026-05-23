<script>
import { session } from "./session.svelte.js";

const NICE_INTERVALS = [0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300];

function fmtTickLabel(t, interval) {
  if (interval < 1) {
    const m = Math.floor(t / 60);
    const s = t - m * 60;
    return `${m}:${s.toFixed(interval < 0.5 ? 2 : 1).padStart(interval < 0.5 ? 5 : 4, '0')}`;
  }
  const m = Math.floor(t / 60);
  const s = Math.round(t - m * 60);
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

let ticks = $derived.by(() => {
  const span = session.zoomEnd - session.zoomStart;
  const visDur = span * session.trackSeconds;
  if (visDur <= 0) return [];

  // pick interval that gives ~8-15 ticks
  let interval = NICE_INTERVALS[NICE_INTERVALS.length - 1];
  for (const iv of NICE_INTERVALS) {
    if (visDur / iv <= 15) { interval = iv; break; }
  }

  const startT = session.zoomStart * session.trackSeconds;
  const endT = session.zoomEnd * session.trackSeconds;
  const firstTick = Math.ceil(startT / interval) * interval;

  const result = [];
  for (let t = firstTick; t <= endT; t += interval) {
    const norm = (t - startT) / visDur;
    result.push({ pct: norm * 100, label: fmtTickLabel(t, interval) });
  }
  return result;
});

let strip = $state(null);
let scrubbing = false;

let playheadPct = $derived.by(() => {
  const span = session.zoomEnd - session.zoomStart;
  if (span <= 0) return 0;
  const p = (session.playhead - session.zoomStart) / span;
  return Math.max(0, Math.min(1, p)) * 100;
});

let playheadInsideZoom = $derived.by(() => {
  return session.playhead >= session.zoomStart && session.playhead <= session.zoomEnd;
});

function xToTimeNorm(clientX) {
  const rect = strip.getBoundingClientRect();
  const norm = (clientX - rect.left) / rect.width;
  const span = session.zoomEnd - session.zoomStart;
  return Math.max(0, Math.min(1, session.zoomStart + norm * span));
}

function onPointerDown(e) {
  if (e.button !== 0) return;
  if (!session.hasAudio) return;
  scrubbing = true;
  session.playhead = xToTimeNorm(e.clientX);
  strip.setPointerCapture?.(e.pointerId);
}
function onPointerMove(e) {
  if (!scrubbing) return;
  session.playhead = xToTimeNorm(e.clientX);
}
function onPointerUp(e) {
  scrubbing = false;
  strip?.releasePointerCapture?.(e.pointerId);
}
</script>

<!-- outer is full-width grey strip; inner is inset to align with spectrogram below -->
<div class="time-axis">
  <div
    class="strip"
    bind:this={strip}
    onpointerdown={onPointerDown}
    onpointermove={onPointerMove}
    onpointerup={onPointerUp}
    onpointercancel={onPointerUp}
  >
    {#each ticks as t, i}
      <span class="tick-label" style="left: {t.pct}%">{t.label}</span>
      <span class="tick-mark major" style="left: {t.pct}%"></span>
      {#if i < ticks.length - 1}
        {#each [0.2, 0.4, 0.6, 0.8] as sub}
          <span class="tick-mark minor"
            style="left: {t.pct + sub * (ticks[i + 1].pct - t.pct)}%"></span>
        {/each}
      {/if}
    {/each}
    {#if session.hasAudio && !session.generating}
      <div
        class="playhead-handle"
        class:dim={!playheadInsideZoom}
        style="left: {playheadPct}%"
      >
        <div class="head"></div>
      </div>
    {/if}
  </div>
</div>

<style>
.time-axis {
  background: #0e0e0e;
  height: 100%;
  width: 100%;
  position: relative;
  z-index: 3;
}
.strip {
  position: absolute;
  left: var(--gap-2);          /* align with spectrogram below */
  right: var(--gap-2);
  top: 0;
  bottom: 0;
  cursor: text;
  user-select: none;
  touch-action: none;
}
.strip:hover { background: #141414; }

.tick-label {
  position: absolute;
  top: 6px;
  transform: translateX(-50%);
  font-size: 10px;
  color: var(--text-muted);
  font-variant-numeric: tabular-nums;
  pointer-events: none;
}
.tick-label:first-of-type { transform: translateX(0); }
.tick-label:nth-last-of-type(2) { transform: translateX(-100%); }

.tick-mark {
  position: absolute;
  bottom: 0;
  width: 1px;
  background: var(--text-muted);
  pointer-events: none;
}
.tick-mark.major { height: 8px; opacity: 0.7; }
.tick-mark.minor { height: 4px; opacity: 0.35; }

.playhead-handle {
  position: absolute;
  top: 0;
  bottom: 0;
  width: 0;
  pointer-events: none;
}
.playhead-handle .head {
  position: absolute;
  bottom: -1px;
  left: -5px;     /* nudged 8px back so the triangle's tip lines up with the playhead line */
  width: 11px;
  height: 13px;
  background: #ffffff;
  clip-path: polygon(50% 100%, 0 0, 100% 0);
  filter: drop-shadow(2px 0 0 rgba(0,0,0,0.85)) drop-shadow(-2px 0 0 rgba(0,0,0,0.85));
}
.playhead-handle.dim .head { opacity: 0.35; }
</style>
