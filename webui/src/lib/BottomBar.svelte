<script>
import { session, apiGetSettings } from "./session.svelte.js";
import { fmtTime } from "./util.js";
import SettingsModal from "./SettingsModal.svelte";
import { onMount } from "svelte";

let currentTime = $derived(session.playhead * session.trackSeconds);
let totalTime = $derived(session.trackSeconds);
let settingsOpen = $state(false);

onMount(async () => {
  const s = await apiGetSettings();
  if (s?.first_run) settingsOpen = true;
});

function togglePlay() {
  session.playing = !session.playing;
}
</script>

<footer class="bottombar">
  <!-- left cluster -->
  <div class="left-cluster">
    <div class="time-display">
      <span class="time-now">{fmtTime(currentTime)}</span>
      <span class="time-sep">/</span>
      <span class="time-total">{fmtTime(totalTime)}</span>
    </div>
    <div class="volume">
      <i class="bi bi-volume-up"></i>
      <div class="vol-wrap">
        <div class="vol-fill" style="width: {session.volume * 100}%"></div>
        <input type="range" min="0" max="1" step="0.01" bind:value={session.volume} class="slider slim vol-slider">
      </div>
    </div>
    <div class="meta-info">44.1 kHz · 24-bit · Stereo</div>
  </div>

  <!-- centered transport -->
  <div class="transport" class:disabled={!session.hasAudio}>
    <button type="button" class="icon-btn" class:active={session.looping} disabled={!session.hasAudio}
      onclick={() => session.looping = !session.looping}><i class="bi bi-arrow-repeat"></i></button>
    <button type="button" class="icon-btn" disabled={!session.hasAudio}
      onclick={() => session.playhead = 0}><i class="bi bi-skip-backward-fill"></i></button>
    <button type="button" class="icon-btn play" disabled={!session.hasAudio} onclick={togglePlay}>
      <i class="bi {session.playing ? 'bi-pause-fill' : 'bi-play-fill'}"></i>
    </button>
    <button type="button" class="icon-btn" disabled={!session.hasAudio}
      onclick={() => { session.playing = false; session.playhead = 0; }}>
      <i class="bi bi-stop-fill"></i>
    </button>
    <button type="button" class="icon-btn" disabled={!session.hasAudio}
      onclick={() => session.playhead = 1}><i class="bi bi-skip-forward-fill"></i></button>
  </div>

  <!-- right cluster -->
  <div class="right-cluster">
    <button class="mode-toggle" class:active={session.advancedMode}
      onclick={() => session.advancedMode = !session.advancedMode}
      title={session.advancedMode ? "Switch to simple mode" : "Switch to advanced mode"}>
      <i class="bi {session.advancedMode ? 'bi-toggles' : 'bi-sliders'}"></i>
      {session.advancedMode ? "Advanced" : "Simple"}
    </button>
    <button class="mode-toggle" onclick={() => settingsOpen = true} title="Settings">
      <i class="bi bi-gear"></i>
    </button>
    <div class="status">
      <span class="status-dot" class:ok={session.modelLoaded}></span>
      <span class="status-label">{session.modelLoaded ? "Model ready" : "Loading…"}</span>
    </div>
    <div class="sys-stats">
      <div class="stat">
        <span class="stat-label">CPU</span><span class="stat-value">{session.stats.cpu ?? 0}%</span>
        <div class="bar"><div class="bar-fill" style="width:{session.stats.cpu ?? 0}%"></div></div>
      </div>
      <div class="stat">
        <span class="stat-label">GPU</span><span class="stat-value">{(session.stats.gpuAllocGb ?? 0).toFixed(1)} GB</span>
        <div class="bar"><div class="bar-fill" style="width:{Math.min(100, (session.stats.gpuAllocGb ?? 0) / 24 * 100)}%"></div></div>
      </div>
      <div class="stat">
        <span class="stat-label">RAM</span><span class="stat-value">{(session.stats.ramUsedGb ?? 0).toFixed(1)}/{(session.stats.ramTotalGb ?? 0).toFixed(0)}</span>
        <div class="bar"><div class="bar-fill" style="width:{session.stats.ram ?? 0}%"></div></div>
      </div>
    </div>
  </div>
</footer>

<SettingsModal bind:visible={settingsOpen} />

<style>
.bottombar {
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  align-items: center;
  padding: 0 var(--gap-4);
  border-top: 1px solid var(--border-color);
  background: var(--bg-dark);
  font-size: 11px;
  color: var(--text-secondary);
  height: var(--bottombar-h);
}
.left-cluster {
  display: flex;
  align-items: center;
  gap: var(--gap-4);
  justify-self: start;
}
.transport {
  display: flex;
  gap: var(--gap-1);
  justify-self: center;
}
.right-cluster {
  display: flex;
  align-items: center;
  gap: var(--gap-4);
  justify-self: end;
}
.icon-btn.play { color: var(--text-primary); font-size: 16px; }
.icon-btn.active { color: var(--accent-blue); }
.icon-btn[disabled] { color: var(--text-muted); cursor: default; }
.icon-btn[disabled]:hover { background: transparent; color: var(--text-muted); }
.volume { display: flex; align-items: center; gap: var(--gap-2); }
.vol-wrap { position: relative; width: 80px; height: 12px; display: flex; align-items: center; }
.vol-fill {
  position: absolute;
  left: 0; top: 50%;
  transform: translateY(-50%);
  height: 4px;
  background: var(--text-primary);
  pointer-events: none;
  border-radius: 1px;
}
.vol-slider {
  position: relative;
  max-width: 80px;
  background: var(--border-color);
}
.time-display {
  display: flex;
  gap: var(--gap-1);
  font-variant-numeric: tabular-nums;
  color: var(--text-primary);
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
}
.time-sep { color: var(--text-muted); }
.time-total { color: var(--text-secondary); }
.meta-info { color: var(--text-muted); }
.mode-toggle {
  display: flex;
  align-items: center;
  gap: var(--gap-2);
  padding: 2px 8px;
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--text-muted);
  border: 1px solid var(--accent-blue);
  border-radius: 3px;
  background: transparent;
  cursor: pointer;
}
.mode-toggle:hover { color: var(--text-secondary); }
.mode-toggle.active { color: var(--accent-blue); border-color: var(--accent-blue); }
.status-label { font-size: 10px; color: var(--text-muted); }
.status {
  display: flex;
  align-items: center;
  gap: var(--gap-2);
  margin-left: auto;
}
.status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--text-muted); display: inline-block; }
.status-dot.ok { background: var(--success-green); }
.sys-stats { display: flex; gap: var(--gap-4); }
.stat {
  display: grid;
  grid-template-columns: auto auto;
  grid-template-rows: auto auto;
  column-gap: var(--gap-2);
  align-items: center;
}
.stat-label { color: var(--text-muted); font-size: 10px; text-transform: uppercase; letter-spacing: 0.06em; }
.stat-value { color: var(--text-primary); font-size: 11px; font-variant-numeric: tabular-nums; }
.bar {
  grid-column: 1 / -1;
  height: 2px;
  background: var(--border-color);
  margin-top: 2px;
  width: 60px;
}
.bar-fill { height: 100%; background: var(--accent-blue); }
</style>
