<script>
import { session, apiUpload, apiClear } from "./session.svelte.js";

let fileInput = $state(null);

async function onFile(e) {
  const file = e.target.files?.[0];
  if (!file) return;
  await apiUpload(file);
  fileInput.value = "";
}

async function onNew() {
  await apiClear();
}

function onClear() {
  session.clearMask();
}

function onSave() {
  if (!session.hasAudio) return;
  const a = document.createElement("a");
  const ts = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  a.href = `/api/audio?v=${session.version}`;
  a.download = `inpaint-${ts}.wav`;
  document.body.appendChild(a);
  a.click();
  a.remove();
}
</script>

<header class="topbar">
  <div class="brand">
    <i class="bi bi-soundwave brand-icon"></i>
    <span class="brand-name">Audio Inpainter</span>
  </div>
  <div class="topbar-actions">
    <button class="btn btn-ghost" onclick={() => fileInput.click()}>
      <i class="bi bi-folder2-open"></i> Load
    </button>
    <button class="btn btn-ghost" onclick={onSave} disabled={!session.hasAudio}>
      <i class="bi bi-download"></i> Save
    </button>
    <button class="btn btn-ghost" onclick={onNew}>
      <i class="bi bi-file-earmark-plus"></i> New
    </button>
    <button class="btn btn-ghost" onclick={onClear}>
      <i class="bi bi-x-lg"></i> Clear
    </button>
    <input type="file" accept="audio/*,.wav,.mp3,.flac"
           bind:this={fileInput} onchange={onFile} style="display: none" />
  </div>
</header>

<style>
.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 var(--gap-4);
  border-bottom: 1px solid var(--border-color);
  background: var(--bg-dark);
}
.brand { display: flex; align-items: center; gap: var(--gap-2); }
.brand-icon { color: var(--accent-blue); font-size: 18px; }
.brand-name { font-size: 14px; font-weight: 500; letter-spacing: 0.01em; }
.topbar-actions { display: flex; gap: var(--gap-1); }
.topbar-actions .btn[disabled] { color: var(--text-muted); cursor: default; }
.topbar-actions .btn[disabled]:hover { color: var(--text-muted); background: transparent; }
</style>
