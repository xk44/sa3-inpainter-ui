<script>
import { apiGetSettings, apiSaveSettings } from "./session.svelte.js";

let { visible = $bindable(false) } = $props();

let settings = $state({
  models_dir: "",
  lora_dir: "",
  lora_train_dir: "",
  embeddings_dir: "",
  sa3_root: "",
  hf_token: "",
});
let loading = $state(false);
let firstRun = $state(false);

const pathFields = [
  { key: "models_dir", label: "Models directory", hint: "Where SA3 model weights are stored" },
  { key: "lora_dir", label: "LoRA directory", hint: "Where trained LoRA .safetensors are saved" },
  { key: "lora_train_dir", label: "LoRA training directory", hint: "Working directory for LoRA training runs" },
  { key: "embeddings_dir", label: "Embeddings directory", hint: "Textual inversion embeddings" },
  { key: "sa3_root", label: "SA3 source root", hint: "Path to stable-audio-3 repo clone" },
];

async function load() {
  loading = true;
  const s = await apiGetSettings();
  if (s) {
    firstRun = !!s.first_run;
    delete s.first_run;
    Object.assign(settings, s);
  }
  loading = false;
}

async function save() {
  const result = await apiSaveSettings(settings);
  if (result) {
    Object.assign(settings, result);
    visible = false;
  }
}

function onBackdropClick() { visible = false; }
function onCardClick(e) { e.stopPropagation(); }
function onKeyDown(e) {
  if (!visible) return;
  if (e.key === "Escape") { e.preventDefault(); visible = false; }
}

$effect(() => { if (visible) load(); });
</script>

<svelte:window onkeydown={onKeyDown} />

{#if visible}
  <!-- svelte-ignore a11y_click_events_have_key_events -->
  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <div class="backdrop" onclick={onBackdropClick}>
    <div class="card" onclick={onCardClick} role="dialog" aria-modal="true" aria-label="Settings">
      <div class="header">
        <span class="title">Settings</span>
        <button class="close-btn" onclick={() => visible = false} aria-label="Close">&times;</button>
      </div>
      {#if loading}
        <div class="loading">Loading...</div>
      {:else}
        {#if firstRun}
          <div class="first-run">Configure your paths before first use. These can be changed later from the gear icon.</div>
        {/if}
        <div class="fields">
          {#each pathFields as { key, label, hint }}
            <label class="field">
              <span class="field-label">{label}</span>
              <input type="text" class="field-input" bind:value={settings[key]}
                     placeholder={hint}>
              <span class="field-hint">{hint}</span>
            </label>
          {/each}
          <label class="field">
            <span class="field-label">HuggingFace token</span>
            <input type="password" class="field-input" bind:value={settings.hf_token}
                   placeholder="hf_...">
            <span class="field-hint">Required for gated models (training, pre-encode)</span>
          </label>
        </div>
        <div class="actions">
          <button class="btn cancel" onclick={() => visible = false}>Cancel</button>
          <button class="btn save" onclick={save}>Save</button>
        </div>
      {/if}
    </div>
  </div>
{/if}

<style>
.backdrop {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.72);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 500;
  backdrop-filter: blur(2px);
}
.card {
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: 8px;
  padding: var(--gap-4);
  min-width: 480px;
  max-width: 600px;
  width: 100%;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6);
}
.header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: var(--gap-3);
  padding-bottom: var(--gap-2);
  border-bottom: 1px solid var(--border-color);
}
.title {
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--text-muted);
}
.close-btn {
  background: none;
  border: none;
  color: var(--text-muted);
  font-size: 14px;
  cursor: pointer;
  padding: 2px 6px;
  border-radius: 4px;
  line-height: 1;
}
.close-btn:hover {
  color: var(--text-primary);
  background: rgba(255, 255, 255, 0.06);
}
.loading {
  color: var(--text-muted);
  font-size: 12px;
  padding: var(--gap-3) 0;
  text-align: center;
}
.first-run {
  font-size: 12px;
  color: var(--accent-blue);
  background: rgba(100, 160, 255, 0.08);
  border: 1px solid rgba(100, 160, 255, 0.2);
  border-radius: 4px;
  padding: 8px 10px;
  margin-bottom: var(--gap-3);
}
.fields {
  display: flex;
  flex-direction: column;
  gap: var(--gap-3);
}
.field {
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.field-label {
  font-size: 11px;
  font-weight: 600;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.field-input {
  background: var(--bg-dark);
  border: 1px solid var(--border-color);
  border-radius: 4px;
  padding: 6px 8px;
  color: var(--text-primary);
  font-size: 12px;
  font-family: ui-monospace, "JetBrains Mono", "Fira Mono", monospace;
}
.field-input:focus {
  outline: none;
  border-color: var(--accent-blue);
}
.field-hint {
  font-size: 10px;
  color: var(--text-muted);
}
.actions {
  display: flex;
  justify-content: flex-end;
  gap: var(--gap-2);
  margin-top: var(--gap-4);
  padding-top: var(--gap-2);
  border-top: 1px solid var(--border-color);
}
.btn {
  padding: 5px 16px;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 600;
  cursor: pointer;
  border: 1px solid var(--border-color);
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.cancel {
  background: transparent;
  color: var(--text-muted);
}
.cancel:hover { color: var(--text-secondary); }
.save {
  background: var(--accent-blue);
  color: #fff;
  border-color: var(--accent-blue);
}
.save:hover { filter: brightness(1.1); }
</style>
