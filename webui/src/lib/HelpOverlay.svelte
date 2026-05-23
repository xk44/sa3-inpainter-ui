<script>
  let { visible = $bindable(false) } = $props();

  export function toggle() {
    visible = !visible;
  }

  const shortcuts = [
    { key: "Space",          desc: "Play / Pause" },
    { key: "Ctrl+G",         desc: "Generate / Inpaint / Vary" },
    { key: "R",              desc: "Reroll (random seed + generate)" },
    { key: "Ctrl+Z",         desc: "Undo (mask or audio)" },
    { key: "Ctrl+Shift+Z",   desc: "Redo (mask or audio)" },
    { key: "Ctrl+A",         desc: "Select all (paint full mask)" },
    { key: "Ctrl+Shift+A",   desc: "Clear mask" },
    { key: "← / →",          desc: "Nudge playhead by one latent" },
    { key: "Scroll",         desc: "Zoom (anchored at cursor)" },
    { key: "Shift+Scroll",   desc: "Pan" },
    { key: "Click+Drag",     desc: "Paint inpaint region" },
    { key: "Shift+Drag",     desc: "Erase inpaint region" },
    { key: "Click",          desc: "Seek playhead (no drag)" },
    { key: "?",              desc: "Toggle this help" },
  ];

  function onBackdropClick() {
    visible = false;
  }

  function onCardClick(e) {
    e.stopPropagation();
  }

  function onKeyDown(e) {
    if (!visible) return;
    if (e.key === "?" || e.key === "Escape") {
      e.preventDefault();
      visible = false;
    }
  }
</script>

<svelte:window onkeydown={onKeyDown} />

{#if visible}
  <!-- svelte-ignore a11y_click_events_have_key_events -->
  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <div class="backdrop" onclick={onBackdropClick}>
    <div class="card" onclick={onCardClick} role="dialog" aria-modal="true" aria-label="Keyboard shortcuts">
      <div class="header">
        <span class="title">Keyboard Shortcuts</span>
        <button class="close-btn" onclick={() => visible = false} aria-label="Close">✕</button>
      </div>
      <div class="grid">
        {#each shortcuts as { key, desc }}
          <kbd>{key}</kbd>
          <span class="desc">{desc}</span>
        {/each}
      </div>
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
  min-width: 420px;
  max-width: 540px;
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
  transition: color 0.15s, background 0.15s;
}
.close-btn:hover {
  color: var(--text-primary);
  background: rgba(255, 255, 255, 0.06);
}

.grid {
  display: grid;
  grid-template-columns: auto 1fr;
  column-gap: var(--gap-3);
  row-gap: var(--gap-2);
  align-items: center;
}

kbd {
  font-family: ui-monospace, "JetBrains Mono", "Fira Mono", monospace;
  font-size: 11px;
  background: var(--bg-dark);
  border: 1px solid var(--border-color);
  border-bottom-width: 2px;
  border-radius: 4px;
  padding: 2px 7px;
  color: var(--accent-blue);
  white-space: nowrap;
  justify-self: start;
  line-height: 1.6;
  letter-spacing: 0.02em;
}

.desc {
  font-size: 12px;
  color: var(--text-secondary);
  line-height: 1.5;
}
</style>
