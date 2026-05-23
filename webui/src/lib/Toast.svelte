<script>
import { toasts } from "./toast.svelte.js";
</script>

{#if toasts.list.length > 0}
  <div class="toast-container">
    {#each toasts.list as toast (toast.id)}
      <div class="toast" class:error={toast.type === "error"} class:success={toast.type === "success"}>
        <i class="bi {toast.type === 'error' ? 'bi-exclamation-circle' : 'bi-check-circle'}"></i>
        <span>{toast.message}</span>
      </div>
    {/each}
  </div>
{/if}

<style>
.toast-container {
  position: fixed;
  top: 48px;
  right: 16px;
  z-index: 200;
  display: flex;
  flex-direction: column;
  gap: 8px;
  pointer-events: none;
}
.toast {
  padding: 10px 16px;
  border-radius: 4px;
  font-size: 12px;
  display: flex;
  align-items: center;
  gap: 8px;
  pointer-events: auto;
  animation: slideIn 0.2s ease-out;
  box-shadow: 0 4px 12px rgba(0,0,0,0.4);
}
.toast.error {
  background: #3d1f1f;
  border: 1px solid #f14c4c;
  color: #f14c4c;
}
.toast.success {
  background: #1f3d2a;
  border: 1px solid #4caf50;
  color: #4caf50;
}
@keyframes slideIn {
  from { transform: translateX(100%); opacity: 0; }
  to { transform: translateX(0); opacity: 1; }
}
</style>
