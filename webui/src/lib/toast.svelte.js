let _toasts = $state([]);
let _id = 0;

export const toasts = {
  get list() {
    return _toasts;
  },

  show(message, type = "error", duration = 5000) {
    const id = ++_id;
    _toasts = [..._toasts, { id, message, type }];
    setTimeout(() => {
      _toasts = _toasts.filter((t) => t.id !== id);
    }, duration);
  },

  error(message) {
    this.show(message, "error");
  },
  success(message) {
    this.show(message, "success", 3000);
  },
};
