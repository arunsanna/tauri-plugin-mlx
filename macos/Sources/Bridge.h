#ifndef MLX_BRIDGE_H
#define MLX_BRIDGE_H

#include <stdbool.h>
#include <stdint.h>

/// Token streaming callback: (token_cstr, is_done, tokens_so_far, user_data)
typedef void (*mlx_token_callback)(const char *, bool, uint32_t, void *);

/// Create the MLX engine singleton. Returns opaque pointer.
void *mlx_create_engine(void);

/// Destroy the engine and release GPU memory.
void mlx_destroy_engine(void *engine);

/// Load a model by HuggingFace repo ID.  Returns true on success.
bool mlx_load_model(void *engine, const char *repo_id);

/// Unload the current model (free memory).
void mlx_unload_model(void *engine);

/// Check if a model is currently loaded.
bool mlx_is_loaded(void *engine);

/// Get the repo ID of the loaded model.  Caller must free() the result.
char *mlx_get_model_id(void *engine);

/// Generate text.  Streams tokens via callback.  Returns full text (caller must free).
char *mlx_generate(void *engine,
                         const char *prompt,
                         int32_t max_tokens,
                         float temperature,
                         mlx_token_callback callback,
                         void *user_data);

/// Get the last error message.  Returns NULL if no error.  Caller must free().
char *mlx_get_last_error(void);

#endif /* MLX_BRIDGE_H */
