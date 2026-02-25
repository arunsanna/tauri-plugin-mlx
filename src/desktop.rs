use serde::de::DeserializeOwned;
use tauri::{plugin::PluginApi, AppHandle, Runtime};

use crate::models::{GenerationResult, ModelInfo};

pub fn init<R: Runtime, C: DeserializeOwned>(
    app: &AppHandle<R>,
    _api: PluginApi<R, C>,
) -> crate::Result<Mlx<R>> {
    Ok(Mlx {
        app_handle: app.clone(),
    })
}

/// Access to the MLX inference APIs.
pub struct Mlx<R: Runtime> {
    #[allow(dead_code)]
    app_handle: AppHandle<R>,
}

impl<R: Runtime> Mlx<R> {
    pub async fn load_model(&self, repo_id: &str) -> crate::Result<()> {
        #[cfg(target_os = "macos")]
        {
            crate::macos_bridge::load_model(self.app_handle.clone(), repo_id).await
        }

        #[cfg(not(target_os = "macos"))]
        {
            let _ = repo_id;
            Err(crate::Error::Engine(
                "MLX is only available on macOS with Apple Silicon".to_string(),
            ))
        }
    }

    pub async fn unload_model(&self) {
        #[cfg(target_os = "macos")]
        {
            crate::macos_bridge::unload_model(self.app_handle.clone()).await;
        }
    }

    pub async fn generate(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> crate::Result<GenerationResult> {
        #[cfg(target_os = "macos")]
        {
            crate::macos_bridge::generate(
                self.app_handle.clone(),
                prompt,
                max_tokens,
                temperature,
            )
            .await
        }

        #[cfg(not(target_os = "macos"))]
        {
            let _ = (prompt, max_tokens, temperature);
            Err(crate::Error::Engine(
                "MLX is only available on macOS with Apple Silicon".to_string(),
            ))
        }
    }

    pub async fn is_loaded(&self) -> bool {
        #[cfg(target_os = "macos")]
        {
            crate::macos_bridge::is_loaded(self.app_handle.clone()).await
        }

        #[cfg(not(target_os = "macos"))]
        {
            false
        }
    }

    pub async fn get_model_info(&self) -> Option<ModelInfo> {
        #[cfg(target_os = "macos")]
        {
            crate::macos_bridge::get_model_info(self.app_handle.clone()).await
        }

        #[cfg(not(target_os = "macos"))]
        {
            None
        }
    }
}
