use serde::{Deserialize, Serialize};
use tauri::{command, AppHandle, Runtime};

use crate::models::{CatalogModel, GenerationResult, ModelInfo};
use crate::MlxExt;
use crate::Result;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LoadModelRequest {
    /// HuggingFace repo ID (e.g. "mlx-community/gemma-3-4b-it-4bit")
    pub repo_id: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LoadModelResponse {
    pub status: String,
    pub repo_id: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UnloadModelResponse {
    pub status: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerateRequest {
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
}

fn default_max_tokens() -> usize {
    2048
}
fn default_temperature() -> f32 {
    0.7
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct IsLoadedResponse {
    pub loaded: bool,
    pub repo_id: Option<String>,
}

#[command]
pub(crate) async fn load_model<R: Runtime>(
    app: AppHandle<R>,
    request: LoadModelRequest,
) -> Result<LoadModelResponse> {
    app.mlx().load_model(&request.repo_id).await?;
    Ok(LoadModelResponse {
        status: "loaded".to_string(),
        repo_id: request.repo_id,
    })
}

#[command]
pub(crate) async fn unload_model<R: Runtime>(
    app: AppHandle<R>,
) -> Result<UnloadModelResponse> {
    app.mlx().unload_model().await;
    Ok(UnloadModelResponse {
        status: "unloaded".to_string(),
    })
}

#[command]
pub(crate) async fn generate<R: Runtime>(
    app: AppHandle<R>,
    request: GenerateRequest,
) -> Result<GenerationResult> {
    app.mlx()
        .generate(&request.prompt, request.max_tokens, request.temperature)
        .await
}

#[command]
pub(crate) async fn is_loaded<R: Runtime>(
    app: AppHandle<R>,
) -> Result<IsLoadedResponse> {
    let info = app.mlx().get_model_info().await;
    Ok(IsLoadedResponse {
        loaded: info.is_some(),
        repo_id: info.map(|i| i.repo_id),
    })
}

#[command]
pub(crate) async fn get_model_info<R: Runtime>(
    app: AppHandle<R>,
) -> Result<Option<ModelInfo>> {
    Ok(app.mlx().get_model_info().await)
}

#[command]
pub(crate) async fn list_models<R: Runtime>(
    _app: AppHandle<R>,
) -> Result<Vec<CatalogModel>> {
    Ok(crate::catalog::default_catalog())
}
