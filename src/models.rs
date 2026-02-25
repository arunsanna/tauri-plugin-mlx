use serde::{Deserialize, Serialize};

/// Information about a loaded MLX model.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelInfo {
    /// HuggingFace repo ID (e.g. "mlx-community/gemma-3-4b-it-4bit")
    pub repo_id: String,
    /// Whether the model is currently loaded and ready for inference
    pub loaded: bool,
}

/// A single streamed token emitted during generation.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TokenEvent {
    /// The generated token text
    pub token: String,
    /// Whether this is the final token (generation complete)
    pub done: bool,
    /// Cumulative tokens generated so far
    pub tokens_generated: u32,
}

/// Download progress event emitted while fetching model weights.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DownloadProgress {
    /// Fraction completed (0.0 – 1.0)
    pub fraction: f64,
    /// HuggingFace repo ID being downloaded
    pub repo_id: String,
}

/// Final generation result returned after completion.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerationResult {
    /// Full generated text
    pub text: String,
    /// Total tokens generated
    pub tokens_generated: u32,
    /// Generation time in milliseconds
    pub generation_time_ms: u64,
    /// Tokens per second
    pub tokens_per_second: f64,
}

/// Known MLX models from the mlx-community HuggingFace org.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CatalogModel {
    /// HuggingFace repo ID
    pub repo_id: String,
    /// Human-readable display name
    pub name: String,
    /// Short description
    pub description: String,
    /// Approximate download size in bytes
    pub size_bytes: u64,
    /// Context window length
    pub context_length: usize,
}
