use crate::models::CatalogModel;

/// Default catalog of known MLX models from mlx-community on HuggingFace.
///
/// These are pre-converted 4-bit quantized models optimized for Apple Silicon.
/// Sizes are approximate and based on the full repo download (weights + tokenizer).
pub fn default_catalog() -> Vec<CatalogModel> {
    vec![
        CatalogModel {
            repo_id: "mlx-community/gemma-3-1b-it-4bit".to_string(),
            name: "Gemma 3 1B".to_string(),
            description: "Google's ultra-fast 1B, instant responses".to_string(),
            size_bytes: 629_145_600, // ~600 MB
            context_length: 32768,
        },
        CatalogModel {
            repo_id: "mlx-community/Llama-3.2-3B-Instruct-4bit".to_string(),
            name: "Llama 3.2 3B".to_string(),
            description: "Meta's compact model, fast on any Mac".to_string(),
            size_bytes: 1_825_361_100, // ~1.7 GB
            context_length: 131072,
        },
        CatalogModel {
            repo_id: "mlx-community/gemma-3-4b-it-4bit".to_string(),
            name: "Gemma 3 4B".to_string(),
            description: "Google's best balance of speed and quality".to_string(),
            size_bytes: 2_254_857_830, // ~2.1 GB
            context_length: 128000,
        },
        CatalogModel {
            repo_id: "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit".to_string(),
            name: "Llama 3.1 8B".to_string(),
            description: "Meta's workhorse, strong analysis and reasoning".to_string(),
            size_bytes: 4_724_464_640, // ~4.4 GB
            context_length: 131072,
        },
        CatalogModel {
            repo_id: "mlx-community/gemma-3-12b-it-4bit".to_string(),
            name: "Gemma 3 12B".to_string(),
            description: "Google's high-quality 12B, excellent reasoning".to_string(),
            size_bytes: 6_979_321_856, // ~6.5 GB
            context_length: 128000,
        },
        CatalogModel {
            repo_id: "mlx-community/gemma-3-27b-it-4bit".to_string(),
            name: "Gemma 3 27B".to_string(),
            description: "Google's best, near-cloud quality, 32GB+ RAM".to_string(),
            size_bytes: 15_032_385_536, // ~14 GB
            context_length: 128000,
        },
    ]
}
