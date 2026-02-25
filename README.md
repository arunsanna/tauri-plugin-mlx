# tauri-plugin-mlx

Local LLM inference on Apple Silicon for Tauri v2, powered by [Apple MLX](https://github.com/ml-explore/mlx-swift).

**21-87% faster than llama.cpp** on Apple Silicon — MLX talks directly to Metal and unified memory instead of going through a generic compute path.

## Features

- Native Metal GPU acceleration via MLX (no llama.cpp)
- Streaming token generation with Tauri event system
- Automatic model download and caching from HuggingFace
- Built-in catalog of popular quantized models
- Zero configuration — just pick a model and go

## Requirements

- macOS 14+ (Sonoma) on Apple Silicon (M1/M2/M3/M4/M5)
- Tauri v2
- Xcode (for Swift compilation during build)

## Installation

Add to your `src-tauri/Cargo.toml`:

```toml
[dependencies]
tauri-plugin-mlx = { git = "https://github.com/arunsanna/tauri-plugin-mlx" }
```

Register the plugin in your Tauri app:

```rust
fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_mlx::init())
        .run(tauri::generate_context!())
        .expect("error while running application");
}
```

## Usage (TypeScript)

```typescript
import { loadModel, generate, onToken, listModels } from "tauri-plugin-mlx-api";

// List available models
const models = await listModels();

// Load a model (downloads on first use)
await loadModel("mlx-community/gemma-3-4b-it-4bit");

// Stream tokens as they generate
const unsub = await onToken((event) => {
  process.stdout.write(event.token);
  if (event.done) console.log("\n");
});

// Generate text
const result = await generate("Explain quantum computing in one sentence", {
  maxTokens: 256,
  temperature: 0.7,
});

console.log(`${result.tokensPerSecond.toFixed(1)} tok/s`);
unsub();
```

## Usage (Rust)

```rust
use tauri_plugin_mlx::MlxExt;

// Inside a Tauri command
async fn my_command(app: tauri::AppHandle) -> Result<String, String> {
    app.mlx().load_model("mlx-community/gemma-3-4b-it-4bit").await
        .map_err(|e| e.to_string())?;

    let result = app.mlx().generate("Hello!", 256, 0.7).await
        .map_err(|e| e.to_string())?;

    Ok(result.text)
}
```

## Built-in Model Catalog

| Model        | Repo                                            | Size    | Context |
| ------------ | ----------------------------------------------- | ------- | ------- |
| Gemma 3 1B   | `mlx-community/gemma-3-1b-it-4bit`              | ~600 MB | 32K     |
| Llama 3.2 3B | `mlx-community/Llama-3.2-3B-Instruct-4bit`      | ~1.7 GB | 128K    |
| Gemma 3 4B   | `mlx-community/gemma-3-4b-it-4bit`              | ~2.1 GB | 128K    |
| Llama 3.1 8B | `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` | ~4.4 GB | 128K    |
| Gemma 3 12B  | `mlx-community/gemma-3-12b-it-4bit`             | ~6.5 GB | 128K    |
| Gemma 3 27B  | `mlx-community/gemma-3-27b-it-4bit`             | ~14 GB  | 128K    |

You can also load any model from the [mlx-community](https://huggingface.co/mlx-community) HuggingFace org (4,000+ models) by passing the repo ID directly.

## Architecture

```
TypeScript → Tauri invoke() → Rust commands → C FFI → Swift MLXEngine → mlx-swift-lm → Metal GPU
                                                ↑
                                          extern "C" / @_cdecl
                                     (same pattern as tauri-plugin-speech)
```

The Swift code is compiled to a static library during `cargo build` via `build.rs`, then linked into the Tauri binary. No sidecar processes, no runtime dependencies.

## Events

| Event              | Payload      | Description                                       |
| ------------------ | ------------ | ------------------------------------------------- |
| `plugin:mlx:token` | `TokenEvent` | Emitted for each generated token during streaming |

## Why MLX over llama.cpp?

|                      | MLX                  | llama.cpp               |
| -------------------- | -------------------- | ----------------------- |
| Metal integration    | Native (Apple's own) | Generic compute shaders |
| Unified memory       | First-class support  | Bolt-on                 |
| MoE models           | Full support         | Partial/broken          |
| Mamba-2 / SSM        | Supported            | Limited                 |
| Performance (M4 Max) | **21-87% faster**    | Baseline                |
| Model format         | SafeTensors (MLX)    | GGUF                    |

Sources: [arXiv 2511.05502](https://arxiv.org/abs/2511.05502), [arXiv 2601.19139](https://arxiv.org/html/2601.19139v1)

## License

MIT
