import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ModelInfo {
  repoId: string;
  loaded: boolean;
}

export interface TokenEvent {
  token: string;
  done: boolean;
  tokensGenerated: number;
}

export interface GenerationResult {
  text: string;
  tokensGenerated: number;
  generationTimeMs: number;
  tokensPerSecond: number;
}

export interface CatalogModel {
  repoId: string;
  name: string;
  description: string;
  sizeBytes: number;
  contextLength: number;
}

export interface DownloadProgress {
  /** Fraction completed (0.0 – 1.0) */
  fraction: number;
  /** HuggingFace repo ID being downloaded */
  repoId: string;
}

export interface GenerateOptions {
  maxTokens?: number;
  temperature?: number;
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

/**
 * Load an MLX model by HuggingFace repo ID.
 * Downloads the model on first use (cached for subsequent loads).
 *
 * @example
 * ```ts
 * await loadModel("mlx-community/gemma-3-4b-it-4bit");
 * ```
 */
export async function loadModel(
  repoId: string,
): Promise<{ status: string; repoId: string }> {
  return invoke("plugin:mlx|load_model", { request: { repoId } });
}

/**
 * Unload the current model and free GPU memory.
 */
export async function unloadModel(): Promise<{ status: string }> {
  return invoke("plugin:mlx|unload_model");
}

/**
 * Generate text from a prompt.
 * Returns the complete result after generation finishes.
 * For streaming tokens, use `onToken()` before calling this.
 *
 * @example
 * ```ts
 * const unsub = onToken((event) => console.log(event.token));
 * const result = await generate("Explain quantum computing", { maxTokens: 512 });
 * console.log(result.text);
 * unsub();
 * ```
 */
export async function generate(
  prompt: string,
  options: GenerateOptions = {},
): Promise<GenerationResult> {
  return invoke("plugin:mlx|generate", {
    request: {
      prompt,
      maxTokens: options.maxTokens ?? 2048,
      temperature: options.temperature ?? 0.7,
    },
  });
}

/**
 * Check if a model is currently loaded.
 */
export async function isLoaded(): Promise<{
  loaded: boolean;
  repoId: string | null;
}> {
  return invoke("plugin:mlx|is_loaded");
}

/**
 * Get information about the currently loaded model.
 */
export async function getModelInfo(): Promise<ModelInfo | null> {
  return invoke("plugin:mlx|get_model_info");
}

/**
 * List available models from the built-in catalog.
 */
export async function listModels(): Promise<CatalogModel[]> {
  return invoke("plugin:mlx|list_models");
}

// ---------------------------------------------------------------------------
// Event listeners
// ---------------------------------------------------------------------------

/**
 * Subscribe to streaming token events during generation.
 * Returns an unsubscribe function.
 *
 * @example
 * ```ts
 * const unsub = onToken((event) => {
 *   process.stdout.write(event.token);
 *   if (event.done) console.log("\n--- done ---");
 * });
 * await generate("Hello world");
 * unsub();
 * ```
 */
export function onToken(
  callback: (event: TokenEvent) => void,
): Promise<UnlistenFn> {
  return listen<TokenEvent>("plugin:mlx:token", (event) => {
    callback(event.payload);
  });
}

/**
 * Subscribe to model download progress events.
 * Emitted while fetching model weights from HuggingFace.
 *
 * @example
 * ```ts
 * const unsub = onDownloadProgress((p) => {
 *   console.log(`${(p.fraction * 100).toFixed(1)}% — ${p.repoId}`);
 * });
 * await loadModel("mlx-community/gemma-3-4b-it-4bit");
 * unsub();
 * ```
 */
export function onDownloadProgress(
  callback: (event: DownloadProgress) => void,
): Promise<UnlistenFn> {
  return listen<DownloadProgress>("plugin:mlx:download-progress", (event) => {
    callback(event.payload);
  });
}
