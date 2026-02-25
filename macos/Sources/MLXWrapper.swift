import Foundation
import MLXLLM
import MLXLMCommon

// MARK: - MLX Engine

/// Manages MLX model lifecycle and inference.
/// Exposed to Rust via @_cdecl C functions (see Bridge.h).
final class MLXEngine {
    private var modelContainer: ModelContainer?
    private var currentRepoId: String?

    func loadModel(repoId: String) -> Bool {
        unload()

        let semaphore = DispatchSemaphore(value: 0)
        var success = false

        Task {
            do {
                // loadModelContainer downloads (if needed), caches, and builds the container
                let container = try await loadModelContainer(id: repoId)
                self.modelContainer = container
                self.currentRepoId = repoId
                success = true
                print("[MLXEngine] Model loaded: \(repoId)")
            } catch {
                print("[MLXEngine] Failed to load model '\(repoId)': \(error)")
            }
            semaphore.signal()
        }

        semaphore.wait()
        return success
    }

    func unload() {
        modelContainer = nil
        currentRepoId = nil
    }

    var isLoaded: Bool {
        modelContainer != nil
    }

    var modelId: String? {
        currentRepoId
    }

    func generate(
        prompt: String,
        maxTokens: Int,
        temperature: Float,
        tokenCallback: @escaping (String, Bool, UInt32) -> Void
    ) -> String? {
        guard let container = modelContainer else {
            print("[MLXEngine] No model loaded")
            return nil
        }

        let semaphore = DispatchSemaphore(value: 0)
        var fullText = ""
        var tokenCount: UInt32 = 0

        Task {
            do {
                // Prepare input from prompt string
                let userInput = UserInput(prompt: prompt)
                let lmInput = try await container.prepare(input: userInput)

                // Generate with streaming via AsyncStream<Generation>
                let parameters = GenerateParameters(
                    maxTokens: maxTokens,
                    temperature: temperature
                )
                let stream = try await container.generate(
                    input: lmInput,
                    parameters: parameters
                )

                for await generation in stream {
                    switch generation {
                    case .chunk(let text):
                        fullText += text
                        tokenCount += 1
                        tokenCallback(text, false, tokenCount)
                    case .info:
                        // Generation complete — info carries perf stats
                        tokenCallback("", true, tokenCount)
                    case .toolCall:
                        // Not used for plain text generation
                        break
                    }
                }

                // If stream ended without .info, signal done
                tokenCallback("", true, tokenCount)
            } catch {
                print("[MLXEngine] Generation error: \(error)")
                tokenCallback("", true, tokenCount)
            }
            semaphore.signal()
        }

        semaphore.wait()
        return fullText
    }
}

// MARK: - C FFI Exports

/// Global engine instance, created/destroyed via FFI.
private var engineInstance: MLXEngine?

@_cdecl("mlx_create_engine")
func mlx_create_engine() -> UnsafeMutableRawPointer {
    let engine = MLXEngine()
    engineInstance = engine
    return Unmanaged.passRetained(engine).toOpaque()
}

@_cdecl("mlx_destroy_engine")
func mlx_destroy_engine(_ ptr: UnsafeMutableRawPointer) {
    let engine = Unmanaged<MLXEngine>.fromOpaque(ptr).takeRetainedValue()
    engine.unload()
    engineInstance = nil
}

@_cdecl("mlx_load_model")
func mlx_load_model(_ ptr: UnsafeMutableRawPointer, _ repoId: UnsafePointer<CChar>) -> Bool {
    let engine = Unmanaged<MLXEngine>.fromOpaque(ptr).takeUnretainedValue()
    let repo = String(cString: repoId)
    return engine.loadModel(repoId: repo)
}

@_cdecl("mlx_unload_model")
func mlx_unload_model(_ ptr: UnsafeMutableRawPointer) {
    let engine = Unmanaged<MLXEngine>.fromOpaque(ptr).takeUnretainedValue()
    engine.unload()
}

@_cdecl("mlx_is_loaded")
func mlx_is_loaded(_ ptr: UnsafeMutableRawPointer) -> Bool {
    let engine = Unmanaged<MLXEngine>.fromOpaque(ptr).takeUnretainedValue()
    return engine.isLoaded
}

@_cdecl("mlx_get_model_id")
func mlx_get_model_id(_ ptr: UnsafeMutableRawPointer) -> UnsafeMutablePointer<CChar>? {
    let engine = Unmanaged<MLXEngine>.fromOpaque(ptr).takeUnretainedValue()
    guard let modelId = engine.modelId else { return nil }
    return strdup(modelId)
}

/// C callback type matching Bridge.h
typealias CTokenCallback = @convention(c) (UnsafePointer<CChar>?, Bool, UInt32, UnsafeMutableRawPointer?) -> Void

@_cdecl("mlx_generate")
func mlx_generate(
    _ ptr: UnsafeMutableRawPointer,
    _ prompt: UnsafePointer<CChar>,
    _ maxTokens: Int32,
    _ temperature: Float,
    _ callback: CTokenCallback,
    _ userData: UnsafeMutableRawPointer?
) -> UnsafeMutablePointer<CChar>? {
    let engine = Unmanaged<MLXEngine>.fromOpaque(ptr).takeUnretainedValue()
    let promptStr = String(cString: prompt)

    let result = engine.generate(
        prompt: promptStr,
        maxTokens: Int(maxTokens),
        temperature: temperature
    ) { token, done, count in
        token.withCString { cstr in
            callback(cstr, done, count, userData)
        }
    }

    guard let text = result else { return nil }
    return strdup(text)
}
