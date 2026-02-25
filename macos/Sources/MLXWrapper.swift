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
        // Unload any existing model first
        unload()

        let semaphore = DispatchSemaphore(value: 0)
        var success = false

        Task {
            do {
                let config = ModelConfiguration(id: repoId)
                let container = try await ModelContainer.load(configuration: config)
                self.modelContainer = container
                self.currentRepoId = repoId
                success = true
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
                let result = try await container.perform { context in
                    let input = try await context.processor.prepare(input: .init(prompt: prompt))
                    var output = ""

                    // Generate tokens using MLX
                    let generateParameters = GenerateParameters(temperature: temperature)
                    for try await token in try context.model.generate(
                        input: input,
                        parameters: generateParameters
                    ) {
                        let tokenText = context.tokenizer.decode(tokens: [token])
                        output += tokenText
                        tokenCount += 1
                        tokenCallback(tokenText, false, tokenCount)

                        if tokenCount >= maxTokens {
                            break
                        }
                    }

                    return output
                }

                fullText = result
                // Signal completion
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
func mlx_get_model_id(_ ptr: UnsafeMutableRawPointer) -> UnsafePointer<CChar>? {
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
) -> UnsafePointer<CChar>? {
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
