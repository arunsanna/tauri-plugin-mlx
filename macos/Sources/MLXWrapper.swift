import Foundation
import MLXLLM
import MLXLMCommon

// MARK: - Force-load LLM factory

/// Force the linker to keep MLXLLM.TrampolineModelFactory.
/// MLXLMCommon discovers it at runtime via NSClassFromString; without a direct
/// reference the linker dead-strips it from the static library.
private func registerLLMFactory() {
    ModelFactoryRegistry.shared.addTrampoline {
        TrampolineModelFactory.modelFactory()
    }
}

// MARK: - Logging

/// Write to stderr so output is visible in Tauri dev console
/// (Swift print() goes to stdout which Tauri doesn't capture)
private func mlxLog(_ message: String) {
    let msg = "[MLXEngine] \(message)\n"
    fputs(msg, stderr)
    NSLog("%@", msg)
}

// MARK: - Thread-safe last error storage

private var _lastError: String?
private let _lastErrorLock = NSLock()

private func setLastError(_ msg: String?) {
    _lastErrorLock.lock()
    _lastError = msg
    _lastErrorLock.unlock()
}

private func getLastError() -> String? {
    _lastErrorLock.lock()
    defer { _lastErrorLock.unlock() }
    return _lastError
}

// MARK: - MLX Engine

/// Manages MLX model lifecycle and inference.
/// Exposed to Rust via @_cdecl C functions (see Bridge.h).
final class MLXEngine {
    private var modelContainer: ModelContainer?
    private var currentRepoId: String?

    func loadModel(
        repoId: String,
        progressCallback: ((Double) -> Void)? = nil
    ) -> Bool {
        unload()
        setLastError(nil)

        let semaphore = DispatchSemaphore(value: 0)
        var success = false

        mlxLog("Loading model: \(repoId)")

        // Use Task.detached to avoid Swift cooperative thread pool deadlock.
        // A regular Task {} inherits the current executor context, which can
        // deadlock when the calling thread is blocked by the semaphore.
        Task.detached { [weak self] in
            do {
                let container = try await loadModelContainer(id: repoId) { progress in
                    let fraction = progress.fractionCompleted
                    let pct = Int(fraction * 100)
                    if pct % 5 == 0 {
                        mlxLog("Download progress: \(pct)%")
                    }
                    progressCallback?(fraction)
                }
                self?.modelContainer = container
                self?.currentRepoId = repoId
                success = true
                mlxLog("Model loaded successfully: \(repoId)")
            } catch {
                let errorMsg = "Failed to load '\(repoId)': \(error)"
                mlxLog(errorMsg)
                setLastError(errorMsg)
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
            mlxLog("No model loaded")
            return nil
        }

        let semaphore = DispatchSemaphore(value: 0)
        var fullText = ""
        var tokenCount: UInt32 = 0

        Task.detached {
            do {
                let userInput = UserInput(prompt: prompt)
                let lmInput = try await container.prepare(input: userInput)

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
                        tokenCallback("", true, tokenCount)
                    case .toolCall:
                        break
                    }
                }

                tokenCallback("", true, tokenCount)
            } catch {
                mlxLog("Generation error: \(error)")
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
    registerLLMFactory()
    let engine = MLXEngine()
    engineInstance = engine
    mlxLog("Engine created")
    return Unmanaged.passRetained(engine).toOpaque()
}

@_cdecl("mlx_destroy_engine")
func mlx_destroy_engine(_ ptr: UnsafeMutableRawPointer) {
    let engine = Unmanaged<MLXEngine>.fromOpaque(ptr).takeRetainedValue()
    engine.unload()
    engineInstance = nil
    mlxLog("Engine destroyed")
}

/// C callback type for download progress
typealias CProgressCallback = @convention(c) (Double, UnsafeMutableRawPointer?) -> Void

@_cdecl("mlx_load_model")
func mlx_load_model(
    _ ptr: UnsafeMutableRawPointer,
    _ repoId: UnsafePointer<CChar>,
    _ progressCb: CProgressCallback?,
    _ progressUserData: UnsafeMutableRawPointer?
) -> Bool {
    let engine = Unmanaged<MLXEngine>.fromOpaque(ptr).takeUnretainedValue()
    let repo = String(cString: repoId)

    let progressWrapper: ((Double) -> Void)? = progressCb.map { cb in
        { fraction in cb(fraction, progressUserData) }
    }

    return engine.loadModel(repoId: repo, progressCallback: progressWrapper)
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

/// Return the last error message (caller must free with libc free())
@_cdecl("mlx_get_last_error")
func mlx_get_last_error() -> UnsafeMutablePointer<CChar>? {
    guard let err = getLastError() else { return nil }
    return strdup(err)
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
