//! macOS FFI bridge to Swift MLX inference engine.
//!
//! Communicates with MLXWrapper.swift via @_cdecl / extern "C" functions.
//! Token streaming uses a C callback that emits Tauri events.

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::sync::{Arc, Mutex};

use serde::Serialize;
use tauri::{AppHandle, Emitter, Runtime};

use crate::models::{DownloadProgress, GenerationResult, ModelInfo, TokenEvent};

// ---------------------------------------------------------------------------
// FFI declarations — implemented in MLXWrapper.swift
// ---------------------------------------------------------------------------
extern "C" {
    fn mlx_create_engine() -> *mut std::ffi::c_void;
    fn mlx_destroy_engine(engine: *mut std::ffi::c_void);
    fn mlx_load_model(
        engine: *mut std::ffi::c_void,
        repo_id: *const c_char,
        progress_cb: Option<extern "C" fn(f64, *mut std::ffi::c_void)>,
        progress_user_data: *mut std::ffi::c_void,
    ) -> bool;
    fn mlx_unload_model(engine: *mut std::ffi::c_void);
    fn mlx_is_loaded(engine: *mut std::ffi::c_void) -> bool;
    fn mlx_get_model_id(engine: *mut std::ffi::c_void) -> *mut c_char;
    fn mlx_generate(
        engine: *mut std::ffi::c_void,
        prompt: *const c_char,
        max_tokens: i32,
        temperature: f32,
        callback: extern "C" fn(*const c_char, bool, u32, *mut std::ffi::c_void),
        user_data: *mut std::ffi::c_void,
    ) -> *mut c_char;
    fn mlx_get_last_error() -> *mut c_char;
}

/// Read and free the last error string from the Swift side.
fn read_last_error() -> Option<String> {
    let ptr = unsafe { mlx_get_last_error() };
    if ptr.is_null() {
        return None;
    }
    let msg = unsafe { CStr::from_ptr(ptr).to_string_lossy().to_string() };
    unsafe { libc::free(ptr as *mut libc::c_void) };
    Some(msg)
}

// ---------------------------------------------------------------------------
// Global engine singleton
// ---------------------------------------------------------------------------
struct EnginePtr(*mut std::ffi::c_void);
unsafe impl Send for EnginePtr {}
unsafe impl Sync for EnginePtr {}

struct AppHandleWrapper {
    emit_fn: Box<dyn Fn(&str, serde_json::Value) + Send + Sync>,
}

impl AppHandleWrapper {
    fn new<R: Runtime>(app: AppHandle<R>) -> Self {
        Self {
            emit_fn: Box::new(move |event, payload| {
                let _ = app.emit(event, payload);
            }),
        }
    }

    fn emit_event<T: Serialize>(&self, event: &str, payload: T) -> Result<(), String> {
        match serde_json::to_value(payload) {
            Ok(json) => {
                (self.emit_fn)(event, json);
                Ok(())
            }
            Err(e) => Err(format!("Failed to serialize payload: {}", e)),
        }
    }
}

struct AppHandlePtr(*mut std::ffi::c_void);
unsafe impl Send for AppHandlePtr {}
unsafe impl Sync for AppHandlePtr {}

lazy_static::lazy_static! {
    static ref ENGINE: Arc<Mutex<Option<EnginePtr>>> = Arc::new(Mutex::new(None));
    static ref APP_HANDLE: Arc<Mutex<Option<AppHandlePtr>>> = Arc::new(Mutex::new(None));
}

fn ensure_engine<R: Runtime>(app: &AppHandle<R>) -> *mut std::ffi::c_void {
    let mut engine = ENGINE.lock().unwrap();
    if engine.is_none() {
        unsafe {
            let eng = mlx_create_engine();
            *engine = Some(EnginePtr(eng));
        }

        let mut app_handle = APP_HANDLE.lock().unwrap();
        if app_handle.is_none() {
            let wrapper = Box::new(AppHandleWrapper::new(app.clone()));
            let ptr = Box::into_raw(wrapper) as *mut std::ffi::c_void;
            *app_handle = Some(AppHandlePtr(ptr));
        }
    }
    engine.as_ref().unwrap().0
}

// ---------------------------------------------------------------------------
// Token streaming callback
// ---------------------------------------------------------------------------
extern "C" fn token_callback(
    token_ptr: *const c_char,
    done: bool,
    tokens_generated: u32,
    _user_data: *mut std::ffi::c_void,
) {
    let token = if token_ptr.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(token_ptr).to_string_lossy().to_string() }
    };

    let event = TokenEvent {
        token,
        done,
        tokens_generated,
    };

    let app_handle = APP_HANDLE.lock().unwrap();
    if let Some(app_ptr) = &*app_handle {
        unsafe {
            let app: &AppHandleWrapper = &*(app_ptr.0 as *const AppHandleWrapper);
            let _ = app.emit_event("plugin:mlx:token", event);
        }
    }
}

// ---------------------------------------------------------------------------
// Download progress callback
// ---------------------------------------------------------------------------

/// User-data passed through the C progress callback.
struct ProgressCtx {
    repo_id: String,
}

extern "C" fn progress_callback(fraction: f64, user_data: *mut std::ffi::c_void) {
    if user_data.is_null() {
        return;
    }
    let ctx = unsafe { &*(user_data as *const ProgressCtx) };
    let event = DownloadProgress {
        fraction,
        repo_id: ctx.repo_id.clone(),
    };

    let app_handle = APP_HANDLE.lock().unwrap();
    if let Some(app_ptr) = &*app_handle {
        unsafe {
            let app: &AppHandleWrapper = &*(app_ptr.0 as *const AppHandleWrapper);
            let _ = app.emit_event("plugin:mlx:download-progress", event);
        }
    }
}

// ---------------------------------------------------------------------------
// Public bridge API (called from desktop.rs)
// ---------------------------------------------------------------------------

pub async fn load_model<R: Runtime>(
    app: AppHandle<R>,
    repo_id: &str,
) -> crate::Result<()> {
    let engine = ensure_engine(&app);
    let repo_cstr =
        CString::new(repo_id).map_err(|e| crate::Error::LoadFailed(e.to_string()))?;

    log::info!("MLX loading model: {}", repo_id);

    let ctx = Box::new(ProgressCtx {
        repo_id: repo_id.to_string(),
    });
    let ctx_ptr = Box::into_raw(ctx) as *mut std::ffi::c_void;

    let success = unsafe {
        mlx_load_model(
            engine,
            repo_cstr.as_ptr(),
            Some(progress_callback),
            ctx_ptr,
        )
    };

    // Free the context
    unsafe {
        let _ = Box::from_raw(ctx_ptr as *mut ProgressCtx);
    }

    if success {
        log::info!("MLX model loaded: {}", repo_id);
        Ok(())
    } else {
        let detail = read_last_error().unwrap_or_else(|| "unknown error".to_string());
        log::error!("MLX load failed for '{}': {}", repo_id, detail);
        Err(crate::Error::LoadFailed(format!(
            "Failed to load '{}': {}",
            repo_id, detail
        )))
    }
}

pub async fn unload_model<R: Runtime>(app: AppHandle<R>) {
    let engine = ensure_engine(&app);
    unsafe {
        mlx_unload_model(engine);
    }
    log::info!("MLX model unloaded");
}

pub async fn is_loaded<R: Runtime>(app: AppHandle<R>) -> bool {
    let engine = ensure_engine(&app);
    unsafe { mlx_is_loaded(engine) }
}

pub async fn get_model_info<R: Runtime>(app: AppHandle<R>) -> Option<ModelInfo> {
    let engine = ensure_engine(&app);
    let loaded = unsafe { mlx_is_loaded(engine) };
    if !loaded {
        return None;
    }

    let id_ptr = unsafe { mlx_get_model_id(engine) };
    if id_ptr.is_null() {
        return None;
    }

    let repo_id = unsafe { CStr::from_ptr(id_ptr).to_string_lossy().to_string() };
    unsafe {
        libc::free(id_ptr as *mut libc::c_void);
    }

    Some(ModelInfo {
        repo_id,
        loaded: true,
    })
}

pub async fn generate<R: Runtime>(
    app: AppHandle<R>,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
) -> crate::Result<GenerationResult> {
    let engine = ensure_engine(&app);
    if !unsafe { mlx_is_loaded(engine) } {
        return Err(crate::Error::ModelNotLoaded);
    }

    let prompt_cstr =
        CString::new(prompt).map_err(|e| crate::Error::GenerationFailed(e.to_string()))?;

    let start = std::time::Instant::now();

    let result_ptr = unsafe {
        mlx_generate(
            engine,
            prompt_cstr.as_ptr(),
            max_tokens as i32,
            temperature,
            token_callback,
            std::ptr::null_mut(),
        )
    };

    let elapsed_ms = start.elapsed().as_millis() as u64;

    if result_ptr.is_null() {
        return Err(crate::Error::GenerationFailed(
            "MLX generation returned null".to_string(),
        ));
    }

    let text = unsafe { CStr::from_ptr(result_ptr).to_string_lossy().to_string() };
    unsafe {
        libc::free(result_ptr as *mut libc::c_void);
    }

    // Rough token count: split on whitespace as approximation
    let tokens_generated = text.split_whitespace().count() as u32;
    let tokens_per_second = if elapsed_ms > 0 {
        tokens_generated as f64 / (elapsed_ms as f64 / 1000.0)
    } else {
        0.0
    };

    Ok(GenerationResult {
        text,
        tokens_generated,
        generation_time_ms: elapsed_ms,
        tokens_per_second,
    })
}

/// Cleanup — free the engine and app handle.
pub fn mlx_cleanup() -> crate::Result<()> {
    let engine_ptr = {
        match ENGINE.lock() {
            Ok(mut guard) => guard.take(),
            Err(poisoned) => poisoned.into_inner().take(),
        }
    };

    if let Some(EnginePtr(ptr)) = engine_ptr {
        unsafe {
            mlx_destroy_engine(ptr);
        }
    }

    let app_ptr = {
        match APP_HANDLE.lock() {
            Ok(mut guard) => guard.take(),
            Err(poisoned) => poisoned.into_inner().take(),
        }
    };

    if let Some(AppHandlePtr(ptr)) = app_ptr {
        unsafe {
            let _boxed: Box<AppHandleWrapper> = Box::from_raw(ptr as *mut AppHandleWrapper);
        }
    }

    Ok(())
}
