use tauri::{
    plugin::{Builder, TauriPlugin},
    Manager, Runtime,
};

pub use models::*;

#[cfg(desktop)]
mod desktop;

mod catalog;
mod commands;
mod error;
mod models;

#[cfg(target_os = "macos")]
mod macos_bridge;

pub use error::{Error, Result};

#[cfg(desktop)]
use desktop::Mlx;

/// Extensions to [`tauri::App`], [`tauri::AppHandle`] and [`tauri::Window`]
/// to access the MLX inference APIs.
pub trait MlxExt<R: Runtime> {
    fn mlx(&self) -> &Mlx<R>;
}

impl<R: Runtime, T: Manager<R>> MlxExt<R> for T {
    fn mlx(&self) -> &Mlx<R> {
        self.state::<Mlx<R>>().inner()
    }
}

/// Cleanup native MLX resources.
///
/// On macOS, destroys the MLX engine and frees GPU memory.
/// On other platforms, this is a no-op.
pub fn mlx_cleanup() -> Result<()> {
    #[cfg(target_os = "macos")]
    {
        macos_bridge::mlx_cleanup()
    }

    #[cfg(not(target_os = "macos"))]
    {
        Ok(())
    }
}

/// Initialize the MLX plugin.
pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("mlx")
        .invoke_handler(tauri::generate_handler![
            commands::load_model,
            commands::unload_model,
            commands::generate,
            commands::is_loaded,
            commands::get_model_info,
            commands::list_models,
        ])
        .setup(|app, api| {
            #[cfg(desktop)]
            let mlx = desktop::init(app, api)?;
            app.manage(mlx);
            Ok(())
        })
        .build()
}
