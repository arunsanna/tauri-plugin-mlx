use serde::{ser::Serializer, Serialize};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("MLX engine error: {0}")]
    Engine(String),
    #[error("Model not loaded")]
    ModelNotLoaded,
    #[error("Model load failed: {0}")]
    LoadFailed(String),
    #[error("Generation failed: {0}")]
    GenerationFailed(String),
}

impl Serialize for Error {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.to_string().as_ref())
    }
}
