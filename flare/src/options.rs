//! Typed prepared-core acquisition options.

use std::path::PathBuf;

#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct HubOptions {
    pub revision: Option<String>,
    pub filename: Option<String>,
    pub cache_dir: Option<PathBuf>,
    pub token_file: Option<PathBuf>,
    pub offline: bool,
    pub tensor_parallel: Vec<u32>,
    #[serde(alias = "model_cache", alias = "package_cache")]
    pub core_cache: Option<PathBuf>,
    #[serde(skip)]
    pub hub_cache: Option<PathBuf>,
}
