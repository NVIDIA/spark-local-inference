//! Typed host configuration independent of frontend argument parsing.
use flare_helios::offload::{parse_drain_on_shutdown, parse_max_age, parse_size};
use serde::Deserialize;
use std::{
    collections::HashMap,
    fs,
    net::IpAddr,
    num::NonZeroUsize,
    path::{Path, PathBuf},
};
type Failure = Box<dyn std::error::Error + Send + Sync>;

#[derive(Default, Deserialize)]
#[serde(default)]
pub struct Configuration {
    pub server: Server,
    pub defaults: Model,
    pub models: HashMap<String, Model>,
    pub memory: Memory,
    pub prefix_cache: PrefixCache,
    pub paths: Paths,
}

#[derive(Default, Deserialize)]
#[serde(default)]
pub struct Server {
    pub host: Option<IpAddr>,
    pub port: Option<std::num::NonZeroU16>,
    pub api_key: Option<String>,
    pub thinking_token_budget: Option<std::num::NonZeroU32>,
    pub scheduler_checkpoint_extra_work_fraction: Option<f64>,
}

#[derive(Default, Deserialize)]
#[serde(default)]
pub struct Model {
    pub max_concurrency: Option<NonZeroUsize>,
}

#[derive(Default, Deserialize)]
#[serde(default)]
pub struct Memory {
    pub reserve: Option<String>,
}

#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PrefixCache {
    pub memory: Option<String>,
    pub disk_limit: Option<String>,
    pub drain_on_shutdown: Option<String>,
    pub max_age: Option<String>,
}

#[derive(Default, Deserialize)]
#[serde(default)]
pub struct Paths {
    #[serde(alias = "package_cache")]
    pub core_cache: Option<String>,
    pub prefix_cache: Option<String>,
    pub hub_cache: Option<String>,
}

impl Configuration {
    pub fn load(path: &Path, required: bool) -> Result<Self, Failure> {
        let text = match fs::read_to_string(path) {
            Ok(text) => text,
            Err(error) if !required && error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(Self::default());
            }
            Err(error) => return Err(format!("cannot read {}: {error}", path.display()).into()),
        };
        Self::parse(&text).map_err(|error| format!("{}: {error}", path.display()).into())
    }

    pub fn parse(text: &str) -> Result<Self, Failure> {
        let configuration: Self = toml::from_str(text)?;
        if configuration.server.scheduler_checkpoint_extra_work_fraction.is_some_and(|value|
            !value.is_finite() || !(0.0..=1.0).contains(&value)) {
            return Err("server.scheduler_checkpoint_extra_work_fraction must be finite and between zero and one".into());
        }
        size(configuration.memory.reserve.as_deref(), "memory.reserve")?;
        size(
            nonempty(configuration.prefix_cache.memory.as_deref()),
            "prefix_cache.memory",
        )?;
        size(
            nonempty(configuration.prefix_cache.disk_limit.as_deref()),
            "prefix_cache.disk_limit",
        )?;
        if let Some(value) = nonempty(configuration.prefix_cache.drain_on_shutdown.as_deref()) {
            parse_drain_on_shutdown(value)
                .map_err(|error| format!("prefix_cache.drain_on_shutdown: {error}"))?;
        }
        if let Some(value) = nonempty(configuration.prefix_cache.max_age.as_deref()) {
            parse_max_age(value).map_err(|error| format!("prefix_cache.max_age: {error}"))?;
        }
        Ok(configuration)
    }

    pub fn core_cache(&self, home: &Path) -> Option<PathBuf> {
        nonempty(self.paths.core_cache.as_deref()).map(|path| expand(home, Path::new(path)))
    }

    /// Hugging Face hub cache that prepared-core acquisition may borrow from.
    pub fn hub_cache(&self, home: &Path) -> Option<PathBuf> {
        nonempty(self.paths.hub_cache.as_deref()).map(|path| expand(home, Path::new(path)))
    }

    pub fn api_key(&self, file: Option<&Path>) -> Result<String, Failure> {
        match file {
            Some(path) => fs::read_to_string(path)
                .map(|value| value.trim().to_owned())
                .map_err(|error| {
                    format!("cannot read API key file {}: {error}", path.display()).into()
                }),
            None => Ok(self.server.api_key.clone().unwrap_or_default()),
        }
    }

    pub fn thinking_token_budget(&self) -> Option<u32> {
        self.server
            .thinking_token_budget
            .map(std::num::NonZeroU32::get)
    }
}

pub fn nonempty(value: Option<&str>) -> Option<&str> {
    value.filter(|value| !value.trim().is_empty())
}

pub fn size(value: Option<&str>, key: &str) -> Result<Option<u64>, Failure> {
    value
        .map(|value| parse_size(value).map_err(|error| format!("{key}: {error}").into()))
        .transpose()
}

pub fn expand(home: &Path, path: &Path) -> PathBuf {
    match path.strip_prefix("~") {
        Ok(rest) => home.join(rest),
        Err(_) => path.to_owned(),
    }
}
