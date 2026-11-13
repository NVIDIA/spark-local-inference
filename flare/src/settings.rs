// SPDX-License-Identifier: Apache-2.0
//! Explicit host configuration shared by application clients.
#[path = "runtime_settings.rs"]
mod runtime;
use crate::configuration::{Configuration, expand, nonempty, size};
use crate::options::HubOptions;
use flare_helios::{
    offload::{PrefixCacheOffload, parse_drain_on_shutdown, parse_max_age},
    runtime_core::{LoadOptions, MemoryLimit},
};
pub(crate) use runtime::RuntimeIdentity;
pub use runtime::RuntimeSettings;
use std::{
    net::{IpAddr, SocketAddr},
    path::{Path, PathBuf},
};
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
#[derive(Clone, Debug)]
pub struct Settings {
    pub runtime: RuntimeSettings,
    pub model_directory: PathBuf,
    pub state_file: PathBuf,
    pub bind: SocketAddr,
    pub api_key: Option<String>,
    pub max_concurrency: Option<usize>,
    pub memory_reserve: Option<u64>,
    pub hub: HubOptions,
    pub trust_store: Option<PathBuf>,
    pub publisher_keys: Vec<PathBuf>,
    pub thinking_token_budget: Option<u32>,
    pub scheduler_checkpoint_extra_work_fraction: Option<f64>,
    pub prefix_cache: PrefixCacheOffload,
    pub model_concurrency: std::collections::HashMap<String, usize>,
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn settings_keep_shared_paths_and_model_override_precedence() {
        let config=Configuration::parse("[paths]\ncore_cache='~/models'\nprefix_cache='~/prefix'\n[defaults]\nmax_concurrency=2\n[models.example]\nmax_concurrency=3\n[memory]\nreserve='8G'\n[prefix_cache]\nmemory='1GiB'\n").unwrap();
        let settings =
            Settings::resolve(config, Path::new("user"), Path::new("custom/config.toml")).unwrap();
        assert_eq!(settings.model_directory, Path::new("user/models"));
        assert_eq!(settings.state_file, Path::new("custom/desktop-state.json"));
        assert_eq!(
            settings.load_options(&["example"]).unwrap().max_concurrency,
            Some(3)
        );
        assert_eq!(
            settings.load_options(&["another"]).unwrap().max_concurrency,
            Some(2)
        );
        assert_eq!(
            settings
                .load_options(&[])
                .unwrap()
                .prefix_cache
                .memory_bytes,
            Some(1 << 30)
        );
        assert_eq!(settings.memory_reserve, Some(8 << 30));
        assert!(settings.runtime.prefix_cache_disk_enabled);
        assert_eq!(settings.load_options(&[]).unwrap().prefix_cache.disk, Some(PathBuf::from("user/prefix")));
    }
    #[test]
    fn checkpoint_extra_work_fraction_reaches_application_settings() {
        for fraction in [None, Some(0.0), Some(0.25), Some(1.0)] {
            let text = fraction.map(|value| format!("[server]\nscheduler_checkpoint_extra_work_fraction={value}")).unwrap_or_default();
            let configuration = Configuration::parse(&text).unwrap();
            let settings = Settings::resolve(configuration, Path::new("user"), Path::new("config.toml")).unwrap();
            assert_eq!(settings.scheduler_checkpoint_extra_work_fraction, fraction);
        }
    }

    #[test]
    fn conflicting_alias_defaults_are_rejected() {
        let config = Configuration::parse(
            "[models.first]\nmax_concurrency=2\n[models.second]\nmax_concurrency=3\n",
        )
        .unwrap();
        let settings =
            Settings::resolve(config, Path::new("user"), Path::new("config.toml")).unwrap();
        assert!(settings.load_options(&["first", "second"]).is_err());
    }
}
impl Settings {
    pub fn load(path: Option<&Path>) -> Result<Self> {
        let home = crate::environment::home_directory()?;
        let default = home.join(".config/flare/config.toml");
        let selected = path.unwrap_or(&default);
        let configuration = Configuration::load(selected, path.is_some())?;
        Self::resolve(configuration, &home, selected)
    }
    fn resolve(configuration: Configuration, home: &Path, selected: &Path) -> Result<Self> {
        let port = configuration.server.port.map(|v| v.get()).unwrap_or(8000);
        let model_directory = configuration
            .core_cache(home)
            .unwrap_or_else(|| crate::cache_support::default_core_root(home));
        let hub = HubOptions {
            core_cache: Some(model_directory.clone()),
            hub_cache: Some(
                configuration
                    .hub_cache(home)
                    .unwrap_or_else(|| crate::cache_support::hub_cache_default(home)),
            ),
            ..HubOptions::default()
        };
        let memory_reserve = size(configuration.memory.reserve.as_deref(), "memory.reserve")?;
        let prefix_cache = PrefixCacheOffload {
            memory_bytes: size(
                nonempty(configuration.prefix_cache.memory.as_deref()),
                "prefix_cache.memory",
            )?,
            disk: nonempty(configuration.paths.prefix_cache.as_deref())
                .map(|v| expand(home, Path::new(v))),
            disk_limit: size(
                nonempty(configuration.prefix_cache.disk_limit.as_deref()),
                "prefix_cache.disk_limit",
            )?,
            drain_on_shutdown: nonempty(configuration.prefix_cache.drain_on_shutdown.as_deref())
                .map(parse_drain_on_shutdown)
                .transpose()?
                .unwrap_or_default(),
            max_age: nonempty(configuration.prefix_cache.max_age.as_deref())
                .map(parse_max_age)
                .transpose()?
                .unwrap_or_default(),
            pool: None,
        };
        let thinking_token_budget = configuration.thinking_token_budget();
        let model_concurrency = configuration
            .models
            .into_iter()
            .filter_map(|(name, model)| model.max_concurrency.map(|value| (name, value.get())))
            .collect();
        let bind = SocketAddr::new(
            configuration
                .server
                .host
                .unwrap_or(IpAddr::from([127, 0, 0, 1])),
            port,
        );
        let runtime = RuntimeSettings {
            host: bind.ip().to_string(),
            port: bind.port(),
            prefix_cache_disk_enabled: prefix_cache.disk.is_some(),
            prefix_cache_memory: prefix_cache
                .memory_bytes
                .map(|n| n.to_string())
                .unwrap_or_default(),
            prefix_cache_disk: prefix_cache
                .disk
                .as_ref()
                .map(|path| path.display().to_string())
                .unwrap_or_else(|| default_prefix_disk(&model_directory).display().to_string()),
            prefix_cache_disk_limit: prefix_cache
                .disk_limit
                .map(|n| n.to_string())
                .unwrap_or_default(),
            prefix_cache_max_age: duration_text(prefix_cache.max_age),
            prefix_cache_drain_on_shutdown: duration_text(prefix_cache.drain_on_shutdown),
            ..RuntimeSettings::default()
        };
        Ok(Self {
            runtime,
            model_directory,
            state_file: selected.with_file_name("desktop-state.json"),
            bind: SocketAddr::new(
                configuration
                    .server
                    .host
                    .unwrap_or(IpAddr::from([127, 0, 0, 1])),
                port,
            ),
            api_key: configuration.server.api_key.filter(|key| !key.is_empty()),
            max_concurrency: configuration.defaults.max_concurrency.map(|v| v.get()),
            memory_reserve,
            hub,
            trust_store: None,
            publisher_keys: Vec::new(),
            thinking_token_budget,
            scheduler_checkpoint_extra_work_fraction: configuration.server.scheduler_checkpoint_extra_work_fraction,
            prefix_cache,
            model_concurrency,
        })
    }
    pub fn load_options(&self, names: &[&str]) -> Result<LoadOptions> {
        let mut selected = None;
        for name in names
            .iter()
            .filter(|_| self.runtime.max_concurrency.is_none())
        {
            if let Some(value) = self.model_concurrency.get(*name) {
                if selected.is_some_and(|previous| previous != *value) {
                    return Err(
                        "Conflicting concurrency settings for this model and its aliases".into(),
                    );
                }
                selected = Some(*value);
            }
        }
        let base = LoadOptions {
            max_concurrency: selected.or(self.max_concurrency),
            memory_limit: self.memory_reserve.map(MemoryLimit::Reserve),
            prefix_cache: self.prefix_cache.clone(),
            ..LoadOptions::default()
        };
        Ok(self
            .runtime
            .apply(base, default_prefix_disk(&self.model_directory))?)
    }
    pub(crate) fn runtime_identity(&self, names: &[&str]) -> Result<RuntimeIdentity> {
        Ok(RuntimeIdentity::new(&self.runtime, &self.load_options(names)?)?)
    }
}
fn default_prefix_disk(model_directory: &Path) -> PathBuf {
    model_directory
        .parent()
        .unwrap_or(model_directory)
        .join("prefix-cache")
}
fn duration_text(value: std::time::Duration) -> String {
    if value.is_zero() {
        "off".into()
    } else if value == std::time::Duration::MAX {
        "on".into()
    } else {
        format!("{}s", value.as_secs())
    }
}

#[cfg(test)]
mod runtime_tests {
    use super::*;
    #[test]
    fn runtime_identity_preserves_active_options_and_configuration_secrets() {
        let config=Configuration::parse("[server]\napi_key='unchanged-secret'\n[defaults]\nmax_concurrency=2\n[models.example]\nmax_concurrency=3\n[memory]\nreserve='8G'\n").unwrap();
        let active=Settings::resolve(config,Path::new("user"),Path::new("config.toml")).unwrap();
        let identity=active.runtime_identity(&["example"]).unwrap();
        let mut desired=active.clone();
        assert_eq!(identity,desired.runtime_identity(&["example"]).unwrap());
        desired.runtime.memory_limit="-8GiB".into();
        assert_eq!(identity,desired.runtime_identity(&["example"]).unwrap());
        desired.runtime.embedding_table_storage="disk".into();
        assert_eq!(identity,desired.runtime_identity(&["example"]).unwrap());
        desired.runtime.max_concurrency=Some(1);
        assert_ne!(identity,desired.runtime_identity(&["example"]).unwrap());
        assert_eq!(active.load_options(&["example"]).unwrap().max_concurrency,Some(3));
        assert_eq!(desired.load_options(&["example"]).unwrap().max_concurrency,Some(1));
        assert_eq!(desired.api_key.as_deref(),Some("unchanged-secret"));
        assert!(!serde_json::to_string(&desired.runtime).unwrap().contains("secret"));
    }
    #[test]
    fn explicit_desktop_concurrency_wins_over_conflicting_model_aliases() {
        let config=Configuration::parse("[models.first]\nmax_concurrency=2\n[models.second]\nmax_concurrency=3\n").unwrap();
        let mut settings=Settings::resolve(config,Path::new("user"),Path::new("config.toml")).unwrap();
        settings.runtime.max_concurrency=Some(1);
        assert_eq!(settings.load_options(&["first","second"]).unwrap().max_concurrency,Some(1));
    }
}
