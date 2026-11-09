// SPDX-License-Identifier: Apache-2.0
//! Desktop serving overrides applied only when a model is loaded.
use flare_helios::{
    engram::Storage,
    offload::{parse_drain_on_shutdown, parse_max_age, parse_size},
    runtime_core::{KvCacheLimit, LoadOptions, MemoryLimit},
};
use serde::{Deserialize, Serialize};
use std::{
    net::{IpAddr, SocketAddr},
    path::PathBuf,
};
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RuntimeSettings {
    pub memory_limit: String,
    pub kv_cache_limit: String,
    pub host: String,
    pub port: u16,
    pub max_concurrency: Option<usize>,
    pub verify_weights: bool,
    pub prefix_cache_memory: String,
    pub prefix_cache_disk: String,
    pub prefix_cache_disk_limit: String,
    pub prefix_cache_disk_enabled: bool,
    pub prefix_cache_max_age: String,
    pub prefix_cache_drain_on_shutdown: String,
    pub embedding_table_storage: String,
    pub embedding_table_cache_size: String,
    pub mtp_draft_tokens: Option<usize>,
    pub history_drafts: Option<bool>,
    pub adaptive_verification: Option<bool>,
}
impl Default for RuntimeSettings {
    fn default() -> Self {
        Self {
            memory_limit: String::new(),
            kv_cache_limit: "memory".into(),
            host: "127.0.0.1".into(),
            port: 8000,
            max_concurrency: None,
            verify_weights: false,
            prefix_cache_memory: String::new(),
            prefix_cache_disk: String::new(),
            prefix_cache_disk_limit: String::new(),
            prefix_cache_disk_enabled: false,
            prefix_cache_max_age: "off".into(),
            prefix_cache_drain_on_shutdown: "off".into(),
            embedding_table_storage: "auto".into(),
            embedding_table_cache_size: String::new(),
            mtp_draft_tokens: None,
            history_drafts: None,
            adaptive_verification: None,
        }
    }
}
impl RuntimeSettings {
    pub fn validate(&self) -> Result<(), String> {
        self.apply(LoadOptions::default(), PathBuf::from("prefix-cache"))
            .map(|_| ())
    }
    pub fn bind(&self) -> Result<SocketAddr, String> {
        if self.port == 0 {
            return Err("Listen port must be 1 through 65535".into());
        }
        if self.host.len() > 128 {
            return Err("Listen host must be an IP address".into());
        }
        let host: IpAddr = self
            .host
            .trim()
            .parse()
            .map_err(|_| "Listen host must be an IPv4 or IPv6 address")?;
        Ok(SocketAddr::new(host, self.port))
    }
    pub(crate) fn apply(
        &self,
        mut base: LoadOptions,
        default_disk: PathBuf,
    ) -> Result<LoadOptions, String> {
        self.bind()?;
        for (name, value) in [
            ("memory limit", &self.memory_limit),
            ("KV limit", &self.kv_cache_limit),
            ("prefix memory", &self.prefix_cache_memory),
            ("prefix disk limit", &self.prefix_cache_disk_limit),
            ("prefix expiration", &self.prefix_cache_max_age),
            ("shutdown drain", &self.prefix_cache_drain_on_shutdown),
            ("embedding storage", &self.embedding_table_storage),
            ("embedding cache", &self.embedding_table_cache_size),
        ] {
            if value.len() > 64 {
                return Err(format!("{name} value is too long"));
            }
        }
        if self.prefix_cache_disk.len() > 32768 || self.prefix_cache_disk.contains('\0') {
            return Err("Prefix cache directory is invalid or too long".into());
        }
        if self.max_concurrency.is_some_and(|n| !(1..=64).contains(&n)) {
            return Err("Maximum concurrency must be 1 through 64, or automatic".into());
        }
        if !self.memory_limit.trim().is_empty() {
            base.memory_limit = Some(self.memory_limit.parse::<MemoryLimit>()?);
        }
        base.kv_cache_limit = self.kv_cache_limit.trim().parse::<KvCacheLimit>()?;
        if let Some(n) = self.max_concurrency {
            base.max_concurrency = Some(n);
        }
        base.mtp_draft_tokens = self.mtp_draft_tokens;
        base.history_drafts = self.history_drafts;
        base.adaptive_verification = self.adaptive_verification;
        base.embedding_table_storage = match self.embedding_table_storage.trim() {
            "" | "auto" => None,
            "disk" => Some(Storage::Disk),
            "ram" => Some(Storage::Ram),
            _ => return Err("Embedding storage must be auto, disk, or ram".into()),
        };
        base.embedding_table_cache_bytes =
            budget(&self.embedding_table_cache_size, "Embedding row cache")?;
        if base.embedding_table_storage == Some(Storage::Ram)
            && base.embedding_table_cache_bytes.is_some_and(|n| n != 0)
        {
            return Err("Embedding row cache requires disk table storage".into());
        }
        if let Some(bytes) = budget(&self.prefix_cache_memory, "Prefix memory")? {
            base.prefix_cache.memory_bytes = Some(bytes);
        }
        base.prefix_cache.disk = if !self.prefix_cache_disk_enabled {
            None
        } else if self.prefix_cache_disk.trim().is_empty() {
            Some(base.prefix_cache.disk.unwrap_or(default_disk))
        } else {
            Some(PathBuf::from(self.prefix_cache_disk.trim()))
        };
        base.prefix_cache.disk_limit = budget(&self.prefix_cache_disk_limit, "Prefix disk limit")?;
        base.prefix_cache.max_age =
            parse_max_age(if self.prefix_cache_max_age.trim().is_empty() {
                "off"
            } else {
                self.prefix_cache_max_age.trim()
            })?;
        base.prefix_cache.drain_on_shutdown =
            parse_drain_on_shutdown(if self.prefix_cache_drain_on_shutdown.trim().is_empty() {
                "off"
            } else {
                self.prefix_cache_drain_on_shutdown.trim()
            })?;
        Ok(base)
    }
}
fn budget(value: &str, name: &str) -> Result<Option<u64>, String> {
    if value.trim().is_empty() {
        Ok(None)
    } else {
        parse_size(value.trim())
            .map(Some)
            .map_err(|e| format!("{name}: {e}"))
    }
}
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RuntimeIdentity {
    bind: SocketAddr,
    verify_weights: bool,
    kv_cache_limit: KvCacheLimit,
    memory_limit: Option<MemoryLimit>,
    max_concurrency: Option<usize>,
    mtp_draft_tokens: Option<usize>,
    history_drafts: Option<bool>,
    adaptive_verification: Option<bool>,
    embedding_table_storage: Option<Storage>,
    embedding_table_cache_bytes: Option<u64>,
    prefix_cache: flare_helios::offload::PrefixCacheOffload,
}
impl RuntimeIdentity {
    pub(crate) fn new(runtime: &RuntimeSettings, options: &LoadOptions) -> Result<Self, String> {
        Ok(Self {
            bind: runtime.bind()?,
            verify_weights: runtime.verify_weights,
            kv_cache_limit: options.kv_cache_limit,
            memory_limit: options.memory_limit,
            max_concurrency: options.max_concurrency,
            mtp_draft_tokens: options.mtp_draft_tokens,
            history_drafts: options.history_drafts,
            adaptive_verification: options.adaptive_verification,
            embedding_table_storage: Some(options.embedding_table_storage.unwrap_or_default()),
            embedding_table_cache_bytes: options.embedding_table_cache_bytes,
            prefix_cache: options.prefix_cache.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn defaults_disable_disk_and_weight_verification() {
        let runtime = RuntimeSettings::default();
        let options = runtime
            .apply(LoadOptions::default(), PathBuf::from("cache/prefix-cache"))
            .unwrap();
        assert!(!runtime.verify_weights);
        assert_eq!(options.prefix_cache.disk, None);
        assert_eq!(options.prefix_cache.disk_limit, None);
        assert_eq!(options.kv_cache_limit, KvCacheLimit::Memory);
        assert_eq!(options.memory_limit, None);
    }
    #[test]
    fn memory_and_kv_modes_reuse_engine_semantics() {
        for (text, expected) in [
            ("30G", MemoryLimit::Bytes(30 << 30)),
            ("-8G", MemoryLimit::Reserve(8 << 30)),
            ("0.8", MemoryLimit::Fraction(0.8)),
        ] {
            let runtime = RuntimeSettings {
                memory_limit: text.into(),
                kv_cache_limit: "context".into(),
                ..Default::default()
            };
            let options = runtime
                .apply(LoadOptions::default(), "prefix".into())
                .unwrap();
            assert_eq!(options.memory_limit, Some(expected));
            assert_eq!(options.kv_cache_limit, KvCacheLimit::Context);
            assert_eq!(options.max_concurrency, None);
        }
    }
    #[test]
    fn cache_controls_inherit_configuration_and_allow_explicit_disk_disable() {
        let base = LoadOptions {
            prefix_cache: flare_helios::offload::PrefixCacheOffload {
                memory_bytes: Some(1024),
                disk: Some("configured".into()),
                disk_limit: Some(999),
                ..Default::default()
            },
            ..Default::default()
        };
        let mut runtime = RuntimeSettings {
            prefix_cache_disk_enabled: true,
            prefix_cache_max_age: "2m".into(),
            prefix_cache_drain_on_shutdown: "90s".into(),
            ..Default::default()
        };
        let enabled = runtime.apply(base.clone(), "fallback".into()).unwrap();
        assert_eq!(enabled.prefix_cache.memory_bytes, Some(1024));
        assert_eq!(enabled.prefix_cache.disk, Some("configured".into()));
        assert_eq!(enabled.prefix_cache.disk_limit, None);
        assert_eq!(
            enabled.prefix_cache.max_age,
            std::time::Duration::from_secs(120)
        );
        assert_eq!(
            enabled.prefix_cache.drain_on_shutdown,
            std::time::Duration::from_secs(90)
        );
        runtime.prefix_cache_disk_enabled = false;
        assert_eq!(
            runtime
                .apply(base, "fallback".into())
                .unwrap()
                .prefix_cache
                .disk,
            None
        );
    }
    #[test]
    fn malformed_limits_hosts_and_embedding_combinations_are_rejected() {
        for runtime in [
            RuntimeSettings {
                memory_limit: "0.0".into(),
                ..Default::default()
            },
            RuntimeSettings {
                memory_limit: "nan".into(),
                ..Default::default()
            },
            RuntimeSettings {
                kv_cache_limit: "bytes".into(),
                ..Default::default()
            },
            RuntimeSettings {
                host: "hostname".into(),
                ..Default::default()
            },
            RuntimeSettings {
                port: 0,
                ..Default::default()
            },
            RuntimeSettings {
                max_concurrency: Some(0),
                ..Default::default()
            },
            RuntimeSettings {
                max_concurrency: Some(65),
                ..Default::default()
            },
            RuntimeSettings {
                embedding_table_storage: "ram".into(),
                embedding_table_cache_size: "1G".into(),
                ..Default::default()
            },
            RuntimeSettings {
                prefix_cache_max_age: "invalid".into(),
                ..Default::default()
            },
        ] {
            assert!(runtime.validate().is_err());
        }
        assert!(
            RuntimeSettings {
                embedding_table_storage: String::new(),
                host: "0.0.0.0".into(),
                ..Default::default()
            }
            .validate()
            .is_ok()
        );
    }
}
