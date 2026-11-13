// SPDX-License-Identifier: Apache-2.0
//! Published Hub discovery hints; signed core inspection remains authoritative.
use serde::{Deserialize, de::DeserializeOwned};
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

fn null_default<'de, D: serde::Deserializer<'de>, T: Deserialize<'de> + Default>(
    deserializer: D,
) -> Result<T, D::Error> {
    Ok(Option::<T>::deserialize(deserializer)?.unwrap_or_default())
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct CatalogEntry {
    #[serde(deserialize_with = "null_default")]
    pub id: String,
    #[serde(deserialize_with = "null_default")]
    pub downloads: u64,
    #[serde(deserialize_with = "null_default")]
    pub tags: Vec<String>,
    #[serde(deserialize_with = "null_default")]
    pub author: String,
    #[serde(alias = "pipeline_tag", deserialize_with = "null_default")]
    pub task: String,
    #[serde(deserialize_with = "null_default")]
    pub license: String,
    #[serde(deserialize_with = "null_default")]
    pub quantization: String,
    pub context_tokens: Option<u64>,
    pub parameter_count: Option<u64>,
    #[serde(deserialize_with = "null_default")]
    pub hardware: String,
    /// Published platform branches matching the requested hardware are a display
    /// hint; they do not establish core compatibility or memory fit.
    #[serde(skip_deserializing)]
    pub build_target_match: Option<bool>,
    #[serde(deserialize_with = "null_default")]
    pub revision: String,
    #[serde(deserialize_with = "null_default")]
    pub revision_commit: String,
    #[serde(deserialize_with = "null_default")]
    pub revision_updated_at: String,
    /// Every file in one commit's inventory, including documentation. This is
    /// neither the selected deployment's transfer size nor its memory budget.
    pub revision_bytes: Option<u64>,
    #[serde(deserialize_with = "null_default")]
    pub base_model: String,
}

#[derive(Clone, Default, Deserialize)]
#[serde(default)]
struct HubEntry {
    #[serde(flatten)]
    entry: CatalogEntry,
    #[serde(deserialize_with = "null_default")]
    siblings: Vec<Sibling>,
    config: Value,
    #[serde(rename = "cardData")]
    card: Value,
    #[serde(deserialize_with = "null_default")]
    sha: String,
    #[serde(rename = "lastModified", deserialize_with = "null_default")]
    last_modified: String,
}
#[derive(Clone, Deserialize)]
struct Sibling {
    rfilename: String,
    #[serde(default)]
    size: Option<u64>,
}
#[derive(Deserialize)]
struct Refs {
    branches: Vec<Branch>,
}
#[derive(Deserialize)]
struct Branch {
    name: String,
}

fn explicit_core(item: &HubEntry) -> bool {
    item.entry
        .tags
        .iter()
        .any(|tag| matches!(tag.as_str(), "flare-runtime-core" | "flare-runtime-package"))
        || matches!(
            item.config["model_type"].as_str(),
            Some("flare-runtime-core" | "flare-runtime-package")
        )
}
fn prepared_layout(item: &HubEntry) -> bool {
    item.siblings
        .iter()
        .any(|s| s.rfilename == "model.safetensors")
        && item.siblings.iter().any(|s| {
            s.rfilename.starts_with("deployments/") && s.rfilename.ends_with(".safetensors")
        })
}
fn bounded(mut value: String) -> String {
    if value.len() > 512 {
        let mut end = 512;
        while !value.is_char_boundary(end) {
            end -= 1;
        }
        value.truncate(end);
    }
    value
}
fn strings(value: &Value) -> String {
    bounded(match value {
        Value::String(value) => value.clone(),
        Value::Array(values) => values
            .iter()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>()
            .join(", "),
        _ => String::new(),
    })
}
fn apply_config(entry: &mut CatalogEntry, config: &Value) {
    if let Some(context) = config["flare"]["serving_context"]
        .as_u64()
        .filter(|n| *n > 0)
    {
        entry.context_tokens = Some(context);
    }
    // Packed safetensors element counts do not measure logical model parameters.
    if let Some(parameters) = config["flare"]["parameter_count"]
        .as_u64()
        .or_else(|| config["num_parameters"].as_u64())
        .filter(|n| *n > 0)
    {
        entry.parameter_count = Some(parameters);
    }
    let method = config["quantization_config"]["quant_method"]
        .as_str()
        .or_else(|| config["text_config"]["quantization_config"]["quant_method"].as_str());
    if entry.quantization.is_empty() {
        if let Some(method) = method.filter(|method| {
            matches!(
                *method,
                "nvfp4" | "mxfp4" | "mxfp8" | "fp8" | "int8" | "int4"
            )
        }) {
            entry.quantization = method.to_ascii_uppercase();
        }
    }
}
fn published_metadata(item: &mut HubEntry) {
    if item.entry.author.is_empty() {
        item.entry.author = item
            .entry
            .id
            .split_once('/')
            .map_or("", |(owner, _)| owner)
            .into();
    }
    if item.entry.task.is_empty() {
        item.entry.task = strings(&item.card["pipeline_tag"]);
    }
    item.entry.author = bounded(std::mem::take(&mut item.entry.author));
    item.entry.task = bounded(std::mem::take(&mut item.entry.task));
    item.entry.license = strings(&item.card["license"]);
    if item.entry.license == "other" {
        let name = strings(&item.card["license_name"]);
        if !name.is_empty() {
            item.entry.license = name;
        }
    }
    item.entry.base_model = strings(&item.card["base_model"]);
    item.entry.parameter_count = item.card["parameter_count"].as_u64().filter(|n| *n > 0);
    let quantization: BTreeSet<_> = item
        .entry
        .tags
        .iter()
        .filter_map(|tag| {
            matches!(
                tag.as_str(),
                "nvfp4"
                    | "mxfp4"
                    | "mxfp8"
                    | "fp8"
                    | "int8"
                    | "int4"
                    | "exl2"
                    | "exl3"
                    | "awq"
                    | "gptq"
            )
            .then(|| tag.to_ascii_uppercase())
        })
        .collect();
    item.entry.quantization = quantization.into_iter().collect::<Vec<_>>().join(" / ");
    apply_config(&mut item.entry, &item.config);
}
fn candidates(entries: impl IntoIterator<Item = HubEntry>, query: &str) -> Vec<HubEntry> {
    let query = query.trim().to_lowercase();
    let mut unique = BTreeMap::new();
    for mut item in entries {
        let Some((_, name)) = item.entry.id.split_once('/') else {
            continue;
        };
        let publisher = item
            .siblings
            .iter()
            .any(|file| file.rfilename == "publisher.json");
        let named_and_tagged = name.to_ascii_lowercase().contains("-flare")
            && item.entry.tags.iter().any(|tag| tag == "flare");
        if !(explicit_core(&item) || prepared_layout(&item) || publisher || named_and_tagged)
            || item.entry.id.len() > 512
            || !item.entry.id.to_lowercase().contains(&query)
            || !matches!(crate::hub::repository(&item.entry.id), Ok((_, None)))
        {
            continue;
        }
        published_metadata(&mut item);
        unique.entry(item.entry.id.clone()).or_insert(item);
    }
    let mut entries: Vec<_> = unique.into_values().collect();
    entries.sort_by(|a, b| {
        b.entry
            .downloads
            .cmp(&a.entry.downloads)
            .then_with(|| a.entry.id.cmp(&b.entry.id))
    });
    entries.truncate(50);
    entries
}
fn read<T: DeserializeOwned>(
    agent: &ureq::Agent,
    url: &str,
    limit: u64,
) -> crate::settings::Result<T> {
    let text = agent
        .get(url)
        .call()?
        .body_mut()
        .with_config()
        .limit(limit)
        .read_to_string()?;
    Ok(serde_json::from_str(&text)?)
}
fn fetch(agent: &ureq::Agent, tag_search: bool) -> crate::settings::Result<Vec<HubEntry>> {
    let mut url = url::Url::parse("https://huggingface.co/api/models")?;
    url.query_pairs_mut()
        .append_pair(if tag_search { "filter" } else { "search" }, "flare")
        .append_pair("full", "true")
        .append_pair("cardData", "true")
        .append_pair("limit", "500")
        .append_pair("sort", "downloads")
        .append_pair("direction", "-1");
    read(agent, url.as_str(), 4 * 1024 * 1024)
}
fn optional<T: DeserializeOwned>(
    agent: &ureq::Agent,
    url: &str,
    limit: u64,
    deadline: Instant,
) -> Option<T> {
    (Instant::now() < deadline)
        .then(|| read(agent, url, limit).ok())
        .flatten()
}
fn branch_platform(name: &str) -> Option<&'static str> {
    flare_helios::core::naming::PLATFORMS
        .into_iter()
        .find(|platform| {
            name == *platform
                || name
                    .strip_prefix(platform)
                    .is_some_and(|suffix| suffix.starts_with('-'))
        })
}
fn revision_inventory(item: &HubEntry) -> Option<u64> {
    if item.siblings.is_empty()
        || item.sha.len() != 40
        || !item.sha.bytes().all(|b| b.is_ascii_hexdigit())
    {
        return None;
    }
    item.siblings
        .iter()
        .try_fold(0u64, |total, file| total.checked_add(file.size?))
}
fn select_revision<'a>(
    branches: &'a BTreeSet<String>,
    hardware: Option<&str>,
) -> Option<&'a String> {
    hardware
        .and_then(|target| {
            branches.get(target).or_else(|| {
                branches.iter().find(|name| branch_platform(name) == Some(target))
            })
        })
        .or_else(|| branches.iter().next())
}
fn apply_refs(
    entry: &mut CatalogEntry,
    refs: Refs,
    hardware: Option<&str>,
) -> BTreeSet<String> {
    let branches: BTreeSet<_> = refs
        .branches
        .into_iter()
        .map(|branch| branch.name)
        .filter(|name| name.len() <= 512 && branch_platform(name).is_some())
        .collect();
    let platforms: BTreeSet<_> = branches
        .iter()
        .filter_map(|name| branch_platform(name))
        .collect();
    entry.build_target_match = hardware
        .filter(|target| flare_helios::core::naming::PLATFORMS.contains(target))
        .filter(|_| !platforms.is_empty())
        .map(|target| platforms.contains(target));
    entry.hardware = platforms
        .into_iter()
        .map(str::to_ascii_uppercase)
        .collect::<Vec<_>>()
        .join(" / ");
    branches
}
fn apply_revision(entry: &mut CatalogEntry, revision: &str, remote: &HubEntry) {
    entry.revision = revision.into();
    entry.revision_bytes = revision_inventory(remote);
    entry.revision_commit = remote.sha.clone();
    entry.revision_updated_at = bounded(remote.last_modified.clone());
    apply_config(entry, &remote.config);
}
fn enrich(
    mut item: HubEntry,
    hardware: Option<&str>,
    agent: &ureq::Agent,
    deadline: Instant,
) -> Option<CatalogEntry> {
    let base = format!("https://huggingface.co/api/models/{}", item.entry.id);
    if item
        .siblings
        .iter()
        .any(|file| file.rfilename == "config.json")
    {
        let revision = if item.sha.len() == 40 && item.sha.bytes().all(|b| b.is_ascii_hexdigit()) {
            &item.sha
        } else {
            "main"
        };
        if let Some(config) = optional::<Value>(
            agent,
            &format!(
                "https://huggingface.co/{}/raw/{revision}/config.json",
                item.entry.id
            ),
            64 * 1024,
            deadline,
        ) {
            item.config = config;
            apply_config(&mut item.entry, &item.config);
        }
    }
    if let Some(refs) = optional::<Refs>(agent, &format!("{base}/refs"), 256 * 1024, deadline) {
        let branches = apply_refs(&mut item.entry, refs, hardware);
        let revision = select_revision(&branches, hardware);
        if let Some(revision) = revision {
            let encoded: String =
                url::form_urlencoded::byte_serialize(revision.as_bytes()).collect();
            if let Some(remote) = optional::<HubEntry>(
                agent,
                &format!("{base}/revision/{encoded}?blobs=true"),
                2 * 1024 * 1024,
                deadline,
            ) {
                apply_revision(&mut item.entry, revision, &remote);
                if remote
                    .siblings
                    .iter()
                    .any(|file| file.rfilename == "config.json")
                    && remote.sha.len() == 40
                    && remote.sha.bytes().all(|b| b.is_ascii_hexdigit())
                {
                    if let Some(config) = optional::<Value>(
                        agent,
                        &format!(
                            "https://huggingface.co/{}/raw/{}/config.json",
                            item.entry.id, remote.sha
                        ),
                        64 * 1024,
                        deadline,
                    ) {
                        apply_config(&mut item.entry, &config);
                    }
                }
            }
        }
    }
    Some(item.entry)
}
/// Discover published core exports. Hub metadata never establishes trust or fit.
pub fn search(query: &str) -> crate::settings::Result<Vec<CatalogEntry>> {
    search_for_hardware(query, None)
}

/// Enrich at most 50 candidates on four workers. No enrichment request starts
/// after 12 seconds; an in-flight request has a two-second timeout.
/// If the requested branch is absent, another published platform branch supplies
/// display metadata only; `revision` records that branch without asserting fit.
pub fn search_for_hardware(
    query: &str,
    hardware: Option<&str>,
) -> crate::settings::Result<Vec<CatalogEntry>> {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(20)))
        .build()
        .into();
    let (named, tagged) = std::thread::scope(|scope| {
        let named = scope.spawn(|| fetch(&agent, false));
        let tagged = fetch(&agent, true);
        (named.join(), tagged)
    });
    let named = named.map_err(|_| "Hugging Face discovery worker failed")?;
    let entries = match (named, tagged) {
        (Ok(named), Ok(tagged)) => candidates(named.into_iter().chain(tagged), query),
        (Ok(entries), Err(_)) | (Err(_), Ok(entries)) => candidates(entries, query),
        (Err(error), Err(_)) => return Err(error),
    };
    let slots: Vec<_> = entries
        .iter()
        .map(|item| Mutex::new(Some(item.entry.clone())))
        .collect();
    let cursor = AtomicUsize::new(0);
    let deadline = Instant::now() + Duration::from_secs(12);
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(2)))
        .build()
        .into();
    std::thread::scope(|scope| {
        for _ in 0..4 {
            scope.spawn(|| {
                loop {
                    if Instant::now() >= deadline {
                        break;
                    }
                    let index = cursor.fetch_add(1, Ordering::Relaxed);
                    let Some(item) = entries.get(index) else {
                        break;
                    };
                    *slots[index].lock().unwrap_or_else(|e| e.into_inner()) =
                        enrich(item.clone(), hardware, &agent, deadline);
                }
            });
        }
    });
    Ok(slots
        .into_iter()
        .filter_map(|slot| slot.into_inner().unwrap_or_else(|e| e.into_inner()))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn item(value: Value) -> HubEntry {
        serde_json::from_value(value).unwrap()
    }
    #[test]
    fn null_optional_fields_preserve_valid_repository_metadata() {
        let mut value = item(
            json!({"id":"author/core","author":null,"pipeline_tag":null,"downloads":null,"tags":null,"siblings":null,"sha":null,"lastModified":null,"cardData":null,"config":null}),
        );
        published_metadata(&mut value);
        assert_eq!(value.entry.author, "author");
        assert!(value.entry.task.is_empty());
        assert!(value.entry.tags.is_empty());
        assert_eq!(value.entry.downloads, 0);
        assert!(value.last_modified.is_empty());
    }
    #[test]
    fn published_metadata_never_infers_context_parameters_or_precision_from_names() {
        let mut value = item(
            json!({"id":"author/27B-NVFP4-1M","pipeline_tag":"text-generation","tags":["nvfp4","mxfp8"],"cardData":{"license":"other","license_name":"lil-license-1.0","base_model":"upstream/model"},"config":{"max_position_embeddings":1048576},"safetensors":{"total":123}}),
        );
        published_metadata(&mut value);
        assert_eq!(value.entry.task, "text-generation");
        assert_eq!(value.entry.license, "lil-license-1.0");
        assert_eq!(value.entry.base_model, "upstream/model");
        assert_eq!(value.entry.quantization, "MXFP8 / NVFP4");
        assert_eq!(value.entry.context_tokens, None);
        assert_eq!(value.entry.parameter_count, None);
        apply_config(
            &mut value.entry,
            &json!({"flare":{"serving_context":1048576,"parameter_count":27000000000u64},"quantization_config":{"quant_method":"compressed-tensors"}}),
        );
        assert_eq!(value.entry.context_tokens, Some(1048576));
        assert_eq!(value.entry.parameter_count, Some(27000000000));
        assert_eq!(value.entry.quantization, "MXFP8 / NVFP4");
        let mut unknown = item(
            json!({"id":"a/NVFP4-27B-1M","config":{"quantization_config":{"quant_method":"modelopt"}}}),
        );
        published_metadata(&mut unknown);
        assert!(unknown.entry.quantization.is_empty());
        assert_eq!(unknown.entry.context_tokens, None);
        assert_eq!(unknown.entry.parameter_count, None);
    }
    #[test]
    fn revision_inventory_sums_all_files_not_repository_storage_or_one_deployment() {
        let mut value = item(
            json!({"sha":"1".repeat(40),"usedStorage":999999,"siblings":[{"rfilename":"README.md","size":13},{"rfilename":"model.safetensors","size":100},{"rfilename":"deployments/tp1/data.safetensors","size":20},{"rfilename":"deployments/tp8/data.safetensors","size":30}]}),
        );
        assert_eq!(revision_inventory(&value), Some(163));
        value.siblings[0].size = None;
        assert_eq!(revision_inventory(&value), None);
        value.siblings[0].size = Some(u64::MAX);
        assert_eq!(revision_inventory(&value), None);
        value.siblings[0].size = Some(13);
        value.sha = "invalid".into();
        assert_eq!(revision_inventory(&value), None);
    }
    #[test]
    fn target_selection_and_display_fallback_are_deterministic() {
        let branches = ["gb202", "gb10", "gb202-tp8"]
            .map(String::from)
            .into_iter()
            .collect();
        assert_eq!(
            select_revision(&branches, Some("gb202")).map(String::as_str),
            Some("gb202")
        );
        assert_eq!(
            select_revision(&branches, Some("gb300")).map(String::as_str),
            Some("gb10")
        );
        assert_eq!(branch_platform("gb202-tp8"), Some("gb202"));
        assert_eq!(branch_platform("gb202unrelated"), None);
        assert_eq!(branch_platform("main"), None);
        let suffixed = ["gb10", "gb202-tp8"].map(String::from).into_iter().collect();
        assert_eq!(
            select_revision(&suffixed, Some("gb202")).map(String::as_str),
            Some("gb202-tp8")
        );
    }
    #[test]
    fn published_build_match_uses_all_platform_branches() {
        for target in flare_helios::core::naming::PLATFORMS {
            for published in flare_helios::core::naming::PLATFORMS {
                for suffix in ["", "-tp8"] {
                    let name = format!("{published}{suffix}");
                    let refs = serde_json::from_value(json!({"branches": [{"name": name}]})).unwrap();
                    let mut entry = CatalogEntry::default();
                    apply_refs(&mut entry, refs, Some(target));
                    assert_eq!(entry.hardware, published.to_ascii_uppercase());
                    assert_eq!(entry.build_target_match, Some(target == published));
                }
            }
            let refs = serde_json::from_value(json!({"branches": [
                {"name": "gb10-tp1"}, {"name": "gb202"}, {"name": "gb300-tp8"}
            ]})).unwrap();
            let mut entry = CatalogEntry::default();
            let branches = apply_refs(&mut entry, refs, Some(target));
            assert_eq!(entry.build_target_match, Some(true));
            assert_eq!(branch_platform(select_revision(&branches, Some(target)).unwrap()), Some(target));
        }
    }
    #[test]
    fn unknown_builds_or_hardware_do_not_hide_discovery_entries() {
        for names in [vec![], vec!["main"], vec!["gb10unrelated", "custom-build"]] {
            let refs = serde_json::from_value(json!({"branches": names.into_iter()
                .map(|name| json!({"name": name})).collect::<Vec<_>>()})).unwrap();
            let mut entry = CatalogEntry::default();
            apply_refs(&mut entry, refs, Some("gb10"));
            assert_eq!(entry.build_target_match, None);
        }
        for target in [None, Some("unknown")] {
            let refs = serde_json::from_value(json!({"branches": [{"name": "gb202"}]})).unwrap();
            let mut entry = CatalogEntry::default();
            apply_refs(&mut entry, refs, target);
            assert_eq!(entry.build_target_match, None);
        }
        let unverified = item(json!({"hardware": "GB202", "build_target_match": false}));
        assert_eq!(unverified.entry.build_target_match, None);
    }
    #[test]
    fn update_time_belongs_to_the_resolved_build_revision() {
        let mut repository = item(json!({
            "id": "author/core",
            "lastModified": "2026-11-09T03:41:33.000Z"
        }));
        assert!(repository.entry.revision_updated_at.is_empty());
        let remote = item(json!({
            "sha": "1".repeat(40),
            "lastModified": "2026-11-05T18:06:29.000Z",
            "siblings": [{"rfilename": "model.safetensors", "size": 123}]
        }));
        apply_revision(&mut repository.entry, "gb10", &remote);
        assert_eq!(repository.entry.revision, "gb10");
        assert_eq!(repository.entry.revision_commit, "1".repeat(40));
        assert_eq!(repository.entry.revision_bytes, Some(123));
        assert_eq!(
            repository.entry.revision_updated_at,
            "2026-11-05T18:06:29.000Z"
        );
        for missing in [json!({}), json!({"lastModified": null})] {
            apply_revision(&mut repository.entry, "gb202", &item(missing));
            assert_eq!(repository.entry.revision, "gb202");
            assert!(repository.entry.revision_updated_at.is_empty());
        }
    }
    #[test]
    fn optional_enrichment_failure_preserves_legacy_candidates() {
        let values = vec![item(
            json!({"id":"a/Qwen-FLARE","siblings":[{"rfilename":"publisher.json"}],"tags":["nvfp4"],"lastModified":"2026-11-09T03:41:33.000Z","cardData":{"license":"mit"}}),
        )];
        let mut found = candidates(values, " QWEN ");
        assert_eq!(found.len(), 1);
        let agent: ureq::Agent = ureq::Agent::config_builder().build().into();
        let result = enrich(
            found.remove(0),
            Some("gb10"),
            &agent,
            Instant::now() - Duration::from_secs(1),
        )
        .unwrap();
        assert_eq!(result.author, "a");
        assert_eq!(result.license, "mit");
        assert_eq!(result.quantization, "NVFP4");
        assert_eq!(result.revision_bytes, None);
        assert!(result.revision_updated_at.is_empty());
        assert_eq!(result.build_target_match, None);
    }
    #[test]
    fn query_deduplicates_sorts_and_rejects_generic_tag_matches() {
        let found = candidates(
            vec![
                item(json!({"id":"a/CORE","tags":["flare-runtime-package"],"downloads":3})),
                item(json!({"id":"a/CORE","tags":["flare-runtime-package"],"downloads":3})),
                item(json!({"id":"b/core","tags":["flare-runtime-core"],"downloads":5})),
                item(json!({"id":"a/core@main","tags":["flare-runtime-core"]})),
                item(json!({"id":"a/solar-forecaster","tags":["flare"]})),
                item(json!({"id":"a/Flare-TTS","tags":["flare"]})),
            ],
            "core",
        );
        assert_eq!(found.len(), 2);
        assert_eq!(found[0].entry.id, "b/core");
        assert_eq!(found[1].entry.id, "a/CORE");
    }
}
