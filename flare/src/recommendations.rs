// SPDX-License-Identifier: Apache-2.0
//! Curated display identities retain the actual prepared acquisition source.
use super::{ModelMetadata, ModelView};
pub(crate) const RECOMMENDED: u32 = 32;
pub(crate) const PREPARED_REPOSITORY: &str = "local-inference-lab/Qwen3.8-27B-FLARE";
pub(crate) const DISPLAY_MODEL: &str = "nvidia/Qwen3.8-27B-NVFP4";
pub(crate) fn is_recommended(repositories: &[String]) -> bool {
    repositories
        .iter()
        .any(|source| source == PREPARED_REPOSITORY)
}
pub(crate) fn local_name(fallback: &str, repositories: &[String]) -> String {
    if is_recommended(repositories) {
        DISPLAY_MODEL.into()
    } else {
        fallback.into()
    }
}
pub(crate) fn acquisition_name(source: &str) -> String {
    match crate::hub::repository(source) {
        Ok((repository, _)) if repository == PREPARED_REPOSITORY => DISPLAY_MODEL.into(),
        _ => source.into(),
    }
}
pub(crate) fn catalog(entries: &[ModelView]) -> Vec<ModelView> {
    let mut recommended = entries
        .iter()
        .find(|model| model.source == PREPARED_REPOSITORY)
        .cloned()
        .unwrap_or_else(|| ModelView {
            id: format!("hub:{PREPARED_REPOSITORY}"),
            source: PREPARED_REPOSITORY.into(),
            metadata: ModelMetadata {
                author: "local-inference-lab".into(),
                base_model: DISPLAY_MODEL.into(),
                ..Default::default()
            },
            flags: 16,
            ..Default::default()
        });
    recommended.name = DISPLAY_MODEL.into();
    recommended.flags |= RECOMMENDED;
    if recommended.metadata.base_model.is_empty() {
        recommended.metadata.base_model = DISPLAY_MODEL.into();
    }
    let mut result = vec![recommended];
    result.extend(
        entries
            .iter()
            .filter(|model| model.source != PREPARED_REPOSITORY)
            .cloned(),
    );
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn aliases_require_exact_prepared_repository_provenance() {
        let declared = "Qwen3.8-27B-NVIDIA-NVFP4-QAD-MXFP8";
        assert_eq!(
            local_name(declared, &[PREPARED_REPOSITORY.into()]),
            DISPLAY_MODEL
        );
        for unrelated in [
            "another/Qwen3.8-27B-FLARE",
            "local-inference-lab/Qwen3.8-27B-FLARE-other",
            "local-inference-lab/qwen3.8-27b-flare",
        ] {
            assert_eq!(local_name(declared, &[unrelated.into()]), declared);
            assert_eq!(acquisition_name(unrelated), unrelated);
        }
        assert_eq!(local_name(declared, &[]), declared);
        assert_eq!(
            acquisition_name(&format!("{PREPARED_REPOSITORY}@gb10")),
            DISPLAY_MODEL
        );
        assert!(!is_recommended(&[]));
    }
    #[test]
    fn recommendation_is_available_offline_without_invented_capabilities() {
        let entries = catalog(&[]);
        assert_eq!(entries.len(), 1);
        let model = &entries[0];
        assert_eq!(model.id, format!("hub:{PREPARED_REPOSITORY}"));
        assert_eq!(model.source, PREPARED_REPOSITORY);
        assert_eq!(model.name, DISPLAY_MODEL);
        assert_eq!(model.flags, 16 | RECOMMENDED);
        assert_eq!(model.logical_bytes, None);
        assert_eq!(model.metadata.context_tokens, None);
        assert_eq!(model.metadata.parameter_count, None);
        assert_eq!(model.metadata.revision_bytes, None);
        assert!(model.metadata.hardware.is_empty());
        assert_eq!(model.metadata.author, "local-inference-lab");
        assert_eq!(model.metadata.base_model, DISPLAY_MODEL);
    }
    #[test]
    fn network_enrichment_is_preserved_and_recommendation_is_deduplicated() {
        let entry = ModelView {
            source: PREPARED_REPOSITORY.into(),
            id: format!("hub:{PREPARED_REPOSITORY}"),
            name: "prepared export".into(),
            flags: 16,
            metadata: ModelMetadata {
                author: "local-inference-lab".into(),
                license: "declared-license".into(),
                context_tokens: Some(262144),
                revision_bytes: Some(123),
                ..Default::default()
            },
            ..Default::default()
        };
        let other = ModelView {
            source: "other/model".into(),
            name: "other model".into(),
            ..Default::default()
        };
        let merged = catalog(&[entry.clone(), entry, other]);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].source, PREPARED_REPOSITORY);
        assert_eq!(merged[0].name, DISPLAY_MODEL);
        assert_eq!(merged[0].metadata.license, "declared-license");
        assert_eq!(merged[0].metadata.context_tokens, Some(262144));
        assert_eq!(merged[0].metadata.revision_bytes, Some(123));
        assert_eq!(merged[1].name, "other model");
    }
}

#[cfg(test)]
mod cached_tests {
    use super::*;
    use crate::application::{LibraryEntry, VerificationState};
    #[test]
    fn cached_recommendation_uses_provenance_and_keeps_actual_core_and_download_ids() {
        let entry = LibraryEntry {
            model: "Qwen3.8-27B-NVIDIA-NVFP4-QAD-MXFP8".into(),
            model_key: "opaque-core-key".into(),
            version: "gb10-tp1-build".into(),
            selected: true,
            leased: false,
            verification: VerificationState::Unverified,
            hub_repositories: vec![PREPARED_REPOSITORY.into()],
            metadata: None,
            error: None,
        };
        let local = ModelView {
            id: "local:opaque-core-key@gb10-tp1-build".into(),
            name: local_name(&entry.model, &entry.hub_repositories),
            version: entry.version.clone(),
            path: "C:/cores/opaque/model.safetensors".into(),
            flags: 1 | RECOMMENDED,
            metadata: ModelMetadata {
                context_tokens: Some(262144),
                hardware: "GB10".into(),
                rank_count: Some(1),
                ..Default::default()
            },
            ..Default::default()
        };
        let matched = super::super::catalog_with_library(
            &catalog(&[]),
            &[entry],
            &[local.clone()],
            Some(&local.id),
        );
        assert_eq!(matched.len(), 1);
        let view = &matched[0];
        assert_eq!(view.id, local.id);
        assert_eq!(view.version, "gb10-tp1-build");
        assert_eq!(view.path, local.path);
        assert_eq!(view.name, DISPLAY_MODEL);
        assert_eq!(view.source, PREPARED_REPOSITORY);
        assert_eq!(view.flags, 1 | 16 | RECOMMENDED);
        assert_eq!(view.metadata.context_tokens, Some(262144));
        assert_eq!(view.metadata.author, "local-inference-lab");
        assert_eq!(view.metadata.license, "");
    }
    #[test]
    fn application_starts_with_an_offline_recommendation_without_a_cached_model() {
        let temp = tempfile::tempdir().unwrap();
        let config = temp.path().join("config.toml");
        std::fs::write(
            &config,
            format!(
                "[paths]\ncore_cache='{}'\n",
                temp.path()
                    .join("cores")
                    .display()
                    .to_string()
                    .replace('\\', "/")
            ),
        )
        .unwrap();
        let app = super::super::Controller::new(Some(config)).unwrap();
        let view = app.snapshot();
        let model = view
            .models
            .iter()
            .find(|model| model.flags & RECOMMENDED != 0)
            .unwrap();
        assert_eq!(model.name, DISPLAY_MODEL);
        assert_eq!(model.source, PREPARED_REPOSITORY);
        assert_eq!(model.flags & 1, 0);
        assert!(!view.has_metrics);
        assert!(view.active_model.is_empty());
        app.close();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while !app.is_closed() {
            assert!(std::time::Instant::now() < deadline);
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }
}
