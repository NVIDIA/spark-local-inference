/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::{fs, path::PathBuf};

use flare_helios::frontend::images::{ImageLimits, ImageProcessor, RgbImage};
use flare_helios::frontend::{Asset, Frontend, FrontendAssets};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

fn pattern(width: usize, height: usize, mut seed: u32) -> Vec<u8> {
    (0..width * height * 3)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed as u8
        })
        .collect()
}

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn check_hash(
    name: &str,
    field: &str,
    bytes: &[u8],
    expected: &Value,
    diagnostics: Option<&PathBuf>,
) -> Result<(), String> {
    let actual = hash(bytes);
    if expected.as_str() != Some(&actual) {
        if let Some(directory) = diagnostics {
            fs::create_dir_all(directory).map_err(|error| error.to_string())?;
            fs::write(directory.join(format!("{name}.{field}")), bytes)
                .map_err(|error| error.to_string())?;
        }
        return Err(format!(
            "{name}: {field} differs: expected {expected}, received {actual}"
        ));
    }
    Ok(())
}

fn main() -> Result<(), String> {
    run().map_err(|error| error.to_string())
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<_> = std::env::args_os().collect();
    if !(3..=4).contains(&arguments.len()) {
        return Err(
            "usage: test_images ABSOLUTE_SNAPSHOT FIXTURE_JSON [MISMATCH_BINARY_DIRECTORY]".into(),
        );
    }
    let snapshot = PathBuf::from(&arguments[1]);
    let fixture_path = PathBuf::from(&arguments[2]);
    let diagnostics = arguments.get(3).map(PathBuf::from);
    let fixture: Value =
        serde_json::from_slice(&fs::read(&fixture_path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let asset = |name: &str| Asset {
        range: None,
        path: snapshot.join(name),
        sha256: fixture["assets"][name].as_str().unwrap().into(),
    };
    let processor = ImageProcessor::load(
        flare_helios::frontend::test_support::image(
            &asset("preprocessor_config.json"),
            &asset("config.json"),
        )
        .unwrap(),
        ImageLimits::default(),
    )?;
    let frontend = Frontend::load(
        FrontendAssets {
            tokenizer: asset("tokenizer.json"),
            tokens: flare_helios::frontend::test_support::tokens(&asset("tokenizer_config.json"))
                .unwrap(),
            chat_template: Some(asset("chat_template.jinja")),
        },
        Map::new(),
    )?;
    let mut errors = Vec::new();
    for case in fixture["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let mut images = Vec::new();
        for (index, input) in case["inputs"].as_array().unwrap().iter().enumerate() {
            let image = if let Some(path) = input["png"].as_str() {
                let bytes = fs::read(fixture_path.parent().unwrap().join(path))
                    .map_err(|e| e.to_string())?;
                check_hash(name, "png input", &bytes, &input["sha256"], None)?;
                processor.decode_png(&bytes)?
            } else {
                let width = input["width"].as_u64().unwrap() as usize;
                let height = input["height"].as_u64().unwrap() as usize;
                RgbImage::new(
                    width,
                    height,
                    pattern(width, height, input["seed"].as_u64().unwrap() as u32),
                )?
            };
            check_hash(
                name,
                "input RGB",
                image.pixels(),
                &case["input_rgb_sha256"][index],
                None,
            )?;
            let (_, pixels) = processor.resized_rgb(&image)?;
            if let Err(error) = check_hash(
                name,
                &format!("resize{index}.u8"),
                &pixels,
                &case["resized_rgb_sha256"][index],
                diagnostics.as_ref(),
            ) {
                errors.push(error);
            }
            images.push(image);
        }
        let processed = processor.process(&images)?;
        if json!(
            processed
                .geometry
                .iter()
                .map(|g| g.grid_thw)
                .collect::<Vec<_>>()
        ) != case["grid_thw"]
            || json!(processed.cu_seqlens) != case["cu_seqlens"]
        {
            errors.push(format!(
                "{name}: image grids or cumulative sequence lengths differ"
            ));
        }
        let rendered = frontend.render_chat(
            case["messages"].as_array().unwrap(),
            json!({"add_generation_prompt": true}).as_object().unwrap(),
        )?;
        let prompt = processor.expand_prompt(&frontend, &rendered.text, &processed)?;
        if rendered.text != case["rendered"]
            || prompt.text != case["expanded"]
            || json!(prompt.token_ids) != case["token_ids"]
            || json!(prompt.mm_token_type_ids) != case["mm_token_type_ids"]
            || json!(prompt.rope_delta) != case["rope_delta"]
        {
            errors.push(format!(
                "{name}: prompt expansion, token IDs, modality IDs or rope delta differ"
            ));
        }
        for (field, bytes) in [
            (
                "pixel_values",
                processed
                    .pixel_values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>(),
            ),
            (
                "position_ids",
                processed
                    .position_ids
                    .iter()
                    .flatten()
                    .flat_map(|value| value.to_le_bytes())
                    .collect(),
            ),
            (
                "interpolation_indices",
                processed
                    .interpolation_indices
                    .iter()
                    .flatten()
                    .flat_map(|value| value.to_le_bytes())
                    .collect(),
            ),
            (
                "interpolation_weights",
                processed
                    .interpolation_weights
                    .iter()
                    .flatten()
                    .flat_map(|value| value.to_le_bytes())
                    .collect(),
            ),
            (
                "mrope",
                prompt.position_ids[1..]
                    .iter()
                    .flatten()
                    .flat_map(|value| value.to_le_bytes())
                    .collect(),
            ),
        ] {
            if let Err(error) = check_hash(
                name,
                field,
                &bytes,
                &case["hashes"][field],
                diagnostics.as_ref(),
            ) {
                errors.push(error);
            }
        }
        println!("{name}: {} patches checked", processed.position_ids.len());
    }
    let large_limits = ImageLimits {
        max_patches_per_image: 65_536,
        max_total_patches: 65_536,
        ..ImageLimits::default()
    };
    let geometry_processor = ImageProcessor::load(
        flare_helios::frontend::test_support::image(
            &asset("preprocessor_config.json"),
            &asset("config.json"),
        )
        .unwrap(),
        large_limits,
    )?;
    for case in fixture["geometry_cases"].as_array().unwrap() {
        let (width, height) = (
            case["width"].as_u64().unwrap() as usize,
            case["height"].as_u64().unwrap() as usize,
        );
        match geometry_processor.geometry(width, height) {
            Ok(geometry) if json!([geometry.height, geometry.width]) == case["resized"] => {}
            Err(_) if case.get("error").is_some() => {}
            _ => errors.push(format!("geometry mismatch for {width}x{height}")),
        }
    }
    if !errors.is_empty() {
        return Err((errors.join("\n")).into());
    }
    println!(
        "Exact image qualification passed: {} pipeline cases and {} geometry cases",
        fixture["cases"].as_array().unwrap().len(),
        fixture["geometry_cases"].as_array().unwrap().len()
    );
    Ok(())
}
