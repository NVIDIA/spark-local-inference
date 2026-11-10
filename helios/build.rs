/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

fn main() {
    #[cfg(feature = "runtime-package")]
    {
        println!("cargo:rerun-if-changed=proto/runtime_package.proto");
        println!("cargo:rerun-if-changed=proto/package.proto");
        let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
        // The execution schema is the single definition of
        // contracts, graph selection, frontend settings, and engine placement.
        let mut runtime = prost_build::Config::new();
        runtime
            .type_attribute(".", "#[derive(serde::Serialize, serde::Deserialize)]")
            .file_descriptor_set_path(out.join("runtime_package_descriptor.bin"))
            .field_attribute(".flare.runtime.RebuildSpeculation.vocabulary_parallel_sampling", "#[serde(default)]")
            .field_attribute(".flare.runtime.RebuildSpeculation.mtp_only", "#[serde(default)]")
            .field_attribute(".flare.runtime.RebuildManifest.online_mxfp8_scope", "#[serde(default)]")
            .field_attribute(
                ".flare.runtime.ExecutionGraph.packed_token_capacity",
                "#[serde(default)]",
            )
            .compile_protos(&["proto/runtime_package.proto"], &["proto"])
            .expect("compile runtime package protobuf schema");
        // The payload schema imports the execution schema and refers to the generated types
        // instead of restating them.
        let mut package = prost_build::Config::new();
        package
            .type_attribute(".", "#[derive(serde::Serialize, serde::Deserialize)]")
            .extern_path(".flare.runtime", "crate::runtime_package::wire")
            .compile_protos(&["proto/package.proto"], &["proto"])
            .expect("compile payload package protobuf schema");
    }
    #[cfg(feature = "roce")]
    {
        println!("cargo:rerun-if-changed=native/roce_proxy.c");
        cc::Build::new()
            .file("native/roce_proxy.c")
            .std("gnu11")
            .compile("flare_roce_proxy");
        println!("cargo:rustc-link-lib=ibverbs");
        println!("cargo:rustc-link-lib=pthread");
    }
}
