/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Templates fetched and patched by tests/frontend/prepare_templates.py.

pub fn path(recipe: &str, name: &str) -> std::path::PathBuf {
    let directory = std::env::var_os("FLARE_TEST_TEMPLATE_DIR").expect(
        "run python tests/frontend/prepare_templates.py --output DIR, then set FLARE_TEST_TEMPLATE_DIR=DIR",
    );
    std::path::PathBuf::from(directory)
        .join(recipe)
        .join(format!("{name}.jinja"))
}

pub fn template(recipe: &str, name: &str) -> String {
    std::fs::read_to_string(path(recipe, name)).expect("prepared test template must exist")
}
