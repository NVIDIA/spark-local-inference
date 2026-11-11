// SPDX-License-Identifier: Apache-2.0
use sha2::{Digest, Sha256};
use std::{env, fs, path::PathBuf, process::Command};

fn main() {
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let repository = manifest.parent().unwrap().parent().unwrap();
    let arch = match env::var("CARGO_CFG_TARGET_ARCH").unwrap().as_str() {
        "aarch64" => "ARM64",
        "x86_64" => "x64",
        _ => panic!("Flare launcher requires ARM64 or x64"),
    };
    let release = env::var("PROFILE").unwrap() == "release";
    if release
        && !env::var("CARGO_CFG_TARGET_FEATURE")
            .unwrap_or_default()
            .split(',')
            .any(|f| f == "crt-static")
    {
        panic!("Build the release launcher with RUSTFLAGS=-C target-feature=+crt-static");
    }
    let payload = repository
        .join(".local-dev/winui-bundle")
        .join(arch)
        .join("payload.zip");
    let output = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    println!("cargo:rerun-if-changed={}", payload.display());
    let bytes = match fs::read(&payload) {
        Ok(bytes) => bytes,
        Err(error) if !release && error.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(error) => panic!(
            "Create the native payload before building the launcher: {}: {error}",
            payload.display()
        ),
    };
    fs::write(output.join("payload.zip"), &bytes).unwrap();
    println!(
        "cargo:rustc-env=FLARE_PAYLOAD_SHA256={:x}",
        Sha256::digest(&bytes)
    );
    println!("cargo:rustc-env=FLARE_ARCH={arch}");
    let icon = manifest.parent().unwrap().join("Assets/flare-app.ico");
    let app_manifest = manifest.parent().unwrap().join("app.manifest");
    println!("cargo:rerun-if-changed={}", icon.display());
    println!("cargo:rerun-if-changed={}", app_manifest.display());
    let rc_path = output.join("launcher.rc");
    let quote = |path: &std::path::Path| path.to_string_lossy().replace('\\', "/");
    fs::write(
        &rc_path,
        format!(
            "1 ICON \"{}\"\n1 24 \"{}\"\nSTRINGTABLE\nBEGIN\n 103 \"Flare\"\nEND\n",
            quote(&icon),
            quote(&app_manifest)
        ),
    )
    .unwrap();
    let resource = output.join("launcher.res");
    let status = Command::new("rc.exe")
        .arg("/nologo")
        .arg("/fo")
        .arg(&resource)
        .arg(&rc_path)
        .status()
        .expect("Windows SDK rc.exe must be on PATH");
    assert!(status.success(), "Windows resources failed to compile");
    println!(
        "cargo:rustc-link-arg-bin=flare-launcher={}",
        resource.display()
    );
}
