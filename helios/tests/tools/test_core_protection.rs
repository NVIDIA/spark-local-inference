/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Read authenticated Python-produced fixtures through the native resource reader.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let path = args.get(1).ok_or("expected core path")?;
    let namespace = args.get(2).ok_or("expected accepted namespace")?;
    #[cfg(unix)]
    if namespace == "retarget-companion" {
        use flare_helios::runtime_core::{VerifiedCore, test_support};
        let path = std::path::Path::new(path);
        let core = VerifiedCore::open(path, &[test_support::publisher()], None)?;
        let companion = path.with_file_name(args.get(3).ok_or("expected companion name")?);
        std::fs::remove_file(&companion)?;
        std::os::unix::fs::symlink(args.get(4).ok_or("expected replacement target")?, companion)?;
        core.verify_stored_data()?;
        return Ok(());
    }
    if namespace == "ranges" {
        use flare_helios::runtime_core::{rank_ranges, test_support};
        let core = test_support::authorize(path);
        let rank = args.get(3).ok_or("expected rank")?.parse()?;
        let ranges = rank_ranges(&core, rank)?
            .into_iter()
            .map(|part| Ok((part.filename, (part.file.metadata()?.len(), part.ranges))))
            .collect::<Result<std::collections::BTreeMap<_, _>, std::io::Error>>()?;
        println!("{}", serde_json::to_string(&ranges)?);
        return Ok(());
    }
    if namespace == "records" {
        flare_helios::runtime_core::test_support::inspect_graph(
            std::path::Path::new(path),
            args.get(3).ok_or("expected graph name")?,
        )?;
        return Ok(());
    }
    // Optional: `rank=N` verifies only that rank of a multi-host core, then a profile path.
    let mut rest = args.iter().skip(4).peekable();
    let rank = match rest.peek().and_then(|arg| arg.strip_prefix("rank=")) {
        Some(rank) => {
            let rank = rank.parse()?;
            rest.next();
            Some(rank)
        }
        None => None,
    };
    let publishers = match rest.next() {
        Some(path) => serde_json::from_slice(&std::fs::read(path)?)?,
        None => vec![flare_helios::runtime_core::test_support::publisher()],
    };
    let contents = flare_helios::runtime_core::test_support::inspect(
        std::path::Path::new(path),
        namespace,
        args.get(3).is_some_and(|s| s == "verify-weights"),
        &publishers,
        rank,
    )?;
    println!("{}", serde_json::to_string(&contents)?);
    Ok(())
}
