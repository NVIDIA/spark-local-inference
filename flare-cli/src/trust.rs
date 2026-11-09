//! Terminal publisher disclosure and consent.
use flare::trust::{Acceptance, Identity, ReviewOptions};
#[cfg(unix)]
pub use flare::trust::{publishers, trusted_profiles};
use flare_helios::runtime_core::{AuthorizedCore, Publisher, VerifiedCore};
use std::{
    io::{self, BufRead, IsTerminal, Write},
    path::Path,
};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub fn load(
    root: &Path,
    args: &crate::cli::Run,
    progress: &mut flare_helios::startup::StartupProgress,
    origin: Option<&str>,
) -> Result<(AuthorizedCore, Publisher)> {
    load_with(root, args, progress, origin, true)
}
pub fn load_metadata(
    root: &Path,
    args: &crate::cli::Run,
    progress: &mut flare_helios::startup::StartupProgress,
    origin: Option<&str>,
) -> Result<(AuthorizedCore, Publisher)> {
    load_with(root, args, progress, origin, false)
}
fn load_with(
    root: &Path,
    args: &crate::cli::Run,
    progress: &mut flare_helios::startup::StartupProgress,
    origin: Option<&str>,
    verify_payloads: bool,
) -> Result<(AuthorizedCore, Publisher)> {
    if args.hub.tensor_parallel.len() > 1 {
        return Err(
            "select one tensor-parallel size for a run; the core holds one deployment per size"
                .into(),
        );
    }
    let tensor_parallel =
        args.hub.tensor_parallel.first().copied().or_else(|| {
            (args.rank.is_none() && !args.gpus.is_empty()).then(|| args.gpus.len() as u32)
        });
    let phase = progress.phase("Verify core", None);
    let review = flare::trust::review(
        root,
        &ReviewOptions {
            publisher_keys: args.publisher_keys.clone(),
            hub: (&args.hub).into(),
            trust_store: args.trust_store.clone(),
            rank: args.rank.filter(|&rank| rank != 0),
            tensor_parallel,
        },
    )?;
    phase.finish("signatures");
    let disclosure = &review.disclosure;
    if let Some(origin) = origin {
        let downloaded = format!("https://huggingface.co/{origin}");
        if downloaded != disclosure.official_source {
            progress.finish_animation();
            eprintln!(
                "\nWarning: downloaded from {downloaded}; the publisher signed {} as its source.",
                disclosure.official_source
            );
        }
    }
    let explicit = match (&args.dangerously_approve_key, &args.dangerously_accept_publisher) {
        (Some(key), Some(owner)) if key == &disclosure.fingerprint && owner == &disclosure.namespace => Some(owner.clone()),
        (None, None) => None,
        _ => return Err("--dangerously-approve-key and --dangerously-accept-publisher must exactly match the authenticated key and namespace".into()),
    };
    if !disclosure.remembered {
        progress.finish_animation();
        eprintln!(
            "\n{}",
            signing_info(
                review.core(),
                &disclosure.fingerprint,
                &disclosure.identity,
                args.verify_weights
            )
        );
    }
    let accepted = if disclosure.remembered {
        disclosure.namespace.clone()
    } else if let Some(owner) = explicit {
        owner
    } else {
        if !io::stdin().is_terminal() || !io::stderr().is_terminal() {
            return Err("unattended first use requires --dangerously-approve-key FINGERPRINT --dangerously-accept-publisher NAMESPACE".into());
        }
        let namespace = &disclosure.namespace;
        match &disclosure.license_name {
            Some(license) => eprint!(
                "\nType \x1b[1m{namespace}\x1b[0m to agree to the {license} terms and trust this signing key: "
            ),
            None => eprint!("\nType \x1b[1m{namespace}\x1b[0m to trust this signing key: "),
        }
        io::stderr().flush()?;
        let response = acceptance_line(io::stdin().lock())?;
        if response != *namespace {
            return Err("publisher acceptance declined or did not match".into());
        }
        response
    };
    let first_use = !disclosure.remembered;
    let acceptance = Acceptance {
        fingerprint: disclosure.fingerprint.clone(),
        namespace: accepted,
    };
    let result = review.accept(
        &acceptance,
        args.verify_weights,
        verify_payloads,
        Some(progress),
    )?;
    if first_use {
        eprintln!("Signing key accepted for {}.\n", acceptance.namespace);
    }
    Ok(result)
}
pub fn revoke(fingerprint: &str, path: Option<&Path>) -> Result<()> {
    flare::trust::revoke(fingerprint, path)?;
    eprintln!("Removed trust for signer {fingerprint}.");
    Ok(())
}
fn signing_info(
    core: &VerifiedCore,
    fingerprint: &str,
    identity: &Identity,
    verify_weights: bool,
) -> String {
    let publisher = core.publisher();
    let mut lines = vec![
        format!("Publisher:  {} ({})", publisher.name, publisher.namespace),
        format!(
            "Identity:   {}",
            match identity {
                Identity::Verified =>
                    format!("Verified via huggingface.co/{}", publisher.namespace),
                Identity::Unverified(reason) => format!("Unverified; {reason}"),
                Identity::Development => "Local development key; verified by no publisher".into(),
            }
        ),
        "Signature:  Valid".into(),
        format!("Source:     {}", core.official_source()),
    ];
    if let Some(license) = core.license() {
        lines.push(format!("License:    {}", license.name));
        if !license.link.is_empty() {
            lines.push(format!("            {}", license.link));
        }
    }
    lines.extend([
        format!(
            "Stored data: {}",
            if verify_weights {
                "Verification enabled"
            } else {
                "Not checked (--verify-weights to check)"
            }
        ),
        String::new(),
        "Signing key:".into(),
        fingerprint.to_owned(),
        String::new(),
        "Executable GPU code can access your process and data.".into(),
    ]);
    if matches!(identity, Identity::Unverified(_)) {
        lines.push("A valid signature does not verify the claimed name or source.".into());
    }
    if publisher.development {
        lines.push("Local development core; not an official published checkpoint.".into());
    } else {
        lines.push("Do not proceed if this core came from another source.".into());
    }
    if core.license().is_some() {
        lines.push(
            "Typing the publisher namespace agrees to the license terms at the link above.".into(),
        );
    }
    lines.push("Only approve a signing key you independently trust.".into());
    if !io::stderr().is_terminal() {
        return format!("Signing key acceptance\n{}", lines.join("\n"));
    }
    #[cfg(windows)]
    let columns = crate::windows::terminal_columns();
    #[cfg(unix)]
    let columns = {
        let mut size: libc::winsize = unsafe { std::mem::zeroed() };
        // SAFETY: ioctl writes terminal dimensions into a valid winsize.
        unsafe { libc::ioctl(libc::STDERR_FILENO, libc::TIOCGWINSZ, &mut size) };
        if size.ws_col == 0 {
            80
        } else {
            usize::from(size.ws_col)
        }
    };
    if columns < 28 {
        return format!("Signing key acceptance\n{}", lines.join("\n"));
    }
    let width = columns.min(80) - 4;
    // Only a publisher-confirmed identity earns the calm frame; development and unknown keys warn.
    let color = if matches!(identity, Identity::Verified) {
        "36"
    } else {
        "33"
    };
    let border = format!("\x1b[{color}m");
    let title = " Signing key acceptance ";
    let mut output = format!(
        "{border}╭─{title}{}╮\x1b[0m\n",
        "─".repeat(width + 1 - title.len())
    );
    for line in lines {
        let mut remaining = line.as_str();
        while remaining.chars().count() > width {
            let end = remaining.char_indices().nth(width).unwrap().0;
            let split = remaining[..end]
                .rfind(' ')
                .filter(|n| *n > width / 2)
                .unwrap_or(end);
            let row = remaining[..split].trim_end();
            output.push_str(&format!(
                "{border}│\x1b[0m {row}{} {border}│\x1b[0m\n",
                " ".repeat(width - row.chars().count())
            ));
            remaining = remaining[split..].trim_start();
        }
        output.push_str(&format!(
            "{border}│\x1b[0m {remaining}{} {border}│\x1b[0m\n",
            " ".repeat(width - remaining.chars().count())
        ));
    }
    output.push_str(&format!("{border}╰{}╯\x1b[0m", "─".repeat(width + 2)));
    output
}

fn acceptance_line(input: impl BufRead) -> io::Result<String> {
    let mut response = String::new();
    input.take(256).read_line(&mut response)?;
    if !response.ends_with('\n') {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "publisher acceptance requires a complete line",
        ));
    }
    response.pop();
    if response.ends_with('\r') {
        response.pop();
    }
    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn acceptance_is_exact_and_requires_a_complete_bounded_line() {
        assert_eq!(
            acceptance_line(&b"example-publisher\n"[..]).unwrap(),
            "example-publisher"
        );
        assert_eq!(
            acceptance_line(&b"example-publisher\r\n"[..]).unwrap(),
            "example-publisher"
        );
        assert_eq!(
            acceptance_line(&b" example-publisher \n"[..]).unwrap(),
            " example-publisher "
        );
        assert!(acceptance_line(&b"example-publisher"[..]).is_err());
        assert!(acceptance_line([vec![b'x'; 256], vec![b'\n']].concat().as_slice()).is_err());
        assert!(acceptance_line(&b""[..]).is_err());
    }
}
