//! Terminal launcher hardware discovery.
pub use flare::hardware::detect;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
/// Keep NVML's background threads out of the process that sets CUDA visibility.
pub fn detect_isolated(count: usize) -> Result<String> {
    let output = std::process::Command::new(std::env::current_exe()?)
        .arg("hardware-revision")
        .arg(count.to_string())
        .output()?;
    if !output.status.success() {
        return Err(format!(
            "hardware detection failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )
        .into());
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}
