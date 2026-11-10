//! Terminal launcher process policy.
pub use flare::environment::{home_directory, visibility};
pub fn configure(devices: &[i32]) -> Result<(), Box<dyn std::error::Error>> {
    let inherited = match std::env::var("CUDA_VISIBLE_DEVICES") {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(error) => return Err(error.into()),
    };
    let selected = visibility(devices, inherited.as_deref())?;
    // SAFETY: Called once from main before driver initialization or worker startup.
    unsafe {
        std::env::set_var("CUDA_VISIBLE_DEVICES", selected);
        std::env::set_var("CUDA_FORCE_PRELOAD_LIBRARIES", "0");
        std::env::set_var("CUDA_DISABLE_JIT", "1");
    }
    Ok(())
}
