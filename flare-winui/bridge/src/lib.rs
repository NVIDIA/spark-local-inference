// SPDX-License-Identifier: Apache-2.0
//! Ownership and type conversions for the native Windows presentation layer.
use flare::controller::{Command, Controller, Snapshot};
use std::{
    cell::RefCell,
    ffi::{OsString, c_char},
    panic::{AssertUnwindSafe, catch_unwind},
    path::PathBuf,
    ptr, slice,
};

pub struct FlareApp {
    controller: Controller,
}
pub struct FlareSnapshot {
    view: Snapshot,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FlareString {
    data: *const c_char,
    length: usize,
}
fn text(value: &str) -> FlareString {
    FlareString {
        data: value.as_ptr().cast(),
        length: value.len(),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn model_metadata_preserves_optional_fields_and_utf8() {
        let mut view = Snapshot::default();
        view.models.push(flare::controller::ModelView {
            metadata: flare::controller::ModelMetadata {
                author: "模型作者".into(),
                hardware: "GB10".into(),
                context_tokens: Some(262144),
                revision_bytes: Some(25_528_087_320),
                rank_count: Some(1),
                ..Default::default()
            },
            ..Default::default()
        });
        view.models.push(flare::controller::ModelView::default());
        let held = Box::into_raw(Box::new(FlareSnapshot { view }));
        unsafe {
            let mut out = std::mem::MaybeUninit::<FlareModel>::uninit();
            assert_eq!(flare_snapshot_model(held, 0, out.as_mut_ptr()), 1);
            let found = out.assume_init();
            assert_eq!(found.metadata_known, 1 | 4 | 8);
            assert_eq!(found.context_tokens, 262144);
            assert_eq!(found.revision_bytes, 25_528_087_320);
            assert_eq!(found.rank_count, 1);
            assert_eq!(
                slice::from_raw_parts(found.author.data.cast::<u8>(), found.author.length),
                "模型作者".as_bytes()
            );
            let mut out = std::mem::MaybeUninit::<FlareModel>::uninit();
            assert_eq!(flare_snapshot_model(held, 1, out.as_mut_ptr()), 1);
            assert_eq!(out.assume_init().metadata_known, 0);
            assert_eq!(flare_snapshot_model(held, 2, ptr::null_mut()), 0);
            flare_snapshot_destroy(held);
        }
    }
    #[test]
    fn immutable_snapshot_outlives_closed_application() {
        let folder = tempfile::tempdir().unwrap();
        let config = folder.path().join("config.toml");
        let model = folder
            .path()
            .join("models")
            .display()
            .to_string()
            .replace('\\', "/");
        std::fs::write(&config, format!("[paths]\ncore_cache='{model}'\n")).unwrap();
        let config: Vec<u16> = config.to_str().unwrap().encode_utf16().collect();
        unsafe {
            let app = flare_app_create(config.as_ptr(), config.len());
            assert!(!app.is_null());
            assert_eq!(flare_app_clear_chat(app), 1);
            assert_eq!(flare_app_clear_chat(app), 1);
            let held = flare_app_snapshot(app);
            assert!(!held.is_null());
            flare_app_close(app);
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            while flare_app_is_closed(app) == 0 && std::time::Instant::now() < deadline {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            assert_eq!(flare_app_is_closed(app), 1);
            flare_app_destroy(app);
            let mut status = std::mem::MaybeUninit::<FlareStatus>::uninit();
            assert_eq!(flare_snapshot_status(held, status.as_mut_ptr()), 1);
            let status = status.assume_init();
            let path = slice::from_raw_parts(
                status.model_directory.data.cast::<u8>(),
                status.model_directory.length,
            );
            assert_eq!(
                PathBuf::from(std::str::from_utf8(path).unwrap()),
                PathBuf::from(&model)
            );
            flare_snapshot_destroy(held);
        }
    }
    #[test]
    fn invalid_inputs_return_errors_without_unwinding() {
        assert!(unsafe { flare_app_create(ptr::null(), 1) }.is_null());
        assert!(flare_last_error().length > 0);
        assert_eq!(
            unsafe { flare_snapshot_status(ptr::null(), ptr::null_mut()) },
            0
        );
        assert!(unsafe { utf8(ptr::null(), 0) }.unwrap().is_empty());
        assert!(unsafe { utf8([0xffu8].as_ptr().cast(), 1) }.is_err());
    }
    #[cfg(windows)]
    #[test]
    fn windows_paths_preserve_utf16_code_units() {
        use std::os::windows::ffi::OsStrExt;
        let units = [b'C' as u16, 58, 92, 0xd800, 0x6a21, 0x578b];
        let result = unsafe { path(units.as_ptr(), units.len()) }
            .unwrap()
            .unwrap();
        assert_eq!(result.as_os_str().encode_wide().collect::<Vec<_>>(), units);
    }
    #[test]
    fn chat_views_preserve_counted_utf8_and_bounds() {
        let mut view = Snapshot::default();
        view.preview.conversation_id = 42;
        view.preview.messages.push(flare::preview::ChatMessage {
            role: "assistant".into(),
            content: "Hello, 世界\0continued".into(),
            reasoning: "A thought".into(),
            error: "Stopped by user".into(),
            cancelled: true,
        });
        let held = Box::into_raw(Box::new(FlareSnapshot { view }));
        unsafe {
            assert_eq!(flare_snapshot_chat_message_count(held), 1);
            let mut message = std::mem::MaybeUninit::<FlareChatMessage>::uninit();
            assert_eq!(
                flare_snapshot_chat_message(held, 0, message.as_mut_ptr()),
                1
            );
            let message = message.assume_init();
            assert_eq!(
                slice::from_raw_parts(message.content.data.cast::<u8>(), message.content.length),
                "Hello, 世界\0continued".as_bytes()
            );
            assert_eq!(message.cancelled, 1);
            let mut preview = std::mem::MaybeUninit::<FlarePreview>::uninit();
            assert_eq!(flare_snapshot_preview(held, preview.as_mut_ptr()), 1);
            assert_eq!(preview.assume_init().conversation_id, 42);
            assert_eq!(flare_snapshot_chat_message(held, 1, ptr::null_mut()), 0);
            assert!(flare_last_error().length > 0);
            flare_snapshot_destroy(held);
        }
    }
}
macro_rules! no_argument_command {
    ($name:ident,$variant:expr) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(value: *mut FlareApp) -> u64 {
            boundary(0, || unsafe { app(value) }?.controller.submit($variant))
        }
    };
}
macro_rules! string_command {
    ($name:ident,$variant:path) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(
            value: *mut FlareApp,
            input: *const c_char,
            length: usize,
        ) -> u64 {
            boundary(0, || {
                unsafe { app(value) }?
                    .controller
                    .submit($variant(unsafe { utf8(input, length) }?))
            })
        }
    };
}
macro_rules! path_command {
    ($name:ident,$variant:path) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(
            value: *mut FlareApp,
            input: *const u16,
            length: usize,
        ) -> u64 {
            boundary(0, || {
                unsafe { app(value) }?.controller.submit($variant(
                    unsafe { path(input, length) }?.ok_or("Choose a path")?,
                ))
            })
        }
    };
}
no_argument_command!(flare_app_refresh, Command::Refresh);
no_argument_command!(flare_app_stop, Command::Stop);
string_command!(flare_app_discover, Command::Discover);
string_command!(flare_app_download, Command::Download);
string_command!(flare_app_activate, Command::Activate);
path_command!(flare_app_open_model, Command::OpenModel);
path_command!(flare_app_set_model_directory, Command::SetModelDirectory);
string_command!(flare_app_inspect_model, Command::InspectModel);
string_command!(flare_app_remove_reference, Command::RemoveReference);
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flare_app_locate_reference(
    value: *mut FlareApp,
    id: *const c_char,
    id_length: usize,
    input: *const u16,
    length: usize,
) -> u64 {
    boundary(0, || {
        unsafe { app(value) }?
            .controller
            .submit(Command::LocateReference {
                id: unsafe { utf8(id, id_length) }?,
                path: unsafe { path(input, length) }?.ok_or("Choose a core path")?,
            })
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flare_app_try_model(
    value: *mut FlareApp,
    prompt: *const c_char,
    length: usize,
) -> u64 {
    boundary(0, || {
        unsafe { app(value) }?
            .controller
            .try_model(unsafe { utf8(prompt, length) }?)
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flare_app_cancel_preview(value: *mut FlareApp) -> u32 {
    boundary(0, || {
        unsafe { app(value) }?
            .controller
            .cancel_preview()
            .map(|_| 1)
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flare_app_pause_download(value: *mut FlareApp, id: u64) -> u64 {
    boundary(0, || unsafe { app(value) }?.controller.pause_download(id))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flare_app_clear_chat(value: *mut FlareApp) -> u32 {
    boundary(0, || {
        unsafe { app(value) }?.controller.clear_chat().map(|_| 1)
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flare_app_set_appearance(
    value: *mut FlareApp,
    theme: u32,
    compact: u32,
) -> u64 {
    boundary(0, || {
        if compact > 1 {
            return Err("Invalid spacing preference".into());
        }
        unsafe { app(value) }?
            .controller
            .submit(Command::SetAppearance {
                theme,
                compact: compact != 0,
            })
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flare_app_cancel(value: *mut FlareApp, id: u64) -> u64 {
    boundary(0, || unsafe { app(value) }?.controller.cancel(id))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flare_app_resume(value: *mut FlareApp, id: u64) -> u64 {
    boundary(0, || {
        unsafe { app(value) }?
            .controller
            .submit(Command::Resume(id))
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flare_app_reject_trust(value: *mut FlareApp, id: u64) -> u64 {
    boundary(0, || {
        unsafe { app(value) }?
            .controller
            .submit(Command::RejectTrust(id))
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flare_app_accept_trust(
    value: *mut FlareApp,
    id: u64,
    publisher: *const c_char,
    length: usize,
) -> u64 {
    boundary(0, || {
        unsafe { app(value) }?
            .controller
            .submit(Command::AcceptTrust {
                decision: id,
                publisher: unsafe { utf8(publisher, length) }?,
            })
    })
}
thread_local! {static ERROR:RefCell<String>=const{RefCell::new(String::new())};}
fn failed(error: impl ToString) {
    ERROR.with(|slot| *slot.borrow_mut() = error.to_string());
}
fn boundary<T: Copy>(fallback: T, operation: impl FnOnce() -> Result<T, String>) -> T {
    match catch_unwind(AssertUnwindSafe(operation)) {
        Ok(Ok(value)) => value,
        Ok(Err(error)) => {
            failed(error);
            fallback
        }
        Err(_) => {
            failed("The Rust application boundary failed; restart Flare");
            fallback
        }
    }
}
unsafe fn app<'a>(value: *const FlareApp) -> Result<&'a FlareApp, String> {
    unsafe { value.as_ref() }.ok_or_else(|| "Missing application handle".into())
}
unsafe fn snapshot<'a>(value: *const FlareSnapshot) -> Result<&'a Snapshot, String> {
    unsafe { value.as_ref() }
        .map(|s| &s.view)
        .ok_or_else(|| "Missing snapshot handle".into())
}
unsafe fn utf8(value: *const c_char, length: usize) -> Result<String, String> {
    if length == 0 {
        return Ok(String::new());
    }
    if value.is_null() || length > 1024 * 1024 {
        return Err("Invalid text input".into());
    }
    std::str::from_utf8(unsafe { slice::from_raw_parts(value.cast(), length) })
        .map(str::to_owned)
        .map_err(|_| "Text input must be UTF-8".into())
}
unsafe fn path(value: *const u16, length: usize) -> Result<Option<PathBuf>, String> {
    if length == 0 {
        return Ok(None);
    }
    if value.is_null() || length > 32767 {
        return Err("Invalid Windows path".into());
    }
    let units = unsafe { slice::from_raw_parts(value, length) };
    if units.contains(&0) {
        return Err("A Windows path cannot contain a null character".into());
    }
    #[cfg(windows)]
    let native = {
        use std::os::windows::ffi::OsStringExt;
        OsString::from_wide(units)
    };
    #[cfg(not(windows))]
    let native = OsString::from(String::from_utf16(units).map_err(|_| "Invalid UTF-16 path")?);
    Ok(Some(PathBuf::from(native)))
}

/// # Safety
/// Call before WinUI initialization or other threads that can read the process environment.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flare_runtime_initialize() -> u32 {
    boundary(0, || {
        static INITIALIZED: std::sync::Once = std::sync::Once::new();
        INITIALIZED.call_once(|| unsafe {
            std::env::set_var("CUDA_FORCE_PRELOAD_LIBRARIES", "0");
            std::env::set_var("CUDA_DISABLE_JIT", "1");
        });
        Ok(1)
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flare_app_create(config: *const u16, length: usize) -> *mut FlareApp {
    boundary(ptr::null_mut(), || {
        let config = unsafe { path(config, length) }?;
        Ok(Box::into_raw(Box::new(FlareApp {
            controller: Controller::new(config)?,
        })))
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flare_app_close(value: *mut FlareApp) {
    boundary((), || {
        unsafe { app(value) }?.controller.close();
        Ok(())
    });
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flare_app_is_closed(value: *const FlareApp) -> u32 {
    boundary(0, || {
        Ok(u32::from(unsafe { app(value) }?.controller.is_closed()))
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flare_app_destroy(value: *mut FlareApp) {
    boundary((), || {
        if value.is_null() {
            return Ok(());
        }
        let handle = unsafe { app(value) }?;
        if !handle.controller.is_closed() {
            handle.controller.close();
            return Err("Wait for Flare to close before releasing its handle".into());
        }
        drop(unsafe { Box::from_raw(value) });
        Ok(())
    });
}
#[unsafe(no_mangle)]
pub extern "C" fn flare_last_error() -> FlareString {
    ERROR.with(|slot| text(&slot.borrow()))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flare_app_snapshot(value: *const FlareApp) -> *mut FlareSnapshot {
    boundary(ptr::null_mut(), || {
        Ok(Box::into_raw(Box::new(FlareSnapshot {
            view: unsafe { app(value) }?.controller.snapshot(),
        })))
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flare_snapshot_destroy(value: *mut FlareSnapshot) {
    boundary((), || {
        if !value.is_null() {
            drop(unsafe { Box::from_raw(value) });
        }
        Ok(())
    });
}

#[repr(C)]
pub struct FlareStatus {
    state: u32,
    has_metrics: u32,
    session_id: u64,
    active_model: FlareString,
    detail: FlareString,
    error: FlareString,
    endpoint: FlareString,
    model_directory: FlareString,
    stage: FlareString,
    stage_completed: u64,
    stage_total: u64,
    stage_total_known: u32,
    accepting: u32,
    active_requests: u64,
    waiting_requests: u64,
    ingress_requests: u64,
    generated_tokens_per_second: f64,
    prefill_tokens_per_second: f64,
    generated_tokens: u64,
    cached_pages: u64,
    total_pages: u64,
    busy: u32,
    catalog_loading: u32,
    current_operation: u64,
    detail_loading: u32,
    gpu_error: FlareString,
    appearance_theme: u32,
    appearance_compact: u32,
    hardware_sample_age_ms: u64,
    has_hardware_sample_age: u32,
    first_response_seconds: f64,
    first_response_samples: u64,
    first_response_window_seconds: u64,
    has_first_response: u32,
    first_response_truncated: u32,
}
#[repr(C)]
pub struct FlareModel {
    id: FlareString,
    name: FlareString,
    version: FlareString,
    source: FlareString,
    path: FlareString,
    description: FlareString,
    error: FlareString,
    logical_bytes: u64,
    size_known: u32,
    flags: u32,
    author: FlareString,
    task: FlareString,
    license: FlareString,
    quantization: FlareString,
    hardware: FlareString,
    revision: FlareString,
    revision_commit: FlareString,
    base_model: FlareString,
    context_tokens: u64,
    parameter_count: u64,
    revision_bytes: u64,
    rank_count: u32,
    metadata_known: u32,
}
#[repr(C)]
pub struct FlareDownload {
    operation_id: u64,
    state: u32,
    total_known: u32,
    name: FlareString,
    source: FlareString,
    phase: FlareString,
    error: FlareString,
    network_bytes: u64,
    network_total: u64,
    verified_bytes: u64,
    materialized_bytes: u64,
    bytes_per_second: f64,
    can_cancel: u32,
    can_resume: u32,
    installed_model: FlareString,
    transfer_active: u32,
    planning_completed: u64,
    planning_total: u64,
    planning_total_known: u32,
}
#[repr(C)]
pub struct FlareTrust {
    decision_id: u64,
    publisher: FlareString,
    fingerprint: FlareString,
    official_source: FlareString,
    license: FlareString,
    identity_status: FlareString,
}
unsafe fn output<T>(destination: *mut T, value: T) -> Result<u32, String> {
    if destination.is_null() {
        return Err("Missing output structure".into());
    }
    unsafe { destination.write(value) };
    Ok(1)
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flare_snapshot_model_count(value: *const FlareSnapshot) -> usize {
    boundary(0, || Ok(unsafe { snapshot(value) }?.models.len()))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flare_snapshot_model(
    value: *const FlareSnapshot,
    index: usize,
    out: *mut FlareModel,
) -> u32 {
    boundary(0, || {
        let s = unsafe { snapshot(value) }?
            .models
            .get(index)
            .ok_or("Model index is out of range")?;
        unsafe {
            output(
                out,
                FlareModel {
                    id: text(&s.id),
                    name: text(&s.name),
                    version: text(&s.version),
                    source: text(&s.source),
                    path: text(&s.path),
                    description: text(&s.description),
                    error: text(&s.error),
                    logical_bytes: s.logical_bytes.unwrap_or(0),
                    size_known: u32::from(s.logical_bytes.is_some()),
                    flags: s.flags,
                    author: text(&s.metadata.author),
                    task: text(&s.metadata.task),
                    license: text(&s.metadata.license),
                    quantization: text(&s.metadata.quantization),
                    hardware: text(&s.metadata.hardware),
                    revision: text(&s.metadata.revision),
                    revision_commit: text(&s.metadata.revision_commit),
                    base_model: text(&s.metadata.base_model),
                    context_tokens: s.metadata.context_tokens.unwrap_or(0),
                    parameter_count: s.metadata.parameter_count.unwrap_or(0),
                    revision_bytes: s.metadata.revision_bytes.unwrap_or(0),
                    rank_count: s.metadata.rank_count.unwrap_or(0),
                    metadata_known: u32::from(s.metadata.context_tokens.is_some())
                        | (u32::from(s.metadata.parameter_count.is_some()) << 1)
                        | (u32::from(s.metadata.revision_bytes.is_some()) << 2)
                        | (u32::from(s.metadata.rank_count.is_some()) << 3),
                },
            )
        }
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flare_snapshot_download_count(value: *const FlareSnapshot) -> usize {
    boundary(0, || Ok(unsafe { snapshot(value) }?.downloads.len()))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flare_snapshot_download(
    value: *const FlareSnapshot,
    index: usize,
    out: *mut FlareDownload,
) -> u32 {
    boundary(0, || {
        let s = unsafe { snapshot(value) }?
            .downloads
            .get(index)
            .ok_or("Download index is out of range")?;
        unsafe {
            output(
                out,
                FlareDownload {
                    operation_id: s.operation_id,
                    state: s.state,
                    total_known: u32::from(s.network_total.is_some()),
                    name: text(&s.name),
                    source: text(&s.source),
                    phase: text(&s.phase),
                    error: text(&s.error),
                    network_bytes: s.network_bytes,
                    network_total: s.network_total.unwrap_or(0),
                    verified_bytes: s.verified_bytes,
                    materialized_bytes: s.materialized_bytes,
                    bytes_per_second: s.bytes_per_second,
                    can_cancel: u32::from(s.can_cancel),
                    can_resume: u32::from(s.can_resume),
                    installed_model: text(&s.installed_model),
                    transfer_active: u32::from(s.transfer_active),
                    planning_completed: s.planning_completed,
                    planning_total: s.planning_total.unwrap_or(0),
                    planning_total_known: u32::from(s.planning_total.is_some()),
                },
            )
        }
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flare_snapshot_trust(
    value: *const FlareSnapshot,
    out: *mut FlareTrust,
) -> u32 {
    boundary(0, || {
        let Some(s) = unsafe { snapshot(value) }?.trust.as_ref() else {
            return Ok(0);
        };
        unsafe {
            output(
                out,
                FlareTrust {
                    decision_id: s.decision_id,
                    publisher: text(&s.publisher),
                    fingerprint: text(&s.fingerprint),
                    official_source: text(&s.official_source),
                    license: text(&s.license),
                    identity_status: text(&s.identity_status),
                },
            )
        }
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flare_snapshot_status(
    value: *const FlareSnapshot,
    out: *mut FlareStatus,
) -> u32 {
    boundary(0, || {
        let s = unsafe { snapshot(value) }?;
        unsafe {
            output(
                out,
                FlareStatus {
                    state: s.state as u32,
                    has_metrics: u32::from(s.has_metrics),
                    session_id: s.session_id,
                    active_model: text(&s.active_model),
                    detail: text(&s.detail),
                    error: text(&s.error),
                    endpoint: text(&s.endpoint),
                    model_directory: text(&s.model_directory),
                    stage: text(&s.stage),
                    stage_completed: s.stage_completed,
                    stage_total: s.stage_total.unwrap_or(0),
                    stage_total_known: u32::from(s.stage_total.is_some()),
                    accepting: u32::from(s.accepting),
                    active_requests: s.active_requests,
                    waiting_requests: s.waiting_requests,
                    ingress_requests: s.ingress_requests,
                    generated_tokens_per_second: s.generated_tokens_per_second,
                    prefill_tokens_per_second: s.prefill_tokens_per_second,
                    generated_tokens: s.generated_tokens,
                    cached_pages: s.cached_pages,
                    total_pages: s.total_pages,
                    busy: u32::from(s.busy),
                    catalog_loading: u32::from(s.catalog_loading),
                    current_operation: s.current_operation,
                    detail_loading: u32::from(s.detail_loading),
                    gpu_error: text(&s.gpu_error),
                    appearance_theme: s.appearance.theme,
                    appearance_compact: u32::from(s.appearance.compact),
                    hardware_sample_age_ms: s.hardware_sample_age_ms.unwrap_or(0),
                    has_hardware_sample_age: u32::from(s.hardware_sample_age_ms.is_some()),
                    first_response_seconds: s.first_response_seconds.unwrap_or(0.0),
                    first_response_samples: s.first_response_samples,
                    first_response_window_seconds: s.first_response_window_seconds,
                    has_first_response: u32::from(s.first_response_seconds.is_some()),
                    first_response_truncated: u32::from(s.first_response_truncated),
                },
            )
        }
    })
}
#[repr(C)]
pub struct FlareModelDetail {
    id: FlareString,
    name: FlareString,
    path: FlareString,
    publisher: FlareString,
    official_source: FlareString,
    source_revision: FlareString,
    metadata_status: FlareString,
    inspection_error: FlareString,
    architecture: FlareString,
    compatibility_reason: FlareString,
    memory_note: FlareString,
    context_tokens: u64,
    max_concurrency: u64,
    prepared_memory_bytes: u64,
    rank_count: u32,
    known_fields: u32,
    compatibility: u32,
    can_remove_reference: u32,
    signatures_verified: u32,
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flare_snapshot_model_detail(
    value: *const FlareSnapshot,
    out: *mut FlareModelDetail,
) -> u32 {
    boundary(0, || {
        let Some(s) = unsafe { snapshot(value) }?.model_detail.as_ref() else {
            return Ok(0);
        };
        let known = u32::from(s.context_tokens.is_some())
            | (u32::from(s.max_concurrency.is_some()) << 1)
            | (u32::from(s.prepared_memory_bytes.is_some()) << 2)
            | (u32::from(s.rank_count.is_some()) << 3);
        unsafe {
            output(
                out,
                FlareModelDetail {
                    id: text(&s.id),
                    name: text(&s.name),
                    path: text(&s.path),
                    publisher: text(&s.publisher),
                    official_source: text(&s.official_source),
                    source_revision: text(&s.source_revision),
                    metadata_status: text(&s.metadata_status),
                    inspection_error: text(&s.inspection_error),
                    architecture: text(&s.architecture),
                    compatibility_reason: text(&s.compatibility_reason),
                    memory_note: text(&s.memory_note),
                    context_tokens: s.context_tokens.unwrap_or(0),
                    max_concurrency: s.max_concurrency.unwrap_or(0),
                    prepared_memory_bytes: s.prepared_memory_bytes.unwrap_or(0),
                    rank_count: s.rank_count.unwrap_or(0),
                    known_fields: known,
                    compatibility: s.compatibility as u32,
                    can_remove_reference: u32::from(s.can_remove_reference),
                    signatures_verified: u32::from(s.signatures_verified),
                },
            )
        }
    })
}
#[repr(C)]
pub struct FlareGpu {
    ordinal: u32,
    visible_ordinal: u32,
    has_visible_ordinal: u32,
    name: FlareString,
    uuid: FlareString,
    platform: FlareString,
    total_memory_bytes: u64,
    free_memory_bytes: u64,
    has_total_memory: u32,
    has_free_memory: u32,
    has_architecture: u32,
    architecture_major: u32,
    architecture_minor: u32,
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flare_snapshot_gpu_count(value: *const FlareSnapshot) -> usize {
    boundary(0, || Ok(unsafe { snapshot(value) }?.gpus.len()))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flare_snapshot_gpu(
    value: *const FlareSnapshot,
    index: usize,
    out: *mut FlareGpu,
) -> u32 {
    boundary(0, || {
        let s = unsafe { snapshot(value) }?
            .gpus
            .get(index)
            .ok_or("GPU index is out of range")?;
        let architecture = s.architecture.unwrap_or([0, 0]);
        unsafe {
            output(
                out,
                FlareGpu {
                    ordinal: s.ordinal,
                    visible_ordinal: s.visible_ordinal.unwrap_or(0),
                    has_visible_ordinal: u32::from(s.visible_ordinal.is_some()),
                    name: text(&s.name),
                    uuid: text(&s.uuid),
                    platform: text(s.platform.as_deref().unwrap_or_default()),
                    total_memory_bytes: s.total_memory_bytes.unwrap_or(0),
                    free_memory_bytes: s.free_memory_bytes.unwrap_or(0),
                    has_total_memory: u32::from(s.total_memory_bytes.is_some()),
                    has_free_memory: u32::from(s.free_memory_bytes.is_some()),
                    has_architecture: u32::from(s.architecture.is_some()),
                    architecture_major: architecture[0],
                    architecture_minor: architecture[1],
                },
            )
        }
    })
}
#[repr(C)]
pub struct FlarePreview {
    operation_id: u64,
    session_id: u64,
    busy: u32,
    cancelled: u32,
    request_id: FlareString,
    text: FlareString,
    reasoning: FlareString,
    error: FlareString,
    conversation_id: u64,
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flare_snapshot_preview(
    value: *const FlareSnapshot,
    out: *mut FlarePreview,
) -> u32 {
    boundary(0, || {
        let s = &unsafe { snapshot(value) }?.preview;
        unsafe {
            output(
                out,
                FlarePreview {
                    operation_id: s.operation_id,
                    session_id: s.session_id,
                    busy: u32::from(s.busy),
                    cancelled: u32::from(s.cancelled),
                    request_id: text(&s.request_id),
                    text: text(&s.output),
                    reasoning: text(&s.reasoning),
                    error: text(&s.error),
                    conversation_id: s.conversation_id,
                },
            )
        }
    })
}

#[repr(C)]
pub struct FlareChatMessage {
    role: FlareString,
    content: FlareString,
    reasoning: FlareString,
    error: FlareString,
    cancelled: u32,
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flare_snapshot_chat_message_count(value: *const FlareSnapshot) -> usize {
    boundary(0, || Ok(unsafe { snapshot(value) }?.preview.messages.len()))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flare_snapshot_chat_message(
    value: *const FlareSnapshot,
    index: usize,
    out: *mut FlareChatMessage,
) -> u32 {
    boundary(0, || {
        let message = unsafe { snapshot(value) }?
            .preview
            .messages
            .get(index)
            .ok_or("Chat message index is out of range")?;
        unsafe {
            output(
                out,
                FlareChatMessage {
                    role: text(&message.role),
                    content: text(&message.content),
                    reasoning: text(&message.reasoning),
                    error: text(&message.error),
                    cancelled: u32::from(message.cancelled),
                },
            )
        }
    })
}

#[repr(C)]
pub struct FlareDashboard {
    available: u32,
    known_fields: u32,
    session_id: u64,
    uptime_seconds: f64,
    prefill_tokens_per_second: f64,
    generated_tokens_per_second: f64,
    verify_steps_per_second: f64,
    acceptance: f64,
    depth: f64,
    weight_bytes_per_second: f64,
    prefill_tokens: u64,
    generated_tokens: u64,
    reused_tokens: u64,
    active: u64,
    pending: u64,
    completed: u64,
    failed: u64,
    ingress: u64,
    cached_pages: u64,
    free_pages: u64,
    total_pages: u64,
    page_tokens: u64,
    host_hits: u64,
    host_reused_tokens: u64,
    host_memory_records: u64,
    host_disk_records: u64,
    host_bytes_out: u64,
    host_bytes_in: u64,
}
#[repr(C)]
pub struct FlareDashboardPoint {
    elapsed_seconds: f64,
    prefill_tokens_per_second: f64,
    generated_tokens_per_second: f64,
    verify_steps_per_second: f64,
    acceptance: f64,
    depth: f64,
    known_fields: u32,
}
impl From<&flare::controller::DashboardView> for FlareDashboard {
    fn from(s: &flare::controller::DashboardView) -> Self {
        Self {
            available: u32::from(s.available),
            known_fields: u32::from(s.acceptance.is_some())
                | (u32::from(s.depth.is_some()) << 1)
                | (u32::from(s.weight_bytes_per_second.is_some()) << 2),
            session_id: s.session_id,
            uptime_seconds: s.uptime_seconds,
            prefill_tokens_per_second: s.prefill_tokens_per_second,
            generated_tokens_per_second: s.generated_tokens_per_second,
            verify_steps_per_second: s.verify_steps_per_second,
            acceptance: s.acceptance.unwrap_or(0.0),
            depth: s.depth.unwrap_or(0.0),
            weight_bytes_per_second: s.weight_bytes_per_second.unwrap_or(0.0),
            prefill_tokens: s.prefill_tokens,
            generated_tokens: s.generated_tokens,
            reused_tokens: s.reused_tokens,
            active: s.active,
            pending: s.pending,
            completed: s.completed,
            failed: s.failed,
            ingress: s.ingress,
            cached_pages: s.cached_pages,
            free_pages: s.free_pages,
            total_pages: s.total_pages,
            page_tokens: s.page_tokens,
            host_hits: s.host_hits,
            host_reused_tokens: s.host_reused_tokens,
            host_memory_records: s.host_memory_records,
            host_disk_records: s.host_disk_records,
            host_bytes_out: s.host_bytes_out,
            host_bytes_in: s.host_bytes_in,
        }
    }
}
impl From<&flare::controller::DashboardPoint> for FlareDashboardPoint {
    fn from(s: &flare::controller::DashboardPoint) -> Self {
        Self {
            elapsed_seconds: s.elapsed_seconds,
            prefill_tokens_per_second: s.prefill_tokens_per_second,
            generated_tokens_per_second: s.generated_tokens_per_second,
            verify_steps_per_second: s.verify_steps_per_second,
            acceptance: s.acceptance.unwrap_or(0.0),
            depth: s.depth.unwrap_or(0.0),
            known_fields: u32::from(s.acceptance.is_some()) | (u32::from(s.depth.is_some()) << 1),
        }
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flare_snapshot_dashboard(
    value: *const FlareSnapshot,
    out: *mut FlareDashboard,
) -> u32 {
    boundary(0, || unsafe {
        output(out, FlareDashboard::from(&snapshot(value)?.dashboard))
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flare_snapshot_dashboard_point_count(
    value: *const FlareSnapshot,
) -> usize {
    boundary(0, || Ok(unsafe { snapshot(value) }?.dashboard.points.len()))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flare_snapshot_dashboard_point(
    value: *const FlareSnapshot,
    index: usize,
    out: *mut FlareDashboardPoint,
) -> u32 {
    boundary(0, || {
        let point = unsafe { snapshot(value) }?
            .dashboard
            .points
            .get(index)
            .ok_or("Dashboard point index is out of range")?;
        unsafe { output(out, FlareDashboardPoint::from(point)) }
    })
}

#[cfg(test)]
mod dashboard_tests {
    use super::*;
    #[test]
    fn dashboard_abi_preserves_units_options_totals_and_point_bounds() {
        let mut view = Snapshot::default();
        view.dashboard = flare::controller::DashboardView {
            available: true,
            session_id: 44,
            uptime_seconds: 12.5,
            prefill_tokens_per_second: 100.0,
            generated_tokens_per_second: 20.0,
            verify_steps_per_second: 4.0,
            acceptance: Some(3.0),
            depth: Some(2.5),
            weight_bytes_per_second: Some(4_000_000_000.0),
            prefill_tokens: 101,
            generated_tokens: 102,
            reused_tokens: 103,
            active: 2,
            pending: 3,
            completed: 4,
            failed: 5,
            ingress: 6,
            cached_pages: 7,
            free_pages: 8,
            total_pages: 20,
            page_tokens: 32,
            host_hits: 11,
            host_reused_tokens: 12,
            host_memory_records: 13,
            host_disk_records: 14,
            host_bytes_out: 15,
            host_bytes_in: 16,
            points: vec![
                flare::controller::DashboardPoint {
                    elapsed_seconds: 12.5,
                    prefill_tokens_per_second: 100.0,
                    generated_tokens_per_second: 20.0,
                    verify_steps_per_second: 4.0,
                    acceptance: Some(0.0),
                    depth: None,
                },
                flare::controller::DashboardPoint::default(),
            ],
        };
        let held = Box::into_raw(Box::new(FlareSnapshot { view }));
        unsafe {
            let mut output = std::mem::MaybeUninit::<FlareDashboard>::uninit();
            assert_eq!(flare_snapshot_dashboard(held, output.as_mut_ptr()), 1);
            let found = output.assume_init();
            assert_eq!(
                (found.available, found.known_fields, found.session_id),
                (1, 7, 44)
            );
            assert_eq!(
                (
                    found.uptime_seconds,
                    found.prefill_tokens_per_second,
                    found.generated_tokens_per_second,
                    found.verify_steps_per_second
                ),
                (12.5, 100.0, 20.0, 4.0)
            );
            assert_eq!(
                (found.acceptance, found.depth, found.weight_bytes_per_second),
                (3.0, 2.5, 4_000_000_000.0)
            );
            assert_eq!(
                (
                    found.prefill_tokens,
                    found.generated_tokens,
                    found.reused_tokens
                ),
                (101, 102, 103)
            );
            assert_eq!(
                (
                    found.active,
                    found.pending,
                    found.completed,
                    found.failed,
                    found.ingress
                ),
                (2, 3, 4, 5, 6)
            );
            assert_eq!(
                (
                    found.cached_pages,
                    found.free_pages,
                    found.total_pages,
                    found.page_tokens
                ),
                (7, 8, 20, 32)
            );
            assert_eq!(
                (
                    found.host_hits,
                    found.host_reused_tokens,
                    found.host_memory_records,
                    found.host_disk_records,
                    found.host_bytes_out,
                    found.host_bytes_in
                ),
                (11, 12, 13, 14, 15, 16)
            );
            assert_eq!(flare_snapshot_dashboard_point_count(held), 2);
            let mut point = std::mem::MaybeUninit::<FlareDashboardPoint>::uninit();
            assert_eq!(
                flare_snapshot_dashboard_point(held, 0, point.as_mut_ptr()),
                1
            );
            let point = point.assume_init();
            assert_eq!(
                (
                    point.elapsed_seconds,
                    point.prefill_tokens_per_second,
                    point.generated_tokens_per_second,
                    point.verify_steps_per_second
                ),
                (12.5, 100.0, 20.0, 4.0)
            );
            assert_eq!(
                (point.known_fields, point.acceptance, point.depth),
                (1, 0.0, 0.0)
            );
            let mut unknown = std::mem::MaybeUninit::<FlareDashboardPoint>::uninit();
            assert_eq!(
                flare_snapshot_dashboard_point(held, 1, unknown.as_mut_ptr()),
                1
            );
            assert_eq!(unknown.assume_init().known_fields, 0);
            assert_eq!(flare_snapshot_dashboard_point(held, 2, ptr::null_mut()), 0);
            assert_eq!(flare_snapshot_dashboard(held, ptr::null_mut()), 0);
            flare_snapshot_destroy(held);
        }
        assert_eq!(std::mem::size_of::<FlareDashboard>(), 216);
        assert_eq!(std::mem::size_of::<FlareDashboardPoint>(), 56);
    }
    #[test]
    fn dashboard_unknowns_have_no_availability_flags() {
        let found = FlareDashboard::from(&flare::controller::DashboardView::default());
        assert_eq!((found.available, found.known_fields), (0, 0));
        let known_zero = FlareDashboard::from(&flare::controller::DashboardView {
            available: true,
            acceptance: Some(0.0),
            depth: Some(0.0),
            weight_bytes_per_second: Some(0.0),
            ..Default::default()
        });
        assert_eq!((known_zero.available, known_zero.known_fields), (1, 7));
        unsafe {
            assert_eq!(flare_snapshot_dashboard(ptr::null(), ptr::null_mut()), 0);
            assert_eq!(flare_snapshot_dashboard_point_count(ptr::null()), 0);
        }
    }
}

#[repr(C)]
pub struct FlareRuntimeSettings {
    memory_limit: FlareString,
    kv_cache_limit: FlareString,
    host: FlareString,
    prefix_cache_memory: FlareString,
    prefix_cache_disk: FlareString,
    prefix_cache_disk_limit: FlareString,
    embedding_table_storage: FlareString,
    embedding_table_cache_size: FlareString,
    prefix_cache_max_age: FlareString,
    prefix_cache_drain_on_shutdown: FlareString,
    max_concurrency: u64,
    mtp_draft_tokens: u64,
    port: u32,
    verify_weights: u32,
    has_max_concurrency: u32,
    has_mtp_draft_tokens: u32,
    history_drafts: u32,
    adaptive_verification: u32,
    pending_reload: u32,
    has_api_key: u32,
    prefix_cache_disk_enabled: u32,
    last_settings_operation: u64,
    settings_error: FlareString,
    active_bind: FlareString,
}
fn optional_bool(value: u32) -> Result<Option<bool>, String> {
    match value {
        0 => Ok(None),
        1 => Ok(Some(false)),
        2 => Ok(Some(true)),
        _ => Err("Optional setting must be automatic, off, or on".into()),
    }
}
fn bool_setting(value: u32) -> Result<bool, String> {
    match value {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err("Boolean setting must be zero or one".into()),
    }
}
fn bool_code(value: Option<bool>) -> u32 {
    match value {
        None => 0,
        Some(false) => 1,
        Some(true) => 2,
    }
}
unsafe fn runtime_input(
    value: *const FlareRuntimeSettings,
) -> Result<flare::settings::RuntimeSettings, String> {
    let s = unsafe { value.as_ref() }.ok_or("Missing runtime settings")?;
    let copy = |value: FlareString| unsafe { utf8(value.data, value.length) };
    let optional = |known: u32, value: u64| -> Result<Option<usize>, String> {
        if bool_setting(known)? {
            Ok(Some(
                usize::try_from(value).map_err(|_| "Runtime number is too large")?,
            ))
        } else {
            Ok(None)
        }
    };
    Ok(flare::settings::RuntimeSettings {
        memory_limit: copy(s.memory_limit)?,
        kv_cache_limit: copy(s.kv_cache_limit)?,
        host: copy(s.host)?,
        port: u16::try_from(s.port).map_err(|_| "Listen port must be 1 through 65535")?,
        max_concurrency: optional(s.has_max_concurrency, s.max_concurrency)?,
        verify_weights: bool_setting(s.verify_weights)?,
        prefix_cache_memory: copy(s.prefix_cache_memory)?,
        prefix_cache_disk: copy(s.prefix_cache_disk)?,
        prefix_cache_disk_limit: copy(s.prefix_cache_disk_limit)?,
        prefix_cache_disk_enabled: bool_setting(s.prefix_cache_disk_enabled)?,
        prefix_cache_max_age: copy(s.prefix_cache_max_age)?,
        prefix_cache_drain_on_shutdown: copy(s.prefix_cache_drain_on_shutdown)?,
        embedding_table_storage: copy(s.embedding_table_storage)?,
        embedding_table_cache_size: copy(s.embedding_table_cache_size)?,
        mtp_draft_tokens: optional(s.has_mtp_draft_tokens, s.mtp_draft_tokens)?,
        history_drafts: optional_bool(s.history_drafts)?,
        adaptive_verification: optional_bool(s.adaptive_verification)?,
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flare_app_set_runtime_settings(
    app_value: *mut FlareApp,
    value: *const FlareRuntimeSettings,
) -> u64 {
    boundary(0, || {
        unsafe { app(app_value) }?
            .controller
            .submit(Command::SetRuntimeSettings(unsafe {
                runtime_input(value)
            }?))
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flare_snapshot_runtime_settings(
    value: *const FlareSnapshot,
    out: *mut FlareRuntimeSettings,
) -> u32 {
    boundary(0, || {
        let view = unsafe { snapshot(value) }?;
        let s = &view.runtime_settings;
        unsafe {
            output(
                out,
                FlareRuntimeSettings {
                    memory_limit: text(&s.memory_limit),
                    kv_cache_limit: text(&s.kv_cache_limit),
                    host: text(&s.host),
                    prefix_cache_memory: text(&s.prefix_cache_memory),
                    prefix_cache_disk: text(&s.prefix_cache_disk),
                    prefix_cache_disk_limit: text(&s.prefix_cache_disk_limit),
                    embedding_table_storage: text(&s.embedding_table_storage),
                    embedding_table_cache_size: text(&s.embedding_table_cache_size),
                    prefix_cache_max_age: text(&s.prefix_cache_max_age),
                    prefix_cache_drain_on_shutdown: text(&s.prefix_cache_drain_on_shutdown),
                    max_concurrency: s.max_concurrency.unwrap_or(0) as u64,
                    mtp_draft_tokens: s.mtp_draft_tokens.unwrap_or(0) as u64,
                    port: u32::from(s.port),
                    verify_weights: u32::from(s.verify_weights),
                    has_max_concurrency: u32::from(s.max_concurrency.is_some()),
                    has_mtp_draft_tokens: u32::from(s.mtp_draft_tokens.is_some()),
                    history_drafts: bool_code(s.history_drafts),
                    adaptive_verification: bool_code(s.adaptive_verification),
                    pending_reload: u32::from(view.runtime_settings_pending_reload),
                    has_api_key: u32::from(view.has_api_key),
                    prefix_cache_disk_enabled: u32::from(s.prefix_cache_disk_enabled),
                    last_settings_operation: view.last_settings_operation,
                    settings_error: text(&view.settings_error),
                    active_bind: text(&view.bind_address),
                },
            )
        }
    })
}

#[cfg(test)]
mod runtime_settings_abi_tests {
    use super::*;
    #[test]
    fn runtime_settings_abi_copies_options_and_borrows_only_snapshot_output() {
        let mut view=Snapshot::default();
        view.runtime_settings.max_concurrency=Some(3);view.runtime_settings.mtp_draft_tokens=Some(0);
        view.runtime_settings.history_drafts=Some(false);view.runtime_settings.adaptive_verification=Some(true);
        view.runtime_settings.prefix_cache_disk="C:/模型/prefix-cache".into();
        view.runtime_settings_pending_reload=true;view.has_api_key=true;view.last_settings_operation=42;
        view.settings_error="Save failed".into();view.bind_address="0.0.0.0:8000".into();
        let expected=view.runtime_settings.clone();let held=Box::into_raw(Box::new(FlareSnapshot{view}));
        unsafe {
            let mut out=std::mem::MaybeUninit::<FlareRuntimeSettings>::uninit();
            assert_eq!(flare_snapshot_runtime_settings(held,out.as_mut_ptr()),1);
            let mut found=out.assume_init();
            assert_eq!((found.has_max_concurrency,found.max_concurrency,found.has_mtp_draft_tokens,found.mtp_draft_tokens),(1,3,1,0));
            assert_eq!((found.history_drafts,found.adaptive_verification,found.pending_reload,found.has_api_key,found.prefix_cache_disk_enabled),(1,2,1,1,0));
            assert_eq!(found.last_settings_operation,42);
            assert_eq!(utf8(found.settings_error.data,found.settings_error.length).unwrap(),"Save failed");
            let copied=runtime_input(&found).unwrap();assert_eq!(copied,expected);
            found.history_drafts=9;assert!(runtime_input(&found).is_err());
            found.history_drafts=0;found.port=65536;assert!(runtime_input(&found).is_err());
            flare_snapshot_destroy(held);
            assert_eq!(copied.prefix_cache_disk,"C:/模型/prefix-cache");
            assert_eq!(flare_snapshot_runtime_settings(ptr::null(),ptr::null_mut()),0);
            assert!(runtime_input(ptr::null()).is_err());
        }
    }
    #[test]
    fn tri_state_boolean_encoding_preserves_automatic_and_explicit_zero() {
        for value in [None,Some(false),Some(true)] {assert_eq!(optional_bool(bool_code(value)).unwrap(),value);}
        assert!(bool_setting(2).is_err());assert!(optional_bool(3).is_err());
    }
}
