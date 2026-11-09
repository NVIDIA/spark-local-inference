// SPDX-License-Identifier: Apache-2.0
//! External references are metadata operations and never delete the selected files.
use flare::{
    controller::{Command, Controller, Snapshot},
    details::Compatibility,
};
use std::{
    path::Path,
    time::{Duration, Instant},
};
fn wait(controller: &Controller, ready: impl Fn(&Snapshot) -> bool) -> Snapshot {
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        let view = controller.snapshot();
        if ready(&view) {
            return view;
        }
        assert!(Instant::now() < deadline, "Timed out: {view:?}");
        std::thread::sleep(Duration::from_millis(20));
    }
}
fn same_path(value: &str, path: &Path) -> bool {
    Path::new(value).canonicalize().ok() == path.canonicalize().ok()
}
#[test]
fn inspect_locate_and_remove_external_references_preserve_the_core_files() {
    let temporary = tempfile::tempdir().unwrap();
    let config = temporary.path().join("config.toml");
    std::fs::write(
        &config,
        format!(
            "[paths]\ncore_cache='{}'\n",
            temporary.path().join("cores").display()
        ),
    )
    .unwrap();
    let original = temporary.path().join("original.safetensors");
    let replacement = temporary.path().join("replacement.safetensors");
    std::fs::write(&original, b"unreadable core").unwrap();
    std::fs::write(&replacement, b"another unreadable core").unwrap();
    let controller = Controller::new(Some(config)).unwrap();
    wait(&controller, |view| {
        !view.busy && view.current_operation == 0
    });
    controller
        .submit(Command::OpenModel(original.clone()))
        .unwrap();
    let view = wait(&controller, |view| {
        !view.busy
            && view
                .models
                .iter()
                .any(|model| same_path(&model.path, &original))
    });
    let id = view
        .models
        .iter()
        .find(|model| same_path(&model.path, &original))
        .unwrap()
        .id
        .clone();
    controller
        .submit(Command::InspectModel(id.clone()))
        .unwrap();
    let view = wait(&controller, |view| {
        !view.detail_loading
            && view
                .model_detail
                .as_ref()
                .is_some_and(|detail| detail.id == id)
    });
    let detail = view.model_detail.unwrap();
    assert!(!detail.signatures_verified);
    assert_eq!(detail.compatibility, Compatibility::Unknown);
    assert!(!detail.inspection_error.is_empty());
    controller
        .submit(Command::LocateReference {
            id,
            path: replacement.clone(),
        })
        .unwrap();
    let view = wait(&controller, |view| {
        !view.busy && view.models.iter().filter(|model| model.flags & 2 != 0).count() == 1
            && view.models.iter().any(|model| same_path(&model.path, &replacement))
    });
    controller
        .submit(Command::RemoveReference(view.models.iter().find(|model| same_path(&model.path, &replacement)).unwrap().id.clone()))
        .unwrap();
    wait(&controller, |view| !view.busy && view.models.iter().all(|model| model.flags & 2 == 0));
    assert_eq!(std::fs::read(original).unwrap(), b"unreadable core");
    assert_eq!(
        std::fs::read(replacement).unwrap(),
        b"another unreadable core"
    );
    controller.close();
    let deadline = Instant::now() + Duration::from_secs(15);
    while !controller.is_closed() {
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(20));
    }
}
