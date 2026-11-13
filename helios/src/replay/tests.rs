/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::*;

fn test_worker(
    capacity: usize,
    mut execute: impl FnMut(usize, u32) -> Result<()> + Send + 'static,
    finish: impl FnOnce() -> Shutdown + Send + 'static,
) -> ReplayWorker {
    ReplayWorker::spawn(2, capacity, move |receiver, sender, ready| {
        ready.send(Ok(())).unwrap();
        let failure = command_loop(receiver, sender, &mut execute);
        let mut shutdown = finish();
        shutdown.error = shutdown.error.or(failure);
        shutdown
    })
    .unwrap()
}

fn clean_shutdown() -> Shutdown {
    Shutdown {
        drained: true,
        error: None,
    }
}

fn bundle() -> GraphBundle {
    GraphBundle {
        context: 1,
        device: 0,
        graph_execs: vec![2, 3],
        buffers: vec![BufferSpan {
            address: 4,
            nbytes: 16,
        }],
    }
}

#[test]
fn validates_graph_and_buffer_metadata_without_loading_cuda() {
    assert!(bundle().validate().is_ok());
    let mut invalid = bundle();
    invalid.context = 0;
    assert!(invalid.validate().is_err());
    invalid = bundle();
    invalid.graph_execs.push(2);
    assert!(invalid.validate().is_err());
    invalid = bundle();
    invalid.graph_execs[0] = 0;
    assert!(invalid.validate().is_err());
    invalid = bundle();
    invalid.buffers[0].address = usize::MAX;
    assert!(invalid.validate().is_err());
}

#[test]
fn queues_replay_commands_in_order_and_completes_after_execution() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let seen = calls.clone();
    let worker = test_worker(
        4,
        move |index, repetitions| {
            seen.lock().unwrap().push((index, repetitions));
            Ok(())
        },
        clean_shutdown,
    );
    let first = worker.submit(1, 5).unwrap();
    let second = worker.submit(0, 3).unwrap();
    assert_eq!(
        worker.poll(Duration::from_secs(1)).unwrap().unwrap().ticket,
        first
    );
    assert_eq!(
        worker.poll(Duration::from_secs(1)).unwrap().unwrap().ticket,
        second
    );
    assert_eq!(*calls.lock().unwrap(), [(1, 5), (0, 3)]);
    assert_eq!(worker.outstanding(), 0);
    assert!(worker.close().drained);
}

#[test]
fn unconsumed_completions_count_against_capacity() {
    let worker = test_worker(1, |_, _| Ok(()), clean_shutdown);
    worker.submit(0, 1).unwrap();
    assert!(matches!(
        worker.submit(0, 1).unwrap_err(),
        Error::Execution(ExecutionError::Capacity(_))
    ));
    worker.poll(Duration::from_secs(1)).unwrap().unwrap();
    worker.submit(0, 1).unwrap();
    assert!(worker.close().drained);
    assert!(worker.poll(Duration::ZERO).unwrap().is_some());
}

#[test]
fn close_drains_queued_work_and_is_idempotent() {
    let count = Arc::new(AtomicUsize::new(0));
    let executed = count.clone();
    let drained_count = count.clone();
    let worker = test_worker(
        4,
        move |_, _| {
            executed.fetch_add(1, Ordering::SeqCst);
            Ok(())
        },
        move || {
            assert_eq!(drained_count.load(Ordering::SeqCst), 3);
            clean_shutdown()
        },
    );
    for _ in 0..3 {
        worker.submit(0, 1).unwrap();
    }
    assert!(worker.close().drained);
    assert!(worker.close().drained);
    assert!(matches!(
        worker.submit(0, 1).unwrap_err(),
        Error::Execution(ExecutionError::Closed(_))
    ));
    assert_eq!(count.load(Ordering::SeqCst), 3);
}

#[test]
fn execution_failure_poisoning_prevents_later_graph_launches() {
    let count = Arc::new(AtomicUsize::new(0));
    let executed = count.clone();
    let worker = test_worker(
        3,
        move |_, _| {
            executed.fetch_add(1, Ordering::SeqCst);
            Err(driver_failure("cuGraphLaunch", 700))
        },
        clean_shutdown,
    );
    for _ in 0..3 {
        worker.submit(0, 1).unwrap();
    }
    for _ in 0..3 {
        let error = worker
            .poll(Duration::from_secs(1))
            .unwrap()
            .unwrap()
            .error
            .unwrap();
        assert!(matches!(
            error,
            Error::Execution(ExecutionError::Driver(crate::CudaError { code: 700, .. }))
        ));
        assert_eq!(
            error.to_string(),
            "cuGraphLaunch failed (700, CUDA_ERROR_ILLEGAL_ADDRESS): an illegal memory access was encountered"
        );
    }
    assert_eq!(count.load(Ordering::SeqCst), 1);
    let shutdown = worker.close();
    assert!(shutdown.drained);
    assert!(matches!(
        shutdown.error,
        Some(Error::Execution(ExecutionError::Driver(crate::CudaError {
            code: 700,
            ..
        })))
    ));
}

#[test]
fn failed_drain_is_preserved_on_repeated_close() {
    let worker = test_worker(
        1,
        |_, _| Ok(()),
        || Shutdown {
            drained: false,
            error: Some(driver_failure("cuStreamSynchronize", 700)),
        },
    );
    for _ in 0..2 {
        let shutdown = worker.close();
        assert!(!shutdown.drained);
        assert!(matches!(
            shutdown.error,
            Some(Error::Execution(ExecutionError::Driver(crate::CudaError {
                code: 700,
                ..
            })))
        ));
    }
}

#[test]
fn rejects_invalid_commands_without_consuming_queue_capacity() {
    let worker = test_worker(1, |_, _| Ok(()), clean_shutdown);
    assert!(worker.submit(2, 1).is_err());
    assert!(worker.submit(0, 0).is_err());
    assert_eq!(worker.outstanding(), 0);
    worker.submit(0, 1).unwrap();
}

#[test]
fn poll_timeout_does_not_block_another_threads_submission() {
    let worker = Arc::new(test_worker(1, |_, _| Ok(()), clean_shutdown));
    let waiter = worker.clone();
    let thread = thread::spawn(move || waiter.poll(Duration::from_secs(1)));
    let ticket = worker.submit(0, 1).unwrap();
    assert_eq!(thread.join().unwrap().unwrap().unwrap().ticket, ticket);
}

#[test]
fn worker_panic_reports_disconnected_submissions_instead_of_timeout() {
    let worker = ReplayWorker::spawn(1, 1, |receiver, sender, ready| {
        ready.send(Ok(())).unwrap();
        receiver.recv().unwrap();
        let _completion_sender = sender;
        panic!("injected worker failure before completing a submission");
    })
    .unwrap();
    worker.submit(0, 1).unwrap();
    let error = worker.poll(Duration::from_secs(1)).unwrap_err();
    assert!(matches!(error, Error::Execution(ExecutionError::Worker(_))));
    assert_eq!(
        error.to_string(),
        "native replay worker disconnected with 1 outstanding submissions; close the worker to confirm stream drain"
    );
    assert!(!worker.close().drained);
    assert!(worker.poll(Duration::ZERO).is_err());
}

#[test]
fn drained_disconnected_worker_has_no_pending_event() {
    let worker = test_worker(1, |_, _| Ok(()), clean_shutdown);
    worker.submit(0, 1).unwrap();
    assert!(worker.close().drained);
    assert!(worker.poll(Duration::ZERO).unwrap().is_some());
    assert!(worker.poll(Duration::ZERO).unwrap().is_none());
}

fn driver_failure(operation: &str, code: i32) -> Error {
    ExecutionError::Driver(crate::CudaError {
        operation: operation.to_owned(),
        code,
        name: "CUDA_ERROR_ILLEGAL_ADDRESS".into(),
        description: "an illegal memory access was encountered".into(),
    })
    .into()
}

#[test]
fn initialization_failure_preserves_typed_driver_error() {
    let result = ReplayWorker::spawn(1, 1, |_receiver, _sender, ready| {
        let error = driver_failure("cuStreamCreate", 700);
        ready.send(Err(error.clone())).unwrap();
        Shutdown {
            drained: true,
            error: Some(error),
        }
    });
    let error = match result {
        Err(error) => error,
        Ok(_) => panic!("worker unexpectedly initialized"),
    };
    assert!(matches!(
        error,
        Error::Execution(ExecutionError::Driver(crate::CudaError { code: 700, .. }))
    ));
}

#[test]
fn disconnected_startup_preserves_channel_cause() {
    let result = ReplayWorker::spawn(1, 1, |_receiver, _sender, ready| {
        drop(ready);
        clean_shutdown()
    });
    let error = match result {
        Err(error) => error,
        Ok(_) => panic!("worker unexpectedly initialized"),
    };
    assert!(matches!(error, Error::Execution(ExecutionError::Worker(_))));
    assert_eq!(
        error.to_string(),
        "native replay worker exited during initialization"
    );
    assert!(
        std::error::Error::source(&error)
            .unwrap()
            .is::<mpsc::RecvError>()
    );
}
