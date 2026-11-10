/* SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct flare_app flare_app;
typedef struct flare_snapshot flare_snapshot;
/* Call once at native entry before initializing WinUI or creating threads. */
uint32_t flare_runtime_initialize(void);
/* Snapshot strings are borrowed until flare_snapshot_destroy. Command input is
   copied before a call returns. */
typedef struct flare_string { const char* data; size_t length; } flare_string;
enum {
    FLARE_IDLE = 0, FLARE_LOADING = 1, FLARE_READY = 2,
    FLARE_DRAINING = 3, FLARE_STOPPING = 4, FLARE_FAILED = 5,
    FLARE_RESTART_REQUIRED = 6, FLARE_REVIEW_REQUIRED = 7
};
enum {
    FLARE_MODEL_LOCAL = 1, FLARE_MODEL_EXTERNAL = 2,
    FLARE_MODEL_ACTIVE = 4, FLARE_MODEL_VERIFIED = 8,
    FLARE_MODEL_CATALOG = 16, FLARE_MODEL_RECOMMENDED = 32
};
enum {
    FLARE_DOWNLOAD_QUEUED = 0, FLARE_DOWNLOAD_RUNNING = 1,
    FLARE_DOWNLOAD_COMPLETE = 2, FLARE_DOWNLOAD_FAILED = 3,
    FLARE_DOWNLOAD_CANCELLED = 4, FLARE_DOWNLOAD_PAUSED = 5
};
typedef struct flare_status {
    uint32_t state;
    uint32_t has_metrics;
    uint64_t session_id;
    flare_string active_model;
    flare_string detail;
    flare_string error;
    flare_string endpoint;
    flare_string model_directory;
    flare_string stage;
    uint64_t stage_completed;
    uint64_t stage_total;
    uint32_t stage_total_known;
    uint32_t accepting;
    uint64_t active_requests;
    uint64_t waiting_requests;
    uint64_t ingress_requests;
    double generated_tokens_per_second;
    double prefill_tokens_per_second;
    uint64_t generated_tokens;
    uint64_t cached_pages;
    uint64_t total_pages;
    uint32_t busy;
    uint32_t catalog_loading;
    uint64_t current_operation;
    uint32_t detail_loading;
    flare_string gpu_error;
    uint32_t appearance_theme, appearance_compact;
    uint64_t hardware_sample_age_ms;
    uint32_t has_hardware_sample_age;
    double first_response_seconds;
    uint64_t first_response_samples, first_response_window_seconds;
    uint32_t has_first_response, first_response_truncated;
} flare_status;

enum {
    FLARE_DASHBOARD_ACCEPTANCE = 1, FLARE_DASHBOARD_DEPTH = 2,
    FLARE_DASHBOARD_WEIGHT_BANDWIDTH = 4
};
typedef struct flare_dashboard {
    uint32_t available, known_fields;
    uint64_t session_id;
    double uptime_seconds, prefill_tokens_per_second, generated_tokens_per_second;
    double verify_steps_per_second;
    /* Drafts per offering round, then drafts per verification request round. */
    double acceptance, depth, weight_bytes_per_second;
    uint64_t prefill_tokens, generated_tokens, reused_tokens;
    uint64_t active, pending, completed, failed, ingress;
    uint64_t cached_pages, free_pages, total_pages, page_tokens;
    uint64_t host_hits, host_reused_tokens, host_memory_records, host_disk_records;
    uint64_t host_bytes_out, host_bytes_in;
} flare_dashboard;
typedef struct flare_dashboard_point {
    double elapsed_seconds, prefill_tokens_per_second, generated_tokens_per_second;
    double verify_steps_per_second, acceptance, depth;
    uint32_t known_fields;
} flare_dashboard_point;

/* Desktop overrides. Empty memory/budget/path values inherit configuration or
   core defaults. Optional booleans: 0 inherit, 1 off, 2 on. Read-only output
   fields are ignored on input; authentication secrets are never returned. */
typedef struct flare_runtime_settings {
    flare_string memory_limit, kv_cache_limit, host;
    flare_string prefix_cache_memory, prefix_cache_disk, prefix_cache_disk_limit;
    flare_string embedding_table_storage, embedding_table_cache_size;
    flare_string prefix_cache_max_age, prefix_cache_drain_on_shutdown;
    uint64_t max_concurrency, mtp_draft_tokens;
    uint32_t port, verify_weights, has_max_concurrency, has_mtp_draft_tokens;
    uint32_t history_drafts, adaptive_verification, pending_reload, has_api_key;
    uint32_t prefix_cache_disk_enabled;
    uint64_t last_settings_operation;
    flare_string settings_error, active_bind;
} flare_runtime_settings;

enum { FLARE_COMPATIBILITY_UNKNOWN=0, FLARE_COMPATIBILITY_COMPATIBLE=1, FLARE_COMPATIBILITY_INCOMPATIBLE=2 };
enum { FLARE_DETAIL_CONTEXT=1, FLARE_DETAIL_CONCURRENCY=2, FLARE_DETAIL_MEMORY=4, FLARE_DETAIL_RANKS=8 };
typedef struct flare_model_detail {
    flare_string id, name, path, publisher, official_source, source_revision;
    flare_string metadata_status, inspection_error, architecture, compatibility_reason, memory_note;
    uint64_t context_tokens, max_concurrency, prepared_memory_bytes;
    uint32_t rank_count, known_fields, compatibility, can_remove_reference, signatures_verified;
} flare_model_detail;
typedef struct flare_gpu {
    uint32_t ordinal, visible_ordinal, has_visible_ordinal;
    flare_string name, uuid, platform;
    uint64_t total_memory_bytes, free_memory_bytes;
    uint32_t has_total_memory, has_free_memory, has_architecture, architecture_major, architecture_minor;
} flare_gpu;
typedef struct flare_preview {
    uint64_t operation_id, session_id;
    uint32_t busy, cancelled;
    flare_string request_id, text, reasoning, error;
    uint64_t conversation_id;
} flare_preview;
typedef struct flare_chat_message {
    flare_string role, content, reasoning, error;
    uint32_t cancelled;
} flare_chat_message;
typedef struct flare_model {
    flare_string id;
    flare_string name;
    flare_string version;
    flare_string source;
    flare_string path;
    flare_string description;
    flare_string error;
    uint64_t logical_bytes;
    uint32_t size_known;
    uint32_t flags;
    flare_string author, task, license, quantization, hardware, revision, revision_commit, base_model;
    uint64_t context_tokens, parameter_count, revision_bytes;
    uint32_t rank_count;
    uint32_t metadata_known;
} flare_model;
enum {
    FLARE_MODEL_CONTEXT = 1, FLARE_MODEL_PARAMETERS = 2,
    FLARE_MODEL_REVISION_SIZE = 4, FLARE_MODEL_RANKS = 8
};
typedef struct flare_download {
    uint64_t operation_id;
    uint32_t state;
    uint32_t total_known;
    flare_string name;
    flare_string source;
    flare_string phase;
    flare_string error;
    uint64_t network_bytes;
    uint64_t network_total;
    uint64_t verified_bytes;
    uint64_t materialized_bytes;
    double bytes_per_second;
    uint32_t can_cancel;
    uint32_t can_resume;
    flare_string installed_model;
    uint32_t transfer_active;
    /* Companion metadata probes during download preparation. */
    uint64_t planning_completed, planning_total;
    uint32_t planning_total_known;
} flare_download;
typedef struct flare_trust_review {
    uint64_t decision_id;
    flare_string publisher;
    flare_string fingerprint;
    flare_string official_source;
    flare_string license;
    flare_string identity_status;
} flare_trust_review;
/* Empty config path uses the user's Flare configuration. Windows paths use
   counted UTF-16 and are never converted through a display string. */
flare_app* flare_app_create(const uint16_t* config_path, size_t length);
/* Close cooperatively. Poll is_closed before destroy; keep the UI responsive
   while native work drains. */
void flare_app_close(flare_app* app);
uint32_t flare_app_is_closed(const flare_app* app);
void flare_app_destroy(flare_app* app);
/* Last failed ABI call on this thread. Valid until the next call on this thread.
   A command returning zero was not accepted. */
flare_string flare_last_error(void);
flare_snapshot* flare_app_snapshot(const flare_app* app);
void flare_snapshot_destroy(flare_snapshot* snapshot);
uint32_t flare_snapshot_status(const flare_snapshot* snapshot, flare_status* value);
uint32_t flare_snapshot_dashboard(const flare_snapshot* snapshot, flare_dashboard* value);
size_t flare_snapshot_dashboard_point_count(const flare_snapshot* snapshot);
uint32_t flare_snapshot_dashboard_point(const flare_snapshot* snapshot, size_t index, flare_dashboard_point* value);
uint32_t flare_snapshot_runtime_settings(const flare_snapshot* snapshot, flare_runtime_settings* value);
uint64_t flare_app_set_runtime_settings(flare_app* app, const flare_runtime_settings* value);
size_t flare_snapshot_model_count(const flare_snapshot* snapshot);
uint32_t flare_snapshot_model(const flare_snapshot* snapshot, size_t index, flare_model* value);
size_t flare_snapshot_download_count(const flare_snapshot* snapshot);
uint32_t flare_snapshot_download(const flare_snapshot* snapshot, size_t index, flare_download* value);
uint32_t flare_snapshot_trust(const flare_snapshot* snapshot, flare_trust_review* value);
uint32_t flare_snapshot_model_detail(const flare_snapshot* snapshot, flare_model_detail* value);
size_t flare_snapshot_gpu_count(const flare_snapshot* snapshot);
uint32_t flare_snapshot_gpu(const flare_snapshot* snapshot, size_t index, flare_gpu* value);
uint32_t flare_snapshot_preview(const flare_snapshot* snapshot, flare_preview* value);
size_t flare_snapshot_chat_message_count(const flare_snapshot* snapshot);
uint32_t flare_snapshot_chat_message(const flare_snapshot* snapshot, size_t index, flare_chat_message* value);
uint64_t flare_app_refresh(flare_app* app);
uint64_t flare_app_discover(flare_app* app, const char* query, size_t length);
uint64_t flare_app_open_model(flare_app* app, const uint16_t* path, size_t length);
uint64_t flare_app_download(flare_app* app, const char* repository, size_t length);
uint64_t flare_app_activate(flare_app* app, const char* model_id, size_t length);
uint64_t flare_app_stop(flare_app* app);
uint64_t flare_app_cancel(flare_app* app, uint64_t operation_id);
uint64_t flare_app_resume(flare_app* app, uint64_t operation_id);
uint64_t flare_app_pause_download(flare_app* app, uint64_t operation_id);
uint64_t flare_app_accept_trust(flare_app* app, uint64_t decision_id,
    const char* publisher, size_t publisher_length);
uint64_t flare_app_reject_trust(flare_app* app, uint64_t decision_id);
uint64_t flare_app_set_model_directory(flare_app* app, const uint16_t* path, size_t length);
uint64_t flare_app_inspect_model(flare_app* app, const char* id, size_t length);
uint64_t flare_app_remove_reference(flare_app* app, const char* id, size_t length);
uint64_t flare_app_locate_reference(flare_app* app, const char* id, size_t id_length, const uint16_t* path, size_t path_length);
uint64_t flare_app_try_model(flare_app* app, const char* prompt, size_t length);
uint32_t flare_app_cancel_preview(flare_app* app);
uint32_t flare_app_clear_chat(flare_app* app);
uint64_t flare_app_set_appearance(flare_app* app, uint32_t theme, uint32_t compact);
#ifdef __cplusplus
}
#endif
