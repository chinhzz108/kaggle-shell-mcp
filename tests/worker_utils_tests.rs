//! Offline tests for auth config, bootstrap bundle rendering, tqdm compaction,
//! status normalization, and full mock-server flows for KaggleApiClient,
//! launch_worker, and WorkerClient (including auto-healing endpoint refresh).

use std::fs;
use std::io::{Cursor, Write};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use std::time::Duration;

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::IntoResponse;
use axum::routing::{get, post, put};
use axum::{Json, Router};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use regex::Regex;
use serde_json::{Value, json};
use tempfile::tempdir;
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use zip::ZipWriter;
use zip::write::SimpleFileOptions;

use kaggle_shell_mcp::client::WorkerClient;
use kaggle_shell_mcp::config::{DEFAULT_IDLE_TIMEOUT_MIN, KaggleAuth, Settings};
use kaggle_shell_mcp::launcher::{
    LaunchOptions, build_bootstrap_bundle, launch_worker, pull_kernel_outputs,
};
use kaggle_shell_mcp::sync::{pack_tar_gz, unpack_tar_gz};
use kaggle_shell_mcp::utils::{clean_tqdm, normalize_kernel_status};

#[test]
fn kaggle_auth_distinguishes_kgat_and_legacy() {
    assert_eq!(
        KaggleAuth::from_credentials("alice", "KGAT_secret123"),
        KaggleAuth::Bearer("KGAT_secret123".to_string())
    );
    assert_eq!(
        KaggleAuth::from_credentials("bob", "0123456789abcdef0123456789abcdef"),
        KaggleAuth::Basic {
            username: "bob".to_string(),
            key: "0123456789abcdef0123456789abcdef".to_string(),
        }
    );
}

#[test]
fn bootstrap_bundle_renders_valid_python_and_metadata() {
    let tmp = tempdir().unwrap();
    let settings = Settings::new("tester", "KGAT_dummy").with_state_dir(tmp.path());

    let bundle = build_bootstrap_bundle(
        &settings,
        "sess123",
        "tok456",
        true,
        DEFAULT_IDLE_TIMEOUT_MIN,
        &["tester/my-data".to_string()],
        true,
    )
    .unwrap();

    let worker_src = fs::read_to_string(bundle.bundle_dir.join("bootstrap_worker.py")).unwrap();
    assert!(!worker_src.contains("__BOOTSTRAP_CONFIG_B64__"));

    let re = Regex::new(r#"BOOTSTRAP_CONFIG_B64 = "([^"]+)""#).unwrap();
    let cap = re.captures(&worker_src).expect("config base64 placeholder");
    let decoded_bytes = BASE64.decode(&cap[1]).unwrap();
    let decoded_cfg: Value = serde_json::from_slice(&decoded_bytes).unwrap();
    assert_eq!(decoded_cfg["session_id"], "sess123");
    assert_eq!(decoded_cfg["worker_token"], "tok456");
    assert_eq!(decoded_cfg["idle_timeout_sec"], 0);
    assert_eq!(decoded_cfg["rpc_port"], 8765);

    let meta_str = fs::read_to_string(bundle.bundle_dir.join("kernel-metadata.json")).unwrap();
    let meta: Value = serde_json::from_str(&meta_str).unwrap();
    assert_eq!(meta["id"], "tester/kaggle-mcp-worker");
    assert_eq!(meta["enable_gpu"], true);
    assert_eq!(meta["machine_shape"], "NvidiaTeslaT4");
    assert_eq!(meta["is_private"], true);
    assert_eq!(meta["dataset_sources"], json!(["tester/my-data"]));
    assert_eq!(meta["kernel_sources"], json!(["tester/kaggle-mcp-worker"]));
}

#[test]
fn clean_tqdm_collapses_carriage_returns_and_ansi() {
    let noisy = "Epoch 1\r\n10%|#   |\r50%|##### |\r100%|##########| 10/10 [00:02]\r\nloss=0.12";
    assert_eq!(
        clean_tqdm(noisy),
        "Epoch 1\n100%|##########| 10/10 [00:02]\nloss=0.12"
    );

    let ansi = "\x1b[32mOK\x1b[0m\r\x1b[1mDONE\x1b[0m";
    assert_eq!(clean_tqdm(ansi), "DONE");
}

#[test]
fn normalize_kernel_status_handles_variants() {
    assert_eq!(normalize_kernel_status("running"), "RUNNING");
    assert_eq!(normalize_kernel_status("KernelWorkerStatus.COMPLETE"), "COMPLETE");
    assert_eq!(normalize_kernel_status("KERNEL_WORKER_STATUS_QUEUED"), "QUEUED");
    assert_eq!(normalize_kernel_status("cancelRequested"), "CANCEL_REQUESTED");
    assert_eq!(normalize_kernel_status(""), "UNKNOWN");
}

// ---------------------------------------------------------------------------
// Mock Kaggle API + Mock Worker RPC End-to-End Test
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct MockKaggleState {
    dataset_exists: Arc<AtomicBool>,
    saved_session_id: Arc<Mutex<String>>,
    saved_worker_token: Arc<Mutex<String>>,
    worker_url: Arc<Mutex<String>>,
    api_base_url: Arc<Mutex<String>>,
    blob_uploaded: Arc<AtomicBool>,
}

#[derive(Clone)]
struct MockWorkerState {
    session_id: Arc<Mutex<String>>,
    token: Arc<Mutex<String>>,
    fail_first_status_502: Arc<AtomicBool>,
    status_calls: Arc<AtomicUsize>,
    upload_dir: std::path::PathBuf,
    download_file: std::path::PathBuf,
}

fn build_zip_bytes(file_name: &str, content: &[u8]) -> Vec<u8> {
    let mut cursor = Cursor::new(Vec::new());
    {
        let mut zip = ZipWriter::new(&mut cursor);
        zip.start_file(file_name, SimpleFileOptions::default())
            .unwrap();
        zip.write_all(content).unwrap();
        zip.finish().unwrap();
    }
    cursor.into_inner()
}

#[tokio::test]
async fn mock_end_to_end_launch_exec_sync_refresh_and_pull() {
    let tmp = tempdir().unwrap();
    let state_dir = tmp.path().join("state");
    let worker_upload_dir = tmp.path().join("remote_uploads");
    let worker_download_src = tmp.path().join("remote_artifact.txt");
    fs::write(&worker_download_src, b"checkpoint-weights-v1").unwrap();

    // 1. Spawn Mock Worker Server
    let w_session = Arc::new(Mutex::new(String::new()));
    let w_token = Arc::new(Mutex::new(String::new()));
    let w_state = MockWorkerState {
        session_id: w_session.clone(),
        token: w_token.clone(),
        fail_first_status_502: Arc::new(AtomicBool::new(true)),
        status_calls: Arc::new(AtomicUsize::new(0)),
        upload_dir: worker_upload_dir.clone(),
        download_file: worker_download_src.clone(),
    };

    let worker_app = Router::new()
        .route(
            "/ping",
            get(|State(st): State<MockWorkerState>| async move {
                let sid = st.session_id.lock().await.clone();
                Json(json!({ "ok": true, "session_id": sid, "uptime_sec": 12.5 }))
            }),
        )
        .route(
            "/status",
            get(
                |State(st): State<MockWorkerState>, headers: HeaderMap| async move {
                    let expected = format!("Bearer {}", *st.token.lock().await);
                    let got = headers
                        .get("authorization")
                        .and_then(|v| v.to_str().ok())
                        .unwrap_or("");
                    if got != expected {
                        return (StatusCode::UNAUTHORIZED, Json(json!({"error": "Unauthorized"})));
                    }
                    st.status_calls.fetch_add(1, Ordering::SeqCst);
                    if st.fail_first_status_502.swap(false, Ordering::SeqCst) {
                        return (StatusCode::BAD_GATEWAY, Json(json!({"error": "tunnel hiccup"})));
                    }
                    (
                        StatusCode::OK,
                        Json(json!({
                            "session_id": *st.session_id.lock().await,
                            "active_jobs": 0,
                            "gpus": [{"index": 0, "name": "Tesla T4", "util_pct": 0.0}],
                        })),
                    )
                },
            ),
        )
        .route(
            "/term/exec",
            post(
                |State(st): State<MockWorkerState>,
                 headers: HeaderMap,
                 Json(body): Json<Value>| async move {
                    let expected = format!("Bearer {}", *st.token.lock().await);
                    if headers.get("authorization").and_then(|v| v.to_str().ok()) != Some(&expected) {
                        return (StatusCode::UNAUTHORIZED, Json(json!({"error": "Unauthorized"})));
                    }
                    (
                        StatusCode::OK,
                        Json(json!({
                            "session": body["session"],
                            "exit_code": 0,
                            "cwd": "/kaggle/working",
                            "timed_out": false,
                            "output": format!("ran: {}", body["command"].as_str().unwrap_or("")),
                        })),
                    )
                },
            ),
        )
        .route(
            "/fs/upload",
            post(
                |State(st): State<MockWorkerState>, headers: HeaderMap, body: Bytes| async move {
                    let expected = format!("Bearer {}", *st.token.lock().await);
                    if headers.get("authorization").and_then(|v| v.to_str().ok()) != Some(&expected) {
                        return (StatusCode::UNAUTHORIZED, Json(json!({"error": "Unauthorized"})));
                    }
                    let extracted = unpack_tar_gz(&body, &st.upload_dir).unwrap();
                    (
                        StatusCode::OK,
                        Json(json!({
                            "ok": true,
                            "extracted_entries": extracted.len(),
                            "remote_path": headers
                                .get("x-remote-path")
                                .and_then(|v| v.to_str().ok())
                                .unwrap_or("/kaggle/working"),
                        })),
                    )
                },
            ),
        )
        .route(
            "/fs/download",
            post(
                |State(st): State<MockWorkerState>, headers: HeaderMap| async move {
                    let expected = format!("Bearer {}", *st.token.lock().await);
                    if headers.get("authorization").and_then(|v| v.to_str().ok()) != Some(&expected) {
                        return (StatusCode::UNAUTHORIZED, Vec::<u8>::new()).into_response();
                    }
                    let tar_bytes = pack_tar_gz(&st.download_file).unwrap();
                    (StatusCode::OK, tar_bytes).into_response()
                },
            ),
        )
        .route(
            "/shutdown",
            post(|| async { Json(json!({ "ok": true, "shutting_down": true })) }),
        )
        .with_state(w_state.clone());

    let worker_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let worker_addr = format!("http://{}", worker_listener.local_addr().unwrap());
    tokio::spawn(async move {
        axum::serve(worker_listener, worker_app).await.unwrap();
    });

    // 2. Spawn Mock Kaggle API Server
    let k_state = MockKaggleState {
        dataset_exists: Arc::new(AtomicBool::new(false)),
        saved_session_id: w_session.clone(),
        saved_worker_token: w_token.clone(),
        worker_url: Arc::new(Mutex::new(worker_addr.clone())),
        api_base_url: Arc::new(Mutex::new(String::new())),
        blob_uploaded: Arc::new(AtomicBool::new(false)),
    };

    let kaggle_app = Router::new()
        .route(
            "/v1/datasets.DatasetApiService/ListDatasetFiles",
            post(|State(st): State<MockKaggleState>, headers: HeaderMap| async move {
                assert_eq!(
                    headers.get("authorization").and_then(|v| v.to_str().ok()),
                    Some("Bearer KGAT_test_token")
                );
                if st.dataset_exists.load(Ordering::SeqCst) {
                    (StatusCode::OK, Json(json!({ "datasetFiles": [] })))
                } else {
                    (
                        StatusCode::NOT_FOUND,
                        Json(json!({ "code": 404, "message": "Dataset not found" })),
                    )
                }
            }),
        )
        .route(
            "/v1/blobs.BlobApiService/StartBlobUpload",
            post(|State(st): State<MockKaggleState>| async move {
                let base = st.api_base_url.lock().await.clone();
                Json(json!({
                    "token": "blob-tok-123",
                    "createUrl": format!("{base}/gcs/upload/blob-tok-123"),
                }))
            }),
        )
        .route(
            "/gcs/upload/blob-tok-123",
            put(|State(st): State<MockKaggleState>, headers: HeaderMap, body: Bytes| async move {
                // Signed GCS upload must NOT receive Kaggle Authorization header
                assert!(headers.get("authorization").is_none());
                assert!(!body.is_empty());
                st.blob_uploaded.store(true, Ordering::SeqCst);
                StatusCode::OK
            }),
        )
        .route(
            "/v1/datasets.DatasetApiService/CreateDataset",
            post(|State(st): State<MockKaggleState>, Json(body): Json<Value>| async move {
                assert_eq!(body["isPrivate"], true);
                assert_eq!(body["files"][0]["token"], "blob-tok-123");
                st.dataset_exists.store(true, Ordering::SeqCst);
                Json(json!({ "ref": "tester/kaggle-mcp-state", "status": "ok" }))
            }),
        )
        .route(
            "/v1/kernels.KernelsApiService/SaveKernel",
            post(|State(st): State<MockKaggleState>, Json(body): Json<Value>| async move {
                assert_eq!(body["machineShape"], "NvidiaTeslaT4");
                assert_eq!(body["enableGpu"], true);
                let src = body["text"].as_str().unwrap();
                let re = Regex::new(r#"BOOTSTRAP_CONFIG_B64 = "([^"]+)""#).unwrap();
                let cap = re.captures(src).unwrap();
                let cfg: Value =
                    serde_json::from_slice(&BASE64.decode(&cap[1]).unwrap()).unwrap();
                *st.saved_session_id.lock().await =
                    cfg["session_id"].as_str().unwrap().to_string();
                *st.saved_worker_token.lock().await =
                    cfg["worker_token"].as_str().unwrap().to_string();
                Json(json!({ "ref": "tester/kaggle-mcp-worker", "versionNumber": 1 }))
            }),
        )
        .route(
            "/v1/kernels.KernelsApiService/GetKernelSessionStatus",
            post(|| async {
                Json(json!({ "status": "running", "failureMessage": "" }))
            }),
        )
        .route(
            "/v1/datasets.DatasetApiService/DownloadDataset",
            post(|State(st): State<MockKaggleState>| async move {
                let base = st.api_base_url.lock().await.clone();
                (
                    StatusCode::FOUND,
                    [("location", format!("{base}/gcs/download/state.zip"))],
                )
            }),
        )
        .route(
            "/gcs/download/state.zip",
            get(|State(st): State<MockKaggleState>| async move {
                let sid = st.saved_session_id.lock().await.clone();
                let wurl = st.worker_url.lock().await.clone();
                let payload = json!({
                    "session_id": sid,
                    "status": "running",
                    "tunnel_url": wurl,
                    "ts_url": "",
                    "gpus": [{"index": 0, "name": "Tesla T4"}],
                });
                let zip_bytes =
                    build_zip_bytes("state.json", &serde_json::to_vec(&payload).unwrap());
                (StatusCode::OK, zip_bytes)
            }),
        )
        .route(
            "/v1/kernels.KernelsApiService/ListKernelSessionOutput",
            post(|State(st): State<MockKaggleState>| async move {
                let base = st.api_base_url.lock().await.clone();
                Json(json!({
                    "files": [
                        {
                            "fileName": "checkpoints/last.pt",
                            "url": format!("{base}/gcs/output/last.pt")
                        }
                    ],
                    "log": "worker clean exit(0).",
                    "nextPageToken": null
                }))
            }),
        )
        .route(
            "/gcs/output/last.pt",
            get(|| async { (StatusCode::OK, b"PT_MODEL_DATA".to_vec()) }),
        )
        .with_state(k_state.clone());

    let kaggle_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let kaggle_addr = format!("http://{}", kaggle_listener.local_addr().unwrap());
    *k_state.api_base_url.lock().await = kaggle_addr.clone();
    tokio::spawn(async move {
        axum::serve(kaggle_listener, kaggle_app).await.unwrap();
    });

    // 3. Exercise full lifecycle: launch -> reuse -> auto-healing status -> exec -> push -> pull -> output pull
    let settings = Settings::new("tester", "KGAT_test_token")
        .with_state_dir(&state_dir)
        .with_api_base(&kaggle_addr);

    let launch_res = launch_worker(
        &settings,
        LaunchOptions {
            gpu: true,
            idle_timeout_min: 0,
            wait_ready_sec: 10,
            poll_interval: Duration::from_millis(20),
            dataset_init_delay: Duration::ZERO,
            ..LaunchOptions::default()
        },
    )
    .await
    .unwrap();

    assert_eq!(launch_res["reused_existing"], false);
    assert_eq!(launch_res["kernel_status"], "RUNNING");
    assert_eq!(launch_res["endpoint"], worker_addr);
    assert!(k_state.blob_uploaded.load(Ordering::SeqCst));

    // Reusing existing active session should return reused_existing = true
    let reuse_res = launch_worker(
        &settings,
        LaunchOptions {
            poll_interval: Duration::from_millis(20),
            dataset_init_delay: Duration::ZERO,
            ..LaunchOptions::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(reuse_res["reused_existing"], true);

    // WorkerClient.status() hits 502 on attempt 0 -> refreshes endpoint -> succeeds on attempt 1!
    let client = WorkerClient::new(settings.clone()).unwrap();
    let st_res = client.status().await.unwrap();
    assert_eq!(st_res["active_jobs"], 0);
    assert_eq!(w_state.status_calls.load(Ordering::SeqCst), 2);

    // term_exec
    let exec_res = client.term_exec("nvidia-smi", "default", 30.0).await.unwrap();
    assert_eq!(exec_res["exit_code"], 0);
    assert_eq!(exec_res["output"], "ran: nvidia-smi");

    // fs_push
    let local_proj = tmp.path().join("demo_app");
    fs::create_dir_all(&local_proj).unwrap();
    fs::write(local_proj.join("train.py"), "print('hi')\n").unwrap();
    let push_res = client.fs_push(&local_proj, "/kaggle/working").await.unwrap();
    assert_eq!(push_res["ok"], true);
    assert!(worker_upload_dir.join("demo_app").join("train.py").is_file());

    // fs_pull
    let pulled_dir = tmp.path().join("pulled");
    let pull_res = client
        .fs_pull("/kaggle/working/remote_artifact.txt", &pulled_dir)
        .await
        .unwrap();
    assert_eq!(pull_res["ok"], true);
    assert_eq!(
        fs::read(pulled_dir.join("remote_artifact.txt")).unwrap(),
        b"checkpoint-weights-v1"
    );

    // pull_kernel_outputs (committed Kaggle kernel outputs + log)
    let out_dir = tmp.path().join("kernel_out");
    let out_files = pull_kernel_outputs(&settings, &out_dir).await.unwrap();
    assert_eq!(out_files.len(), 2);
    assert_eq!(
        fs::read(out_dir.join("checkpoints").join("last.pt")).unwrap(),
        b"PT_MODEL_DATA"
    );
    assert_eq!(
        fs::read_to_string(out_dir.join("kaggle-mcp-worker.log")).unwrap(),
        "worker clean exit(0)."
    );
}
