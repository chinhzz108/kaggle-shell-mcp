//! Kaggle API orchestration: bootstrap bundle generation, private dataset handshake,
//! worker launch & tunnel readiness polling, and kernel output pulling.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use reqwest::Client;
use serde_json::{Value, json};

use crate::config::{
    ActiveSession, DEFAULT_IDLE_TIMEOUT_MIN, DRAIN_AT_SEC, MAX_LIFETIME_SEC, Settings,
    load_active_session, save_active_session,
};
use crate::kaggle_api::{KaggleApiClient, SaveKernelRequest};

/// Embedded pure-stdlib Python remote worker template injected into Kaggle Kernels.
pub const WORKER_TEMPLATE: &str = include_str!("remote/worker_template.py");

/// Generated bootstrap artifacts ready to push to Kaggle.
#[derive(Debug, Clone)]
pub struct BootstrapBundle {
    pub bundle_dir: PathBuf,
    pub worker_script: String,
    pub kernel_metadata: Value,
}

/// Options for launching a headless Kaggle worker.
#[derive(Debug, Clone)]
pub struct LaunchOptions {
    pub gpu: bool,
    pub idle_timeout_min: u64,
    pub mount_datasets: Vec<String>,
    pub chain_previous_output: bool,
    pub wait_ready_sec: u64,
    pub force_new: bool,
    pub poll_interval: Duration,
    pub dataset_init_delay: Duration,
}

impl Default for LaunchOptions {
    fn default() -> Self {
        Self {
            gpu: true,
            idle_timeout_min: DEFAULT_IDLE_TIMEOUT_MIN,
            mount_datasets: Vec::new(),
            chain_previous_output: false,
            wait_ready_sec: 240,
            force_new: false,
            poll_interval: Duration::from_secs(6),
            dataset_init_delay: Duration::from_secs(4),
        }
    }
}

fn unix_timestamp_f64() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

fn random_hex(num_bytes: usize) -> String {
    let mut bytes = vec![0u8; num_bytes];
    rand::fill(&mut bytes[..]);
    let mut hex = String::with_capacity(num_bytes * 2);
    for b in bytes {
        use std::fmt::Write;
        let _ = write!(&mut hex, "{b:02x}");
    }
    hex
}

/// Ensure the private handshake dataset `<user>/kaggle-mcp-state` exists.
pub async fn ensure_state_dataset(
    api: &KaggleApiClient,
    settings: &Settings,
    init_delay: Duration,
) -> Result<()> {
    let ds_ref = settings.state_dataset_ref();
    if api.list_dataset_files(&ds_ref).await.is_ok() {
        return Ok(());
    }

    let init_state = json!({
        "session_id": "init",
        "status": "initialized",
        "updated_at": unix_timestamp_f64(),
    });
    let state_bytes = serde_json::to_vec_pretty(&init_state)?;

    api.create_private_dataset_with_file(
        &ds_ref,
        &settings.state_dataset_slug,
        "state.json",
        &state_bytes,
    )
    .await
    .context("Failed to initialize private handshake dataset")?;

    // Give Kaggle backend a brief moment to register the new private dataset slug
    if !init_delay.is_zero() {
        tokio::time::sleep(init_delay).await;
    }
    Ok(())
}

/// Download and parse `state.json` from the private handshake dataset.
pub async fn fetch_remote_state(api: &KaggleApiClient, settings: &Settings) -> Option<Value> {
    let bytes = api
        .download_dataset_file_bytes(&settings.state_dataset_ref(), "state.json")
        .await
        .ok()??;
    serde_json::from_slice(&bytes).ok()
}

/// Probe `GET {base_url}/ping` and verify optional expected `session_id`.
pub async fn probe_worker_ping(base_url: &str, expected_session_id: &str) -> Option<Value> {
    let trimmed = base_url.trim().trim_end_matches('/');
    if trimmed.is_empty() {
        return None;
    }
    let client = Client::builder()
        .timeout(Duration::from_secs(6))
        .build()
        .ok()?;
    let resp = client.get(format!("{trimmed}/ping")).send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let data: Value = resp.json().await.ok()?;
    if !expected_session_id.is_empty()
        && data.get("session_id").and_then(|v| v.as_str()) != Some(expected_session_id)
    {
        return None;
    }
    Some(data)
}

/// Render `bootstrap_worker.py` and `kernel-metadata.json` into `<state_dir>/build_bundle`.
pub fn build_bootstrap_bundle(
    settings: &Settings,
    session_id: &str,
    worker_token: &str,
    gpu: bool,
    idle_timeout_min: u64,
    mount_datasets: &[String],
    chain_previous_output: bool,
) -> Result<BootstrapBundle> {
    let bundle_dir = settings.build_bundle_dir();
    fs::create_dir_all(&bundle_dir)
        .with_context(|| format!("Failed to create bundle dir {}", bundle_dir.display()))?;

    let worker_cfg = json!({
        "session_id": session_id,
        "worker_token": worker_token,
        "kaggle_username": settings.kaggle_username,
        "kaggle_secret": settings.kaggle_secret,
        "state_dataset_ref": settings.state_dataset_ref(),
        "ts_authkey": settings.ts_authkey,
        "ts_hostname": settings.ts_hostname,
        "idle_timeout_sec": idle_timeout_min.saturating_mul(60),
        "max_lifetime_sec": MAX_LIFETIME_SEC,
        "drain_at_sec": DRAIN_AT_SEC,
        "rpc_port": 8765,
    });
    let b64_cfg = BASE64.encode(serde_json::to_vec(&worker_cfg)?);
    let rendered_worker = WORKER_TEMPLATE.replace("__BOOTSTRAP_CONFIG_B64__", &b64_cfg);

    fs::write(bundle_dir.join("bootstrap_worker.py"), &rendered_worker)
        .context("Failed to write bootstrap_worker.py")?;

    let kernel_sources = if chain_previous_output {
        vec![settings.kernel_ref()]
    } else {
        Vec::new()
    };

    let mut kernel_meta = json!({
        "id": settings.kernel_ref(),
        "title": settings.kernel_slug,
        "code_file": "bootstrap_worker.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": true,
        "enable_gpu": gpu,
        "enable_internet": true,
        "dataset_sources": mount_datasets,
        "competition_sources": [],
        "kernel_sources": kernel_sources,
    });
    if gpu && let Some(obj) = kernel_meta.as_object_mut() {
        obj.insert("machine_shape".to_string(), json!("NvidiaTeslaT4"));
    }

    fs::write(
        bundle_dir.join("kernel-metadata.json"),
        serde_json::to_string_pretty(&kernel_meta)?,
    )
    .context("Failed to write kernel-metadata.json")?;

    Ok(BootstrapBundle {
        bundle_dir,
        worker_script: rendered_worker,
        kernel_metadata: kernel_meta,
    })
}

/// Start or reuse a 12h headless Kaggle worker and wait until its RPC tunnel is reachable.
pub async fn launch_worker(settings: &Settings, opts: LaunchOptions) -> Result<Value> {
    let api = KaggleApiClient::new(settings)?;

    // Reuse live worker if already running and responsive
    if !opts.force_new
        && let Some(existing) = load_active_session(&settings.state_dir)
        && let Some(ping) = probe_worker_ping(&existing.base_url(), &existing.session_id).await
    {
        return Ok(json!({
            "reused_existing": true,
            "session_id": existing.session_id,
            "kernel_ref": existing.kernel_ref,
            "endpoint": existing.base_url(),
            "uptime_sec": ping.get("uptime_sec").cloned().unwrap_or(Value::Null),
        }));
    }

    ensure_state_dataset(&api, settings, opts.dataset_init_delay).await?;

    let session_id = random_hex(12);
    let worker_token = random_hex(32);
    let bundle = build_bootstrap_bundle(
        settings,
        &session_id,
        &worker_token,
        opts.gpu,
        opts.idle_timeout_min,
        &opts.mount_datasets,
        opts.chain_previous_output,
    )?;

    let kernel_sources = if opts.chain_previous_output {
        vec![settings.kernel_ref()]
    } else {
        Vec::new()
    };

    let save_req = SaveKernelRequest {
        slug: Some(settings.kernel_ref()),
        new_title: Some(settings.kernel_slug.clone()),
        text: bundle.worker_script,
        language: "python".to_string(),
        kernel_type: "script".to_string(),
        is_private: true,
        enable_gpu: opts.gpu,
        enable_tpu: false,
        enable_internet: true,
        dataset_data_sources: opts.mount_datasets.clone(),
        competition_data_sources: Vec::new(),
        kernel_data_sources: kernel_sources,
        model_data_sources: Vec::new(),
        category_ids: Vec::new(),
        machine_shape: if opts.gpu {
            Some("NvidiaTeslaT4".to_string())
        } else {
            None
        },
    };

    api.save_kernel(&save_req).await?;

    let wait_secs = opts.wait_ready_sec.max(1);
    let deadline = Instant::now() + Duration::from_secs(wait_secs);
    let mut last_kstatus = "PUSHED".to_string();

    while Instant::now() < deadline {
        if !opts.poll_interval.is_zero() {
            tokio::time::sleep(opts.poll_interval).await;
        }

        let kinfo = api.get_kernel_status(&settings.kernel_ref()).await;
        last_kstatus = kinfo.status.clone();

        if matches!(last_kstatus.as_str(), "ERROR" | "CANCELLED") {
            let reason = if kinfo.failure_message.is_empty() {
                "check Kaggle logs".to_string()
            } else {
                kinfo.failure_message
            };
            bail!("Kaggle kernel entered {last_kstatus}: {reason}");
        }

        let Some(rstate) = fetch_remote_state(&api, settings).await else {
            continue;
        };
        if rstate.get("session_id").and_then(|v| v.as_str()) != Some(session_id.as_str()) {
            continue;
        }

        let ts_url = rstate
            .get("ts_url")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        let tunnel_url = rstate
            .get("tunnel_url")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();

        let candidates: Vec<&str> = [ts_url.as_str(), tunnel_url.as_str()]
            .into_iter()
            .filter(|u| !u.is_empty())
            .collect();

        for url in candidates {
            if probe_worker_ping(url, &session_id).await.is_some() {
                let active = ActiveSession {
                    session_id: session_id.clone(),
                    worker_token: worker_token.clone(),
                    kernel_ref: settings.kernel_ref(),
                    state_dataset_ref: settings.state_dataset_ref(),
                    tunnel_url: tunnel_url.clone(),
                    ts_url: if url == ts_url {
                        ts_url.clone()
                    } else {
                        String::new()
                    },
                    created_at: unix_timestamp_f64(),
                    idle_timeout_min: opts.idle_timeout_min,
                };
                save_active_session(&settings.state_dir, &active)?;
                let max_hours = ((MAX_LIFETIME_SEC as f64 / 3600.0) * 100.0).round() / 100.0;
                return Ok(json!({
                    "reused_existing": false,
                    "session_id": session_id,
                    "kernel_ref": settings.kernel_ref(),
                    "kernel_status": last_kstatus,
                    "endpoint": active.base_url(),
                    "gpus": rstate.get("gpus").cloned().unwrap_or_else(|| json!([])),
                    "max_session_hours": max_hours,
                    "idle_timeout_min": opts.idle_timeout_min,
                }));
            }
        }
    }

    // Save pending credentials so `kaggle_status` can attach once GPU queue finishes
    let pending = ActiveSession {
        session_id: session_id.clone(),
        worker_token,
        kernel_ref: settings.kernel_ref(),
        state_dataset_ref: settings.state_dataset_ref(),
        tunnel_url: String::new(),
        ts_url: String::new(),
        created_at: unix_timestamp_f64(),
        idle_timeout_min: opts.idle_timeout_min,
    };
    save_active_session(&settings.state_dir, &pending)?;

    Ok(json!({
        "reused_existing": false,
        "ready": false,
        "session_id": session_id,
        "kernel_ref": settings.kernel_ref(),
        "kernel_status": last_kstatus,
        "message": format!(
            "Kernel pushed (status={last_kstatus}) but tunnel did not become ready within {}s (it may still be queued for GPU). Call `kaggle_status` shortly to attach automatically.",
            opts.wait_ready_sec
        ),
    }))
}

/// Download all committed `/kaggle/working` output files for the configured kernel.
pub async fn pull_kernel_outputs(settings: &Settings, local_dest: &Path) -> Result<Vec<String>> {
    let api = KaggleApiClient::new(settings)?;
    api.pull_kernel_outputs(&settings.kernel_ref(), local_dest)
        .await
}
