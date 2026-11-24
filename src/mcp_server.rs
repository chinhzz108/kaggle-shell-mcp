//! MCP server (via official `rmcp` SDK) and shared operation handlers exposing
//! all 9 Kaggle GPU remote terminal & detached job tools.

use std::path::Path;

use anyhow::{Result, bail};
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, ContentBlock};
use rmcp::{ErrorData as McpError, ServerHandler, ServiceExt, tool, tool_handler, tool_router};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::client::WorkerClient;
use crate::config::{clear_active_session, load_active_session, load_settings};
use crate::kaggle_api::KaggleApiClient;
use crate::launcher::{LaunchOptions, fetch_remote_state, launch_worker, pull_kernel_outputs};

pub const SERVER_INSTRUCTIONS: &str = "Control a headless Kaggle Dual-T4 GPU session (up to 12h) via stateful PTY shell and detached background jobs. Use `kaggle_up` to provision GPUs, `term_exec` for short stateful commands, `job_spawn` / `job_logs` for long training runs, `fs_sync` to push code or pull checkpoints, and `kaggle_down` as soon as work completes to preserve weekly GPU quota.";

pub fn format_json(value: &Value) -> Result<String> {
    Ok(serde_json::to_string_pretty(value)?)
}

fn default_true() -> bool {
    true
}

fn default_wait_ready_sec() -> u64 {
    240
}

fn default_local_output_dir() -> String {
    "./kaggle_outputs".to_string()
}

fn default_session() -> String {
    "default".to_string()
}

fn default_timeout_sec() -> f64 {
    60.0
}

fn default_working_dir() -> String {
    "/kaggle/working".to_string()
}

fn default_tail_lines() -> usize {
    100
}

fn default_signal() -> String {
    "SIGTERM".to_string()
}

// ---------------------------------------------------------------------------
// Tool Parameter Schemas
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, JsonSchema)]
pub struct KaggleUpParams {
    /// Request Dual NVIDIA T4 x2 (`NvidiaTeslaT4`, 2x15GB VRAM). Set false for CPU-only.
    #[serde(default = "default_true")]
    pub gpu: bool,
    /// Optional idle auto-shutdown in minutes (0 = disabled by default; e.g. 30 to auto-stop when idle).
    #[serde(default)]
    pub idle_timeout_min: u64,
    /// Optional list of Kaggle dataset slugs (`["owner/dataset-slug"]`) mounted at `/kaggle/input`.
    #[serde(default)]
    pub mount_datasets: Option<Vec<String>>,
    /// Mount previous `/kaggle/working` output to continue >12h training runs.
    #[serde(default)]
    pub chain_previous_output: bool,
    /// Maximum seconds to wait for the remote worker tunnel to become reachable.
    #[serde(default = "default_wait_ready_sec")]
    pub wait_ready_sec: u64,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct KaggleDownParams {
    /// Download committed `/kaggle/working` outputs after shutting down.
    #[serde(default)]
    pub pull_outputs: bool,
    /// Local destination directory when `pull_outputs` is true.
    #[serde(default = "default_local_output_dir")]
    pub local_output_dir: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct TermExecParams {
    /// Bash command string to execute inside the stateful PTY.
    pub command: String,
    /// Named PTY session (default: `"default"`).
    #[serde(default = "default_session")]
    pub session: String,
    /// Command execution timeout in seconds (default: `60`).
    #[serde(default = "default_timeout_sec")]
    pub timeout_sec: f64,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct JobSpawnParams {
    /// Shell command to launch in a detached process group.
    pub command: String,
    /// Remote working directory (default: `"/kaggle/working"`).
    #[serde(default = "default_working_dir")]
    pub cwd: String,
    /// Optional human-friendly job name.
    #[serde(default)]
    pub name: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct JobLogsParams {
    /// Identifier of the background job returned by `job_spawn` or `job_list`.
    pub job_id: String,
    /// Number of tail lines to return (default: `100`).
    #[serde(default = "default_tail_lines")]
    pub tail_lines: usize,
    /// Optional case-insensitive regex filter applied to cleaned log lines.
    #[serde(default)]
    pub grep: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct JobKillParams {
    /// Identifier of the background job to signal.
    pub job_id: String,
    /// Signal name: `"SIGTERM"` (default) or `"SIGKILL"`.
    #[serde(default = "default_signal")]
    pub signal: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct FsSyncParams {
    /// Transfer direction: `"push"` (local -> Kaggle) or `"pull"` (Kaggle -> local).
    pub direction: String,
    /// Local source file/directory (for `push`) or local destination directory (for `pull`).
    pub local_path: String,
    /// Remote destination directory (for `push`) or remote source path (for `pull`).
    #[serde(default = "default_working_dir")]
    pub remote_path: String,
}

// ---------------------------------------------------------------------------
// Shared Async Operations (used by both MCP server and CLI)
// ---------------------------------------------------------------------------

pub async fn op_kaggle_up(
    gpu: bool,
    idle_timeout_min: u64,
    mount_datasets: Option<Vec<String>>,
    chain_previous_output: bool,
    wait_ready_sec: u64,
) -> Result<String> {
    let settings = load_settings()?;
    let opts = LaunchOptions {
        gpu,
        idle_timeout_min,
        mount_datasets: mount_datasets.unwrap_or_default(),
        chain_previous_output,
        wait_ready_sec,
        ..LaunchOptions::default()
    };
    let res = launch_worker(&settings, opts).await?;
    format_json(&res)
}

pub async fn op_kaggle_status() -> Result<String> {
    let settings = load_settings()?;
    let api = KaggleApiClient::new(&settings)?;
    let kinfo = api.get_kernel_status(&settings.kernel_ref()).await;
    let active = load_active_session(&settings.state_dir);

    let mut result = json!({
        "kernel_ref": settings.kernel_ref(),
        "kaggle_api_status": kinfo.status,
        "failure_message": kinfo.failure_message,
        "local_session_id": active.as_ref().map(|a| a.session_id.as_str()),
    });

    if active.is_some() && kinfo.status == "RUNNING" {
        match WorkerClient::new(settings.clone()) {
            Ok(client) => match client.status().await {
                Ok(live) => {
                    if let Some(obj) = result.as_object_mut() {
                        obj.insert("worker_live".to_string(), live);
                    }
                    return format_json(&result);
                }
                Err(err) => {
                    if let Some(obj) = result.as_object_mut() {
                        obj.insert("worker_connect_note".to_string(), json!(err.to_string()));
                    }
                }
            },
            Err(err) => {
                if let Some(obj) = result.as_object_mut() {
                    obj.insert("worker_connect_note".to_string(), json!(err.to_string()));
                }
            }
        }
    }

    if kinfo.status != "NOT_CREATED_YET"
        && let Some(rstate) = fetch_remote_state(&api, &settings).await
        && let Some(obj) = result.as_object_mut()
    {
        obj.insert("last_handshake_state".to_string(), rstate);
    }

    format_json(&result)
}

pub async fn op_kaggle_down(pull_outputs: bool, local_output_dir: &str) -> Result<String> {
    let settings = load_settings()?;
    let mut out = json!({ "shutdown_sent": false });

    match WorkerClient::new(settings.clone()) {
        Ok(client) => match client.shutdown().await {
            Ok(resp) => {
                if let Some(obj) = out.as_object_mut() {
                    obj.insert("worker_response".to_string(), resp);
                    obj.insert("shutdown_sent".to_string(), json!(true));
                }
            }
            Err(err) => {
                if let Some(obj) = out.as_object_mut() {
                    obj.insert("shutdown_note".to_string(), json!(err.to_string()));
                }
            }
        },
        Err(err) => {
            if let Some(obj) = out.as_object_mut() {
                obj.insert("shutdown_note".to_string(), json!(err.to_string()));
            }
        }
    }

    clear_active_session(&settings.state_dir);

    if pull_outputs {
        let files = pull_kernel_outputs(&settings, Path::new(local_output_dir)).await?;
        if let Some(obj) = out.as_object_mut() {
            obj.insert("pulled_files".to_string(), json!(files));
        }
    }

    format_json(&out)
}

pub async fn op_term_exec(command: &str, session: &str, timeout_sec: f64) -> Result<String> {
    let settings = load_settings()?;
    let client = WorkerClient::new(settings)?;
    let res = client.term_exec(command, session, timeout_sec).await?;
    format_json(&res)
}

pub async fn op_job_spawn(command: &str, cwd: &str, name: &str) -> Result<String> {
    let settings = load_settings()?;
    let client = WorkerClient::new(settings)?;
    let res = client.job_spawn(command, cwd, name).await?;
    format_json(&res)
}

pub async fn op_job_list() -> Result<String> {
    let settings = load_settings()?;
    let client = WorkerClient::new(settings)?;
    let res = client.job_list().await?;
    format_json(&res)
}

pub async fn op_job_logs(job_id: &str, tail_lines: usize, grep: &str) -> Result<String> {
    let settings = load_settings()?;
    let client = WorkerClient::new(settings)?;
    let res = client.job_logs(job_id, tail_lines, grep).await?;
    format_json(&res)
}

pub async fn op_job_kill(job_id: &str, signal: &str) -> Result<String> {
    let settings = load_settings()?;
    let client = WorkerClient::new(settings)?;
    let res = client.job_kill(job_id, signal).await?;
    format_json(&res)
}

pub async fn op_fs_sync(direction: &str, local_path: &str, remote_path: &str) -> Result<String> {
    let settings = load_settings()?;
    let client = WorkerClient::new(settings)?;
    match direction.trim().to_ascii_lowercase().as_str() {
        "push" => {
            let res = client.fs_push(Path::new(local_path), remote_path).await?;
            format_json(&res)
        }
        "pull" => {
            let res = client.fs_pull(remote_path, Path::new(local_path)).await?;
            format_json(&res)
        }
        _ => bail!("`direction` must be 'push' or 'pull'."),
    }
}

fn to_tool_result(res: Result<String>) -> Result<CallToolResult, McpError> {
    match res {
        Ok(text) => Ok(CallToolResult::success(vec![ContentBlock::text(text)])),
        Err(err) => Ok(CallToolResult::error(vec![ContentBlock::text(
            format!("{err:#}"),
        )])),
    }
}

// ---------------------------------------------------------------------------
// RMCP Server Definition
// ---------------------------------------------------------------------------

#[derive(Clone, Default)]
pub struct KaggleMcpServer;

#[tool_router]
impl KaggleMcpServer {
    pub fn new() -> Self {
        Self
    }

    #[tool(
        name = "kaggle_up",
        description = "Provision a headless Kaggle GPU worker (stays alive up to 11h40m safe drain). Requests Dual NVIDIA T4 x2 (2x15GB VRAM) by default; supports optional idle auto-shutdown, mounting Kaggle datasets at /kaggle/input, and chaining previous /kaggle/working output for >12h runs."
    )]
    async fn kaggle_up(
        &self,
        Parameters(params): Parameters<KaggleUpParams>,
    ) -> Result<CallToolResult, McpError> {
        to_tool_result(
            op_kaggle_up(
                params.gpu,
                params.idle_timeout_min,
                params.mount_datasets,
                params.chain_previous_output,
                params.wait_ready_sec,
            )
            .await,
        )
    }

    #[tool(
        name = "kaggle_status",
        description = "Inspect Kaggle kernel status, tunnel health, Dual-T4 GPU/VRAM utilization, RAM, disk, and remaining session time before safe drain."
    )]
    async fn kaggle_status(&self) -> Result<CallToolResult, McpError> {
        to_tool_result(op_kaggle_status().await)
    }

    #[tool(
        name = "kaggle_down",
        description = "Gracefully stop the remote Kaggle worker (exit 0) immediately to save weekly GPU quota; optionally pull committed /kaggle/working outputs."
    )]
    async fn kaggle_down(
        &self,
        Parameters(params): Parameters<KaggleDownParams>,
    ) -> Result<CallToolResult, McpError> {
        to_tool_result(op_kaggle_down(params.pull_outputs, &params.local_output_dir).await)
    }

    #[tool(
        name = "term_exec",
        description = "Execute a bash command inside a persistent stateful PTY on Kaggle (preserves cwd, env vars, CUDA, and virtualenvs). Use for quick inspection, setup, and short commands; use job_spawn for long training runs."
    )]
    async fn term_exec(
        &self,
        Parameters(params): Parameters<TermExecParams>,
    ) -> Result<CallToolResult, McpError> {
        to_tool_result(
            op_term_exec(&params.command, &params.session, params.timeout_sec).await,
        )
    }

    #[tool(
        name = "job_spawn",
        description = "Launch a detached long-running GPU/CPU background job immune to network drops, logging to /kaggle/working/mcp_logs."
    )]
    async fn job_spawn(
        &self,
        Parameters(params): Parameters<JobSpawnParams>,
    ) -> Result<CallToolResult, McpError> {
        to_tool_result(op_job_spawn(&params.command, &params.cwd, &params.name).await)
    }

    #[tool(
        name = "job_list",
        description = "List all running and finished background jobs with PID, runtime, exit code, and log size."
    )]
    async fn job_list(&self) -> Result<CallToolResult, McpError> {
        to_tool_result(op_job_list().await)
    }

    #[tool(
        name = "job_logs",
        description = "Read tail lines of a background job log with optional regex grep and automatic tqdm carriage-return compaction."
    )]
    async fn job_logs(
        &self,
        Parameters(params): Parameters<JobLogsParams>,
    ) -> Result<CallToolResult, McpError> {
        to_tool_result(op_job_logs(&params.job_id, params.tail_lines, &params.grep).await)
    }

    #[tool(
        name = "job_kill",
        description = "Send SIGTERM or SIGKILL to a background job's entire process group and release GPU VRAM."
    )]
    async fn job_kill(
        &self,
        Parameters(params): Parameters<JobKillParams>,
    ) -> Result<CallToolResult, McpError> {
        to_tool_result(op_job_kill(&params.job_id, &params.signal).await)
    }

    #[tool(
        name = "fs_sync",
        description = "Synchronize files or directories between the local machine and Kaggle over compressed .tar.gz streams. direction='push' packs local_path (auto-ignoring .git, .venv, __pycache__, node_modules) into remote_path; direction='pull' downloads remote_path and extracts into local_path."
    )]
    async fn fs_sync(
        &self,
        Parameters(params): Parameters<FsSyncParams>,
    ) -> Result<CallToolResult, McpError> {
        to_tool_result(
            op_fs_sync(&params.direction, &params.local_path, &params.remote_path).await,
        )
    }
}

#[tool_handler(
    name = "kaggle-shell-mcp",
    version = "0.1.0",
    instructions = "Control a headless Kaggle Dual-T4 GPU session (up to 12h) via stateful PTY shell and detached background jobs. Use `kaggle_up` to provision GPUs, `term_exec` for short stateful commands, `job_spawn` / `job_logs` for long training runs, `fs_sync` to push code or pull checkpoints, and `kaggle_down` as soon as work completes to preserve weekly GPU quota."
)]
impl ServerHandler for KaggleMcpServer {}

/// Serve the MCP server over standard input / standard output (`stdio`).
pub async fn run_stdio_server() -> Result<()> {
    let service = KaggleMcpServer::new()
        .serve(rmcp::transport::stdio())
        .await?;
    service.waiting().await?;
    Ok(())
}
