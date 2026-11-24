//! `kgsh` CLI & stdio MCP server entrypoint.
//!
//! Invoking `kgsh` with no arguments starts the MCP `stdio` server.
//! Subcommands provide direct terminal control over the remote Kaggle Dual-T4 worker.

use std::ffi::OsString;

use anyhow::Result;
use clap::{Parser, Subcommand};

use kaggle_shell_mcp::config::DEFAULT_IDLE_TIMEOUT_MIN;
use kaggle_shell_mcp::mcp_server::{
    op_fs_sync, op_job_kill, op_job_list, op_job_logs, op_job_spawn, op_kaggle_down,
    op_kaggle_status, op_kaggle_up, op_term_exec, run_stdio_server,
};

#[derive(Debug, Parser)]
#[command(
    name = "kgsh",
    version = env!("CARGO_PKG_VERSION"),
    about = "kgsh (kaggle-shell-mcp) — Headless 12h Kaggle Dual-T4 GPU Worker & MCP Server (Rust)"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Run stdio MCP server explicitly
    Mcp,
    /// Boot headless Kaggle GPU worker
    Up {
        /// Boot CPU-only instance
        #[arg(long)]
        no_gpu: bool,
        /// Idle auto-shutdown minutes (default: 10; set 0 to disable)
        #[arg(long, default_value_t = DEFAULT_IDLE_TIMEOUT_MIN)]
        idle: u64,
        /// Max seconds to wait for tunnel readiness
        #[arg(long, default_value_t = 240)]
        wait: u64,
        /// Mount Kaggle dataset owner/slug (repeatable)
        #[arg(long = "dataset")]
        datasets: Vec<String>,
        /// Mount previous kernel output (/kaggle/working) to chain >12h runs
        #[arg(long)]
        chain: bool,
    },
    /// Show kernel status and live GPU telemetry
    Status,
    /// Gracefully shut down active Kaggle worker
    Down {
        /// Download committed /kaggle/working outputs
        #[arg(long)]
        pull: bool,
        /// Local output folder when --pull is set
        #[arg(long, default_value = "./kaggle_outputs")]
        out: String,
    },
    /// Execute command in persistent remote PTY
    Exec {
        /// Bash command string
        command: String,
        /// PTY session name
        #[arg(long, default_value = "default")]
        session: String,
        /// Timeout in seconds
        #[arg(long, default_value_t = 60.0)]
        timeout: f64,
    },
    /// Start detached background job
    Spawn {
        /// Background command to run
        command: String,
        /// Remote working directory
        #[arg(long, default_value = "/kaggle/working")]
        cwd: String,
        /// Optional friendly job name
        #[arg(long, default_value = "")]
        name: String,
    },
    /// List background jobs
    Jobs,
    /// Tail logs of a background job
    Logs {
        /// Job identifier
        job_id: String,
        /// Number of tail lines
        #[arg(long, default_value_t = 100)]
        tail: usize,
        /// Optional regex filter
        #[arg(long, default_value = "")]
        grep: String,
    },
    /// Stop a running background job
    Kill {
        /// Job identifier
        job_id: String,
        /// Signal name (SIGTERM or SIGKILL)
        #[arg(long, default_value = "SIGTERM")]
        signal: String,
    },
    /// Upload local file/directory to Kaggle
    Push {
        /// Local source path
        local_path: String,
        /// Remote destination directory
        #[arg(default_value = "/kaggle/working")]
        remote_path: String,
    },
    /// Download remote file/directory from Kaggle
    Pull {
        /// Remote source file or directory
        remote_path: String,
        /// Local destination directory
        #[arg(default_value = "./kaggle_pull")]
        local_path: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Always route tracing logs to stderr so MCP JSON-RPC stdout stays pristine
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    let mut raw_args: Vec<OsString> = std::env::args_os().collect();
    if raw_args.len() <= 1 {
        // Default zero-arg invocation runs the stdio MCP server
        return run_stdio_server().await;
    }

    // Preserve compatibility with legacy `--cli <subcommand>` wrapper syntax
    if raw_args.get(1).and_then(|s| s.to_str()) == Some("--cli") {
        raw_args.remove(1);
        if raw_args.len() <= 1 {
            return run_stdio_server().await;
        }
    }

    let cli = Cli::parse_from(raw_args);
    let output = match cli.command {
        Commands::Mcp => return run_stdio_server().await,
        Commands::Up {
            no_gpu,
            idle,
            wait,
            datasets,
            chain,
        } => {
            let ds = if datasets.is_empty() {
                None
            } else {
                Some(datasets)
            };
            op_kaggle_up(!no_gpu, idle, ds, chain, wait).await?
        }
        Commands::Status => op_kaggle_status().await?,
        Commands::Down { pull, out } => op_kaggle_down(pull, &out).await?,
        Commands::Exec {
            command,
            session,
            timeout,
        } => op_term_exec(&command, &session, timeout).await?,
        Commands::Spawn { command, cwd, name } => op_job_spawn(&command, &cwd, &name).await?,
        Commands::Jobs => op_job_list().await?,
        Commands::Logs { job_id, tail, grep } => op_job_logs(&job_id, tail, &grep).await?,
        Commands::Kill { job_id, signal } => op_job_kill(&job_id, &signal).await?,
        Commands::Push {
            local_path,
            remote_path,
        } => op_fs_sync("push", &local_path, &remote_path).await?,
        Commands::Pull {
            remote_path,
            local_path,
        } => op_fs_sync("pull", &local_path, &remote_path).await?,
    };

    println!("{output}");
    Ok(())
}
