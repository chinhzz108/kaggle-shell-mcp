# Changelog

All notable changes to **`kaggle-shell-mcp`** will be documented in this file.

## [0.1.0] - 2026-11-24 (Rust Rewrite)

### Changed
- **Full Rust Rewrite (`kgsh`)**: Reimplemented the local MCP server, CLI, Kaggle API client, worker RPC client, and filtered `.tar.gz` sync engine in pure Rust (Edition 2024) with zero local Python dependencies.
- **Official Rust MCP SDK (`rmcp`)**: Replaced Python `FastMCP` with `rmcp` over `stdio` exposing all 9 tools with JSON Schema 2020-12 definitions via `schemars`.
- **Direct Kaggle API Client**: Implemented async `reqwest` client targeting `https://api.kaggle.com/v1/...` supporting both `KGAT_` Bearer tokens and legacy hex keys, resumable GCS blob upload (`StartBlobUpload` -> `PUT` -> `CreateDataset`), private dataset handshake ZIP extraction, and kernel output pulling.
- **Single Self-Contained Binary**: Embedded `src/remote/worker_template.py` at compile time via `include_str!` so `kgsh` runs from any directory on Windows, Linux, and macOS.

### Retained from Upstream (`NguyenQuocDat06-AI/kaggle-shell-mcp`)
- **Headless 12h Kaggle Dual-T4 Provisioner (`kaggle_up`, `kaggle_status`, `kaggle_down`)**: Boots background Kaggle GPU kernels (`NvidiaTeslaT4`, 2×15 GB VRAM) via Kaggle API with zero browser interaction.
- **Zero-Setup Tunnel & Private Dataset Handshake**: Automatic `cloudflared` Quick Tunnel supervisor paired with private Kaggle Dataset state sync (`<user>/kaggle-mcp-state`), plus optional Tailscale userspace networking (`TS_AUTHKEY`).
- **Stateful Remote PTY (`term_exec`)**: Persistent `/bin/bash` PTY preserving `cwd`, shell exports, Conda/venv environments, and native CUDA paths.
- **Detached Background Job Engine (`job_spawn`, `job_list`, `job_logs`, `job_kill`)**: Process-group isolated training jobs with persistent logs in `/kaggle/working/mcp_logs` and automatic `tqdm` carriage-return (`\r`) compaction.
- **Filtered Tar Streaming (`fs_sync`)**: Fast bidirectional directory/file sync ignoring `.git`, `.venv`, `__pycache__`, and `node_modules`.
- **Snapshot & Optional Idle Watchdogs**: Graceful `SIGTERM` checkpoint flush at `11h35m` + clean `exit(0)` at `11h40m`, plus opt-in idle GPU auto-shutdown (`idle_timeout_min`, disabled by default).
