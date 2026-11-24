//! Configuration, Kaggle authentication setup, and local session state persistence.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::utils::home_dir;

/// 11h 40m safe exit before Kaggle's 12h hard reaper.
pub const MAX_LIFETIME_SEC: u64 = 42_000;
/// 11h 35m SIGTERM checkpoint drain window.
pub const DRAIN_AT_SEC: u64 = 41_700;
/// 0 = disabled by default (opt-in via `idle_timeout_min` / `--idle`).
pub const DEFAULT_IDLE_TIMEOUT_MIN: u64 = 0;
pub const DEFAULT_KERNEL_SLUG: &str = "kaggle-mcp-worker";
pub const DEFAULT_STATE_DATASET_SLUG: &str = "kaggle-mcp-state";
pub const DEFAULT_TS_HOSTNAME: &str = "kaggle-mcp-worker";
pub const DEFAULT_KAGGLE_API_BASE: &str = "https://api.kaggle.com";

/// Authentication scheme supported by Kaggle's API.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KaggleAuth {
    /// Modern `KGAT_...` personal access token sent as `Authorization: Bearer <token>`.
    Bearer(String),
    /// Legacy username + hex API key sent as HTTP Basic auth.
    Basic { username: String, key: String },
}

impl KaggleAuth {
    /// Build authentication config, distinguishing `KGAT_` bearer tokens from legacy keys.
    pub fn from_credentials(username: &str, secret: &str) -> Self {
        if secret.starts_with("KGAT_") {
            Self::Bearer(secret.to_string())
        } else {
            Self::Basic {
                username: username.to_string(),
                key: secret.to_string(),
            }
        }
    }
}

/// Application settings loaded from environment / `.env`.
#[derive(Debug, Clone)]
pub struct Settings {
    pub kaggle_username: String,
    pub kaggle_secret: String,
    pub ts_authkey: String,
    pub ts_hostname: String,
    pub kernel_slug: String,
    pub state_dataset_slug: String,
    pub kaggle_api_base: String,
    pub state_dir: PathBuf,
}

impl Settings {
    pub fn new(kaggle_username: impl Into<String>, kaggle_secret: impl Into<String>) -> Self {
        Self {
            kaggle_username: kaggle_username.into(),
            kaggle_secret: kaggle_secret.into(),
            ts_authkey: String::new(),
            ts_hostname: DEFAULT_TS_HOSTNAME.to_string(),
            kernel_slug: DEFAULT_KERNEL_SLUG.to_string(),
            state_dataset_slug: DEFAULT_STATE_DATASET_SLUG.to_string(),
            kaggle_api_base: DEFAULT_KAGGLE_API_BASE.to_string(),
            state_dir: resolve_state_dir(),
        }
    }

    pub fn with_state_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.state_dir = dir.into();
        self
    }

    pub fn with_api_base(mut self, base: impl Into<String>) -> Self {
        self.kaggle_api_base = base.into().trim_end_matches('/').to_string();
        self
    }

    pub fn kernel_ref(&self) -> String {
        format!("{}/{}", self.kaggle_username, self.kernel_slug)
    }

    pub fn state_dataset_ref(&self) -> String {
        format!("{}/{}", self.kaggle_username, self.state_dataset_slug)
    }

    pub fn auth(&self) -> KaggleAuth {
        KaggleAuth::from_credentials(&self.kaggle_username, &self.kaggle_secret)
    }

    pub fn active_session_file(&self) -> PathBuf {
        self.state_dir.join("active_session.json")
    }

    pub fn build_bundle_dir(&self) -> PathBuf {
        self.state_dir.join("build_bundle")
    }
}

/// Persisted active worker session metadata stored in `.kaggle_mcp/active_session.json`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ActiveSession {
    pub session_id: String,
    pub worker_token: String,
    pub kernel_ref: String,
    pub state_dataset_ref: String,
    #[serde(default)]
    pub tunnel_url: String,
    #[serde(default)]
    pub ts_url: String,
    #[serde(default)]
    pub created_at: f64,
    #[serde(default)]
    pub idle_timeout_min: u64,
}

impl ActiveSession {
    /// Preferred base URL (`ts_url` over `tunnel_url`), without trailing slash.
    pub fn base_url(&self) -> String {
        let raw = if !self.ts_url.is_empty() {
            &self.ts_url
        } else {
            &self.tunnel_url
        };
        raw.trim_end_matches('/').to_string()
    }
}

/// Detect repository root when running from `<repo>/target/{debug,release}/kgsh`.
fn detect_repo_root() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let exe_dir = exe.parent()?;
    // target/debug or target/release -> parent is target -> parent is repo root
    let target_dir = exe_dir.parent()?;
    if target_dir.file_name().and_then(|n| n.to_str()) == Some("target") {
        let repo_root = target_dir.parent()?;
        if repo_root.join("Cargo.toml").is_file() {
            return Some(repo_root.to_path_buf());
        }
    }
    // target/<triple>/release -> grandparent is target
    if let Some(grandparent) = target_dir.parent()
        && grandparent.file_name().and_then(|n| n.to_str()) == Some("target")
        && let Some(repo_root) = grandparent.parent()
        && repo_root.join("Cargo.toml").is_file()
    {
        return Some(repo_root.to_path_buf());
    }
    None
}

/// Resolve the local state directory for `.kaggle_mcp` session persistence.
pub fn resolve_state_dir() -> PathBuf {
    if let Ok(explicit) = std::env::var("KAGGLE_MCP_STATE_DIR") {
        let trimmed = explicit.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }

    if let Ok(cwd) = std::env::current_dir()
        && (cwd.join(".kaggle_mcp").is_dir() || cwd.join(".env").is_file())
    {
        return cwd.join(".kaggle_mcp");
    }

    if let Some(repo_root) = detect_repo_root() {
        return repo_root.join(".kaggle_mcp");
    }

    if let Some(home) = home_dir() {
        return home.join(".kaggle_mcp");
    }

    PathBuf::from(".kaggle_mcp")
}

/// Discover and load `.env` files (without overriding existing process env vars).
fn load_dotenv_candidates() -> Vec<PathBuf> {
    let mut candidates = Vec::new();

    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd.join(".env"));
    }
    if let Some(repo_root) = detect_repo_root() {
        let p = repo_root.join(".env");
        if !candidates.contains(&p) {
            candidates.push(p);
        }
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(exe_dir) = exe.parent()
    {
        let p = exe_dir.join(".env");
        if !candidates.contains(&p) {
            candidates.push(p);
        }
    }
    if let Some(home) = home_dir() {
        let p = home.join(".kaggle_mcp").join(".env");
        if !candidates.contains(&p) {
            candidates.push(p);
        }
    }

    for env_path in &candidates {
        if env_path.is_file() {
            let _ = dotenvy::from_path(env_path);
        }
    }

    candidates
}

fn env_trimmed(name: &str) -> String {
    std::env::var(name)
        .unwrap_or_default()
        .trim()
        .to_string()
}

fn read_kaggle_json_fallback() -> Option<(String, String)> {
    let home = home_dir()?;
    let path = home.join(".kaggle").join("kaggle.json");
    let text = fs::read_to_string(path).ok()?;
    let v: serde_json::Value = serde_json::from_str(&text).ok()?;
    let user = v.get("username")?.as_str()?.trim().to_string();
    let key = v.get("key")?.as_str()?.trim().to_string();
    if user.is_empty() || key.is_empty() {
        return None;
    }
    Some((user, key))
}

fn read_access_token_fallback() -> Option<String> {
    let home = home_dir()?;
    for name in ["access_token", "access_token.txt"] {
        let path = home.join(".kaggle").join(name);
        if let Ok(text) = fs::read_to_string(path) {
            let trimmed = text.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }
    None
}

/// Load credentials from environment / `.env` (with `~/.kaggle` fallback) and construct `Settings`.
pub fn load_settings() -> Result<Settings> {
    let checked = load_dotenv_candidates();

    let mut username = env_trimmed("KAGGLE_USERNAME");
    let mut secret = env_trimmed("KAGGLE_API_TOKEN");
    if secret.is_empty() {
        secret = env_trimmed("KAGGLE_KEY");
    }

    if secret.is_empty()
        && let Some(tok) = read_access_token_fallback()
    {
        secret = tok;
    }

    if (username.is_empty() || secret.is_empty())
        && let Some((u, k)) = read_kaggle_json_fallback()
    {
        if username.is_empty() {
            username = u;
        }
        if secret.is_empty() {
            secret = k;
        }
    }

    if username.is_empty() || secret.is_empty() {
        let checked_list = checked
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        bail!(
            "Missing KAGGLE_USERNAME or KAGGLE_KEY / KAGGLE_API_TOKEN in environment or .env (checked: {checked_list})."
        );
    }

    let ts_authkey = env_trimmed("TS_AUTHKEY");
    let ts_hostname = {
        let v = env_trimmed("TS_HOSTNAME");
        if v.is_empty() {
            DEFAULT_TS_HOSTNAME.to_string()
        } else {
            v
        }
    };
    let kernel_slug = {
        let v = env_trimmed("KAGGLE_MCP_KERNEL_SLUG");
        if v.is_empty() {
            DEFAULT_KERNEL_SLUG.to_string()
        } else {
            v
        }
    };
    let state_dataset_slug = {
        let v = env_trimmed("KAGGLE_MCP_STATE_DATASET");
        if v.is_empty() {
            DEFAULT_STATE_DATASET_SLUG.to_string()
        } else {
            v
        }
    };
    let kaggle_api_base = {
        let v = env_trimmed("KAGGLE_API_BASE");
        if v.is_empty() {
            DEFAULT_KAGGLE_API_BASE.to_string()
        } else {
            v.trim_end_matches('/').to_string()
        }
    };

    Ok(Settings {
        kaggle_username: username,
        kaggle_secret: secret,
        ts_authkey,
        ts_hostname,
        kernel_slug,
        state_dataset_slug,
        kaggle_api_base,
        state_dir: resolve_state_dir(),
    })
}

pub fn load_active_session(state_dir: &Path) -> Option<ActiveSession> {
    let path = state_dir.join("active_session.json");
    let raw = fs::read_to_string(path).ok()?;
    serde_json::from_str(&raw).ok()
}

pub fn save_active_session(state_dir: &Path, session: &ActiveSession) -> Result<()> {
    fs::create_dir_all(state_dir)
        .with_context(|| format!("Failed to create state directory {}", state_dir.display()))?;
    let target = state_dir.join("active_session.json");
    let tmp = state_dir.join("active_session.tmp");
    let json = serde_json::to_string_pretty(session)?;
    fs::write(&tmp, json).with_context(|| format!("Failed to write {}", tmp.display()))?;
    // On Windows, fs::rename fails if target exists in some older environments,
    // though std::fs::rename uses MoveFileExW(MOVEFILE_REPLACE_EXISTING).
    if let Err(err) = fs::rename(&tmp, &target) {
        let _ = fs::remove_file(&target);
        fs::rename(&tmp, &target).with_context(|| {
            format!(
                "Failed to atomically persist {} (initial error: {err})",
                target.display()
            )
        })?;
    }
    Ok(())
}

pub fn clear_active_session(state_dir: &Path) {
    let target = state_dir.join("active_session.json");
    let _ = fs::remove_file(target);
}
