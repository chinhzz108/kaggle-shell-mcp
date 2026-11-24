//! Shared utility functions: tqdm output compaction, Kaggle status normalization,
//! home path expansion, and path-traversal protection.

use std::path::{Component, Path, PathBuf};
use std::sync::LazyLock;

use anyhow::{Context, Result, bail};
use regex::Regex;

static ANSI_CURSOR_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\x1b\[[0-9;?]*[ABCDEFGHJKSTfhlm]").expect("valid ANSI regex")
});

/// Collapse carriage-return progress bar updates (`tqdm`) and strip ANSI cursor escapes
/// so agent context stays compact.
pub fn clean_tqdm(text: &str) -> String {
    let normalized = text.replace("\r\n", "\n");
    normalized
        .split('\n')
        .map(|raw_line| {
            let line = if raw_line.contains('\r') {
                raw_line
                    .split('\r')
                    .filter(|seg| !seg.is_empty())
                    .next_back()
                    .unwrap_or("")
            } else {
                raw_line
            };
            ANSI_CURSOR_RE.replace_all(line, "").into_owned()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Normalize Kaggle kernel status strings across enum variants and casing styles
/// (e.g. `"running"`, `"KernelWorkerStatus.RUNNING"`, `"KERNEL_WORKER_STATUS_RUNNING"`,
/// `"cancelRequested"`) into canonical `UPPER_SNAKE_CASE`.
pub fn normalize_kernel_status(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return "UNKNOWN".to_string();
    }
    let last_segment = trimmed.rsplit('.').next().unwrap_or(trimmed);
    let stripped = last_segment
        .strip_prefix("KERNEL_WORKER_STATUS_")
        .or_else(|| last_segment.strip_prefix("kernelWorkerStatus"))
        .unwrap_or(last_segment);

    if stripped.contains('_') || stripped.chars().all(|c| !c.is_ascii_lowercase()) {
        return stripped.to_ascii_uppercase();
    }

    // Convert camelCase (e.g. "cancelRequested") to UPPER_SNAKE_CASE ("CANCEL_REQUESTED")
    let mut out = String::with_capacity(stripped.len() + 4);
    for (i, ch) in stripped.chars().enumerate() {
        if ch.is_ascii_uppercase() && i > 0 {
            out.push('_');
        }
        out.push(ch.to_ascii_uppercase());
    }
    out
}

/// Expand leading `~` or `~/` (and `~\` on Windows) using `HOME` or `USERPROFILE`.
pub fn expand_tilde(path: &Path) -> PathBuf {
    let s = path.to_string_lossy();
    if s == "~" {
        if let Some(home) = home_dir() {
            return home;
        }
    } else if let Some(rest) = s.strip_prefix("~/").or_else(|| s.strip_prefix("~\\"))
        && let Some(home) = home_dir()
    {
        return home.join(rest);
    }
    path.to_path_buf()
}

/// Resolve the current user's home directory cross-platform.
pub fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

/// Resolve a relative archive/download path safely within `base_dir`, blocking
/// absolute paths, Windows drive prefixes, and `..` directory traversal.
pub fn safe_join(base_dir: &Path, rel: &str) -> Result<PathBuf> {
    // Normalize Windows backslashes so cross-platform archives cannot sneak `..\` through on Unix
    let normalized = rel.replace('\\', "/");
    let rel_path = Path::new(&normalized);

    let mut clean_rel = PathBuf::new();
    for comp in rel_path.components() {
        match comp {
            Component::Normal(part) => clean_rel.push(part),
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                bail!("Blocked path traversal in path: {rel}");
            }
        }
    }

    if clean_rel.as_os_str().is_empty() {
        return Ok(base_dir.to_path_buf());
    }

    let joined = base_dir.join(&clean_rel);
    if !joined.starts_with(base_dir) {
        bail!("Blocked path traversal outside destination: {rel}");
    }
    Ok(joined)
}

/// Split an `"owner/slug"` Kaggle reference string into `(owner, slug)`.
pub fn split_kaggle_ref(reference: &str) -> Result<(String, String)> {
    let mut parts = reference.trim().split('/');
    let owner = parts
        .next()
        .filter(|s| !s.is_empty())
        .with_context(|| format!("Invalid Kaggle reference '{reference}': expected 'owner/slug'"))?;
    let slug = parts
        .next()
        .filter(|s| !s.is_empty())
        .with_context(|| format!("Invalid Kaggle reference '{reference}': expected 'owner/slug'"))?;
    Ok((owner.to_string(), slug.to_string()))
}
