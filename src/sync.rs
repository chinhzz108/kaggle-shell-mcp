//! Fast filtered `.tar.gz` archive helpers for syncing local <-> Kaggle files
//! with strict path-traversal and symlink protection.

use std::fs;
use std::io::{Cursor, Read};
use std::path::Path;

use anyhow::{Context, Result, bail};
use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use tar::{Archive, Builder, EntryType};

use crate::utils::{expand_tilde, safe_join};

pub const IGNORE_DIRS: &[&str] = &[
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    ".kaggle_mcp",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
];

pub const IGNORE_SUFFIXES: &[&str] = &[".pyc", ".pyo"];

fn should_skip(path: &Path) -> bool {
    if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
        if IGNORE_DIRS.contains(&name) {
            return true;
        }
        if IGNORE_SUFFIXES.iter().any(|suffix| name.ends_with(suffix)) {
            return true;
        }
    }
    false
}

/// Pack a local file or directory into a compressed `.tar.gz` byte buffer.
///
/// For directories, preserves the top-level folder name so pushing `./my_proj`
/// to `/kaggle/working` extracts into `/kaggle/working/my_proj/...`, while
/// skipping VCS, virtualenv, and cache artifacts.
pub fn pack_tar_gz(local_path: &Path) -> Result<Vec<u8>> {
    let expanded = expand_tilde(local_path);
    let resolved = fs::canonicalize(&expanded)
        .with_context(|| format!("Local path does not exist: {}", expanded.display()))?;

    let encoder = GzEncoder::new(Vec::new(), Compression::default());
    let mut builder = Builder::new(encoder);

    let top_name = resolved
        .file_name()
        .and_then(|n| n.to_str())
        .with_context(|| format!("Cannot determine file name for {}", resolved.display()))?;

    if resolved.is_file() {
        builder
            .append_path_with_name(&resolved, top_name)
            .with_context(|| format!("Failed to pack file {}", resolved.display()))?;
    } else if resolved.is_dir() {
        append_dir_filtered(&mut builder, &resolved, top_name)?;
    } else {
        bail!("Unsupported local path type: {}", resolved.display());
    }

    let encoder = builder.into_inner().context("Failed to finalize tar stream")?;
    encoder.finish().context("Failed to finish gzip stream")
}

fn append_dir_filtered(
    builder: &mut Builder<GzEncoder<Vec<u8>>>,
    dir: &Path,
    archive_prefix: &str,
) -> Result<()> {
    builder
        .append_dir(archive_prefix, dir)
        .with_context(|| format!("Failed to append directory entry '{archive_prefix}'"))?;

    let mut entries: Vec<_> = fs::read_dir(dir)
        .with_context(|| format!("Failed to read directory {}", dir.display()))?
        .collect::<std::io::Result<Vec<_>>>()
        .with_context(|| format!("Failed to iterate directory {}", dir.display()))?;

    entries.sort_by_key(|e| e.file_name());

    for entry in entries {
        let path = entry.path();
        if should_skip(&path) {
            continue;
        }
        let file_name = entry.file_name();
        let name_str = file_name.to_string_lossy();
        // Always emit POSIX forward-slash archive paths so Linux Kaggle unpacks properly
        let entry_archive_path = format!("{archive_prefix}/{name_str}");

        let meta = fs::symlink_metadata(&path)
            .with_context(|| format!("Failed to stat {}", path.display()))?;

        if meta.file_type().is_symlink() {
            continue;
        }
        if meta.is_dir() {
            append_dir_filtered(builder, &path, &entry_archive_path)?;
        } else if meta.is_file() {
            builder
                .append_path_with_name(&path, &entry_archive_path)
                .with_context(|| format!("Failed to pack file {}", path.display()))?;
        }
    }

    Ok(())
}

/// Safely extract `.tar.gz` bytes into `dest_dir` and return extracted relative names.
///
/// Rejects path traversal (`..`, absolute paths, Windows drive prefixes) and skips symlinks/hardlinks.
pub fn unpack_tar_gz(tar_bytes: &[u8], dest_dir: &Path) -> Result<Vec<String>> {
    let expanded = expand_tilde(dest_dir);
    fs::create_dir_all(&expanded)
        .with_context(|| format!("Failed to create destination directory {}", expanded.display()))?;
    let canonical_dest = fs::canonicalize(&expanded)
        .with_context(|| format!("Failed to resolve destination {}", expanded.display()))?;

    let decoder = GzDecoder::new(Cursor::new(tar_bytes));
    let mut archive = Archive::new(decoder);
    let mut extracted = Vec::new();

    for entry_res in archive.entries().context("Failed to read tar entries")? {
        let mut entry = entry_res.context("Failed to read tar entry")?;
        let entry_type = entry.header().entry_type();

        // Read raw header path bytes directly because `tar::Entry::path()` rejects `..`
        // or normalizes inconsistently; we want explicit path-traversal rejection.
        let raw_path_bytes = entry.path_bytes();
        let raw_name = String::from_utf8_lossy(&raw_path_bytes).into_owned();
        let posix_name = raw_name.replace('\\', "/");

        // Validate safety BEFORE skipping or extracting anything
        let target = safe_join(&canonical_dest, &posix_name)?;

        if matches!(entry_type, EntryType::Symlink | EntryType::Link) {
            continue;
        }

        if entry_type.is_dir() {
            fs::create_dir_all(&target)
                .with_context(|| format!("Failed to create directory {}", target.display()))?;
            extracted.push(posix_name);
        } else if entry_type.is_file() {
            if let Some(parent) = target.parent() {
                fs::create_dir_all(parent).with_context(|| {
                    format!("Failed to create parent directory {}", parent.display())
                })?;
            }
            let mut buf = Vec::new();
            entry
                .read_to_end(&mut buf)
                .with_context(|| format!("Failed to read archive entry '{posix_name}'"))?;
            fs::write(&target, &buf)
                .with_context(|| format!("Failed to write extracted file {}", target.display()))?;
            extracted.push(posix_name);
        }
    }

    Ok(extracted)
}
