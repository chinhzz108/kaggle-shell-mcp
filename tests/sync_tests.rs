//! Tests for filtered `.tar.gz` directory/file packing and path-traversal protection.

use std::fs;
use std::io::Write;

use flate2::Compression;
use flate2::write::GzEncoder;
use kaggle_shell_mcp::sync::{pack_tar_gz, unpack_tar_gz};
use tempfile::tempdir;

#[test]
fn directory_sync_preserves_top_folder_and_skips_ignored() {
    let tmp = tempdir().unwrap();
    let proj = tmp.path().join("src_tree").join("my_proj");
    fs::create_dir_all(proj.join("models")).unwrap();
    fs::write(proj.join("train.py"), "print('ok')\n").unwrap();
    fs::write(proj.join("models").join("net.py"), "x = 1\n").unwrap();

    // Ignored artifacts
    fs::create_dir_all(proj.join("__pycache__")).unwrap();
    fs::write(
        proj.join("__pycache__").join("train.cpython-312.pyc"),
        [0u8, 1],
    )
    .unwrap();
    fs::create_dir_all(proj.join(".git")).unwrap();
    fs::write(proj.join(".git").join("HEAD"), "ref: refs/heads/main\n").unwrap();
    fs::create_dir_all(proj.join(".venv")).unwrap();
    fs::write(proj.join(".venv").join("pyvenv.cfg"), "home = /usr/bin\n").unwrap();
    fs::write(proj.join("stray.pyc"), [0u8, 2]).unwrap();

    let archive = pack_tar_gz(&proj).unwrap();
    let dest = tmp.path().join("remote_working");
    let extracted = unpack_tar_gz(&archive, &dest).unwrap();

    assert!(dest.join("my_proj").join("train.py").is_file());
    assert!(dest.join("my_proj").join("models").join("net.py").is_file());
    assert!(!dest.join("my_proj").join("__pycache__").exists());
    assert!(!dest.join("my_proj").join(".git").exists());
    assert!(!dest.join("my_proj").join(".venv").exists());
    assert!(!dest.join("my_proj").join("stray.pyc").exists());
    assert!(extracted.iter().any(|p| p == "my_proj/train.py"));
    assert!(extracted.iter().any(|p| p == "my_proj/models/net.py"));
}

#[test]
fn single_file_sync() {
    let tmp = tempdir().unwrap();
    let f = tmp.path().join("weights.bin");
    fs::write(&f, b"1234567890").unwrap();

    let dest = tmp.path().join("out");
    let archive = pack_tar_gz(&f).unwrap();
    unpack_tar_gz(&archive, &dest).unwrap();

    assert_eq!(fs::read(dest.join("weights.bin")).unwrap(), b"1234567890");
}

/// Craft a raw POSIX ustar tar block with an arbitrary (potentially malicious) path name
/// because `tar::Builder` refuses `..` when writing.
fn make_malicious_tar_gz(entry_name: &str, payload: &[u8]) -> Vec<u8> {
    let mut header = [0u8; 512];
    let name_bytes = entry_name.as_bytes();
    header[..name_bytes.len()].copy_from_slice(name_bytes);
    // mode 0644
    header[100..107].copy_from_slice(b"000644 ");
    // uid & gid
    header[108..115].copy_from_slice(b"000000 ");
    header[116..123].copy_from_slice(b"000000 ");
    // octal file size
    let size_oct = format!("{:011o}", payload.len());
    header[124..135].copy_from_slice(size_oct.as_bytes());
    // mtime
    header[136..147].copy_from_slice(b"00000000000");
    // checksum placeholder (spaces)
    header[148..156].fill(b' ');
    // typeflag = '0' (regular file)
    header[156] = b'0';
    // ustar magic + version
    header[257..263].copy_from_slice(b"ustar\0");
    header[263..265].copy_from_slice(b"00");

    // compute checksum
    let chksum: u32 = header.iter().map(|&b| b as u32).sum();
    let chk_str = format!("{chksum:06o}\0 ");
    header[148..156].copy_from_slice(chk_str.as_bytes());

    let mut raw_tar = Vec::new();
    raw_tar.extend_from_slice(&header);
    raw_tar.extend_from_slice(payload);
    let pad = (512 - (payload.len() % 512)) % 512;
    raw_tar.resize(raw_tar.len() + pad, 0);
    // EOF: two 512-byte zero blocks
    raw_tar.resize(raw_tar.len() + 1024, 0);

    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&raw_tar).unwrap();
    encoder.finish().unwrap()
}

#[test]
fn unpack_blocks_path_traversal() {
    let tmp = tempdir().unwrap();
    let malicious = make_malicious_tar_gz("../escape.txt", b"bad");
    let sandbox = tmp.path().join("sandbox");

    let res = unpack_tar_gz(&malicious, &sandbox);
    assert!(res.is_err(), "expected path traversal to be rejected");
    assert!(!tmp.path().join("escape.txt").exists());
}
