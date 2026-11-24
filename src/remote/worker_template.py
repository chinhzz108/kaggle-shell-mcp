#!/usr/bin/env python3
"""Self-contained Kaggle GPU worker: PTY shell, detached jobs, tunnel & watchdogs.

Runs directly inside the Kaggle Kernel Python runtime so CUDA/Conda environment
variables are preserved 100% without needing openssh-server or root apt packages.
"""

from __future__ import annotations

import base64
import fcntl
import hmac
import io
import json
import os
import pty
import re
import select
import shutil
import signal
import struct
import subprocess
import sys
import tarfile
import termios
import threading
import time
import urllib.request
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BOOTSTRAP_CONFIG_B64 = "__BOOTSTRAP_CONFIG_B64__"

CFG = json.loads(base64.b64decode(BOOTSTRAP_CONFIG_B64.encode("ascii")).decode("utf-8"))
SESSION_ID: str = CFG["session_id"]
WORKER_TOKEN: str = CFG["worker_token"]
KAGGLE_USERNAME: str = CFG["kaggle_username"]
KAGGLE_SECRET: str = CFG["kaggle_secret"]
STATE_DATASET_REF: str = CFG["state_dataset_ref"]
TS_AUTHKEY: str = CFG.get("ts_authkey", "")
TS_HOSTNAME: str = CFG.get("ts_hostname", "kaggle-mcp-worker")
IDLE_TIMEOUT_SEC: int = int(CFG.get("idle_timeout_sec", 0))
MAX_LIFETIME_SEC: int = int(CFG.get("max_lifetime_sec", 42000))  # 11h40m
DRAIN_AT_SEC: int = int(CFG.get("drain_at_sec", 41700))          # 11h35m
RPC_PORT: int = int(CFG.get("rpc_port", 8765))

WORKING_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
LOG_DIR = WORKING_DIR / "mcp_logs"
TMP_DIR = Path("/tmp/kaggle_mcp_runtime")
LOG_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)

START_MONO = time.monotonic()
START_WALL = time.time()
SHUTDOWN_EVENT = threading.Event()

ANSI_CURSOR_RE = re.compile(r"\x1b\[[0-9;?]*[ABCDEFGHJKSTfhlm]")
TRYCF_URL_RE = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")


def fixup_cuda_env() -> None:
    extra_bins = ["/opt/bin", "/usr/local/nvidia/bin", "/usr/local/cuda/bin", "/opt/conda/bin"]
    cur_path = os.environ.get("PATH", "")
    parts = [p for p in cur_path.split(":") if p]
    for b in extra_bins:
        if os.path.isdir(b) and b not in parts:
            parts.insert(0, b)
    os.environ["PATH"] = ":".join(parts)

    extra_libs = ["/usr/local/nvidia/lib64", "/usr/local/cuda/lib64"]
    cur_ld = os.environ.get("LD_LIBRARY_PATH", "")
    ld_parts = [p for p in cur_ld.split(":") if p]
    for lib in extra_libs:
        if os.path.isdir(lib) and lib not in ld_parts:
            ld_parts.insert(0, lib)
    os.environ["LD_LIBRARY_PATH"] = ":".join(ld_parts)
    os.environ["PYTHONUNBUFFERED"] = "1"


def clean_tqdm(text: str) -> str:
    """Collapse carriage-return progress bar updates so Agent context stays compact."""
    text = text.replace("\r\n", "\n")
    cleaned_lines: list[str] = []
    for raw_line in text.split("\n"):
        if "\r" in raw_line:
            segments = [s for s in raw_line.split("\r") if s != ""]
            raw_line = segments[-1] if segments else ""
        cleaned_lines.append(ANSI_CURSOR_RE.sub("", raw_line))
    return "\n".join(cleaned_lines)


def query_gpus() -> list[dict]:
    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res.returncode != 0:
            return []
        gpus = []
        for line in res.stdout.strip().splitlines():
            cols = [c.strip() for c in line.split(",")]
            if len(cols) >= 6:
                gpus.append(
                    {
                        "index": int(cols[0]),
                        "name": cols[1],
                        "util_pct": float(cols[2]) if cols[2].replace(".", "").isdigit() else 0.0,
                        "vram_used_mb": float(cols[3]) if cols[3].replace(".", "").isdigit() else 0.0,
                        "vram_total_mb": float(cols[4]) if cols[4].replace(".", "").isdigit() else 0.0,
                        "temp_c": float(cols[5]) if cols[5].replace(".", "").isdigit() else 0.0,
                    }
                )
        return gpus
    except Exception:
        return []


def query_ram_gb() -> dict:
    info: dict[str, float] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith(("MemTotal:", "MemAvailable:")):
                k, v = line.split(":", 1)
                kb = float(v.strip().split()[0])
                info[k] = round(kb / (1024 * 1024), 2)
        total = info.get("MemTotal", 0.0)
        avail = info.get("MemAvailable", 0.0)
        return {"used_gb": round(max(0.0, total - avail), 2), "total_gb": total}
    except Exception:
        return {"used_gb": 0.0, "total_gb": 0.0}


def query_disk_gb(path: Path) -> dict:
    try:
        u = shutil.disk_usage(path)
        gb = 1024 ** 3
        return {
            "used_gb": round(u.used / gb, 2),
            "free_gb": round(u.free / gb, 2),
            "total_gb": round(u.total / gb, 2),
        }
    except Exception:
        return {"used_gb": 0.0, "free_gb": 0.0, "total_gb": 0.0}


class PtySession:
    """Persistent bash PTY preserving cwd, shell variables, and virtualenvs."""

    def __init__(self, name: str, cwd: Path = WORKING_DIR) -> None:
        self.name = name
        self.cwd = str(cwd)
        self.lock = threading.Lock()
        self.master_fd: int = -1
        self.proc: subprocess.Popen | None = None
        self._spawn()

    def _spawn(self) -> None:
        master_fd, slave_fd = pty.openpty()
        # 40 rows x 160 cols prevents command output wrapping
        winsize = struct.pack("HHHH", 40, 160, 0, 0)
        fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, winsize)

        env = os.environ.copy()
        env["TERM"] = "dumb"
        env["PS1"] = ""

        self.proc = subprocess.Popen(
            ["/bin/bash", "--norc", "--noprofile"],
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            cwd=self.cwd if os.path.isdir(self.cwd) else str(WORKING_DIR),
            env=env,
            start_new_session=True,
            close_fds=True,
        )
        os.close(slave_fd)
        fl = fcntl.fcntl(master_fd, fcntl.F_GETFL)
        fcntl.fcntl(master_fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
        self.master_fd = master_fd

        # Disable input echo so sent commands do not duplicate in output
        os.write(self.master_fd, b"stty -echo\n")
        time.sleep(0.15)
        self._drain_nonblocking()

    def _drain_nonblocking(self) -> None:
        while True:
            r, _, _ = select.select([self.master_fd], [], [], 0.02)
            if not r:
                break
            try:
                if not os.read(self.master_fd, 65536):
                    break
            except OSError:
                break

    def ensure_alive(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            try:
                if self.master_fd >= 0:
                    os.close(self.master_fd)
            except OSError:
                pass
            self._spawn()

    def exec_command(self, command: str, timeout_sec: float = 60.0) -> dict:
        with self.lock:
            self.ensure_alive()
            self._drain_nonblocking()

            uid = uuid.uuid4().hex
            half = len(uid) // 2
            # Split token across adjacent quotes so even echoed input can never match sentinel
            sentinel_prefix = f"__MCP_END_{uid}__:"
            payload = (
                f"{command}\n"
                f"__mcp_ec=$?; printf '\\n__MCP_END_%s__:%d:%s\\n' "
                f"'{uid[:half]}''{uid[half:]}' \"$__mcp_ec\" \"$PWD\"\n"
            )

            t0 = time.monotonic()
            os.write(self.master_fd, payload.encode("utf-8", errors="replace"))

            chunks: list[bytes] = []
            buf_tail = ""
            exit_code = -1
            timed_out = False
            deadline = t0 + max(1.0, float(timeout_sec))

            while True:
                rem = deadline - time.monotonic()
                if rem <= 0:
                    timed_out = True
                    # Interrupt hung foreground command so PTY stays reusable
                    try:
                        os.write(self.master_fd, b"\x03\n")
                        time.sleep(0.2)
                        self._drain_nonblocking()
                    except OSError:
                        pass
                    break

                r, _, _ = select.select([self.master_fd], [], [], min(0.25, rem))
                if not r:
                    if self.proc and self.proc.poll() is not None:
                        break
                    continue

                try:
                    data = os.read(self.master_fd, 65536)
                    if not data:
                        break
                    chunks.append(data)
                    buf_tail = (buf_tail + data.decode("utf-8", errors="replace"))[-4096:]
                    if sentinel_prefix in buf_tail:
                        m = re.search(
                            rf"{re.escape(sentinel_prefix)}(-?\d+):([^\r\n]*)",
                            buf_tail,
                        )
                        if m:
                            exit_code = int(m.group(1))
                            self.cwd = m.group(2).strip() or self.cwd
                            break
                except BlockingIOError:
                    continue
                except OSError:
                    break

            raw_text = b"".join(chunks).decode("utf-8", errors="replace")
            idx = raw_text.find(sentinel_prefix)
            if idx != -1:
                raw_text = raw_text[:idx]

            cleaned = clean_tqdm(raw_text).strip("\n")
            if len(cleaned) > 60000:
                cleaned = (
                    cleaned[:15000]
                    + "\n\n... [truncated middle output] ...\n\n"
                    + cleaned[-40000:]
                )

            return {
                "session": self.name,
                "exit_code": exit_code,
                "cwd": self.cwd,
                "timed_out": timed_out,
                "duration_sec": round(time.monotonic() - t0, 3),
                "output": cleaned,
            }

    def close(self) -> None:
        try:
            if self.proc and self.proc.poll() is None:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
        except Exception:
            pass
        try:
            if self.master_fd >= 0:
                os.close(self.master_fd)
        except OSError:
            pass


class PtyManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, PtySession] = {}

    def get(self, name: str = "default") -> PtySession:
        with self._lock:
            if name not in self._sessions:
                self._sessions[name] = PtySession(name)
            return self._sessions[name]

    def names(self) -> list[str]:
        with self._lock:
            return list(self._sessions.keys())

    def close_all(self) -> None:
        with self._lock:
            for s in self._sessions.values():
                s.close()
            self._sessions.clear()


class JobManager:
    """Detached GPU/CPU process manager writing persistent logs to /kaggle/working/mcp_logs."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, dict] = {}
        self._procs: dict[str, subprocess.Popen] = {}
        self._log_handles: dict[str, io.BufferedWriter] = {}

    def spawn(self, command: str, cwd: str = "", name: str = "") -> dict:
        with self._lock:
            job_id = f"job_{int(time.time())}_{uuid.uuid4().hex[:6]}"
            work_cwd = cwd if (cwd and os.path.isdir(cwd)) else str(WORKING_DIR)
            log_path = LOG_DIR / f"{job_id}.log"
            fh = open(log_path, "wb")

            proc = subprocess.Popen(
                ["/bin/bash", "-c", command],
                cwd=work_cwd,
                env=os.environ.copy(),
                stdin=subprocess.DEVNULL,
                stdout=fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

            meta = {
                "job_id": job_id,
                "name": name or job_id,
                "command": command,
                "cwd": work_cwd,
                "pid": proc.pid,
                "status": "running",
                "exit_code": None,
                "started_at": time.time(),
                "ended_at": None,
                "log_path": str(log_path),
            }
            self._jobs[job_id] = meta
            self._procs[job_id] = proc
            self._log_handles[job_id] = fh
            self._persist_index()
            return dict(meta)

    def _refresh_locked(self) -> None:
        changed = False
        for job_id, proc in list(self._procs.items()):
            rc = proc.poll()
            meta = self._jobs[job_id]
            if rc is not None and meta["status"] == "running":
                meta["status"] = "completed" if rc == 0 else "failed"
                meta["exit_code"] = rc
                meta["ended_at"] = time.time()
                fh = self._log_handles.pop(job_id, None)
                if fh:
                    try:
                        fh.flush()
                        fh.close()
                    except Exception:
                        pass
                changed = True
        if changed:
            self._persist_index()

    def _persist_index(self) -> None:
        try:
            idx_file = LOG_DIR / "jobs_index.json"
            idx_file.write_text(json.dumps(list(self._jobs.values()), indent=2), encoding="utf-8")
        except Exception:
            pass

    def list_jobs(self) -> list[dict]:
        with self._lock:
            self._refresh_locked()
            now = time.time()
            out = []
            for meta in self._jobs.values():
                item = dict(meta)
                end_t = item["ended_at"] or now
                item["runtime_sec"] = round(end_t - item["started_at"], 1)
                lp = Path(item["log_path"])
                item["log_size_kb"] = round(lp.stat().st_size / 1024, 1) if lp.exists() else 0.0
                out.append(item)
            return out

    def active_count(self) -> int:
        with self._lock:
            self._refresh_locked()
            return sum(1 for m in self._jobs.values() if m["status"] == "running")

    def read_logs(self, job_id: str, tail_lines: int = 100, grep: str = "") -> dict:
        with self._lock:
            self._refresh_locked()
            if job_id not in self._jobs:
                raise KeyError(f"Unknown job_id: {job_id}")
            meta = dict(self._jobs[job_id])
            fh = self._log_handles.get(job_id)
            if fh:
                try:
                    fh.flush()
                except Exception:
                    pass

        lp = Path(meta["log_path"])
        if not lp.is_file():
            return {"job": meta, "logs": ""}

        raw = lp.read_bytes()
        if len(raw) > 2 * 1024 * 1024:
            raw = raw[-2 * 1024 * 1024 :]
        cleaned = clean_tqdm(raw.decode("utf-8", errors="replace"))
        lines = cleaned.splitlines()
        if grep:
            pat = re.compile(grep, re.IGNORECASE)
            lines = [ln for ln in lines if pat.search(ln)]
        tail = lines[-max(1, int(tail_lines)) :]
        return {"job": meta, "logs": "\n".join(tail), "returned_lines": len(tail)}

    def kill_job(self, job_id: str, sig_name: str = "SIGTERM") -> dict:
        sig = getattr(signal, sig_name.upper(), signal.SIGTERM)
        with self._lock:
            self._refresh_locked()
            if job_id not in self._jobs:
                raise KeyError(f"Unknown job_id: {job_id}")
            proc = self._procs.get(job_id)
            if proc and proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), sig)
                except ProcessLookupError:
                    pass
        time.sleep(0.4)
        with self._lock:
            self._refresh_locked()
            return dict(self._jobs[job_id])

    def terminate_all_running(self) -> None:
        with self._lock:
            self._refresh_locked()
            running_ids = [jid for jid, m in self._jobs.items() if m["status"] == "running"]
        for jid in running_ids:
            try:
                self.kill_job(jid, "SIGTERM")
            except Exception:
                pass


class WorkerState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.last_rpc_mono = time.monotonic()
        self.tunnel_url = ""
        self.ts_url = ""
        self.kaggle_sdk_ready = False
        self.pty_mgr = PtyManager()
        self.job_mgr = JobManager()
        self.gpu_util_window: deque[float] = deque(maxlen=6)

    def touch_rpc(self) -> None:
        with self.lock:
            self.last_rpc_mono = time.monotonic()

    def idle_sec(self) -> float:
        with self.lock:
            return time.monotonic() - self.last_rpc_mono


STATE = WorkerState()


def ensure_kaggle_sdk_for_publish() -> None:
    if STATE.kaggle_sdk_ready:
        return
    if KAGGLE_SECRET.startswith("KGAT_"):
        os.environ["KAGGLE_API_TOKEN"] = KAGGLE_SECRET
        os.environ.pop("KAGGLE_KEY", None)
        # Ensure Kaggle SDK inside container supports KGAT_ bearer tokens + six
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-U", "kaggle>=2.0.0", "six"],
            check=False,
        )
    else:
        os.environ["KAGGLE_KEY"] = KAGGLE_SECRET
    os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
    STATE.kaggle_sdk_ready = True


def publish_state(status: str = "running") -> None:
    """Publish current tunnel URL and session heartbeat to private Kaggle Dataset."""
    try:
        ensure_kaggle_sdk_for_publish()
        from kaggle.api.kaggle_api_extended import KaggleApi

        ds_dir = TMP_DIR / "state_ds"
        ds_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "title": STATE_DATASET_REF.split("/")[-1],
            "id": STATE_DATASET_REF,
            "licenses": [{"name": "CC0-1.0"}],
        }
        (ds_dir / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        state_payload = {
            "session_id": SESSION_ID,
            "status": status,
            "tunnel_url": STATE.tunnel_url,
            "ts_url": STATE.ts_url,
            "started_at": START_WALL,
            "updated_at": time.time(),
            "gpus": query_gpus(),
        }
        (ds_dir / "state.json").write_text(json.dumps(state_payload, indent=2), encoding="utf-8")

        api = KaggleApi()
        api.authenticate()
        try:
            api.dataset_create_version(
                folder=str(ds_dir),
                version_notes=f"{status}:{SESSION_ID[:8]}",
                quiet=True,
                convert_to_csv=False,
                dir_mode="skip",
            )
        except Exception:
            api.dataset_create_new(
                folder=str(ds_dir),
                public=False,
                quiet=True,
                convert_to_csv=False,
                dir_mode="skip",
            )
        print(f"[worker] published state ({status}) -> {STATE_DATASET_REF}", flush=True)
    except Exception as exc:
        print(f"[worker] publish_state error: {exc}", flush=True)


def start_tailscale_if_configured() -> None:
    if not TS_AUTHKEY:
        return
    try:
        ts_dir = TMP_DIR / "ts"
        ts_dir.mkdir(parents=True, exist_ok=True)
        tgz = ts_dir / "ts.tgz"
        urllib.request.urlretrieve(
            "https://pkgs.tailscale.com/stable/tailscale_latest_amd64.tgz",
            tgz,
        )
        subprocess.run(["tar", "-xzf", str(tgz), "-C", str(ts_dir)], check=True)
        extracted = next(ts_dir.glob("tailscale_*_amd64"))
        sock = str(TMP_DIR / "tailscaled.sock")
        state_dir = str(TMP_DIR / "ts-state")

        subprocess.Popen(
            [
                str(extracted / "tailscaled"),
                "--tun=userspace-networking",
                f"--statedir={state_dir}",
                f"--socket={sock}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(2.0)
        subprocess.run(
            [
                str(extracted / "tailscale"),
                f"--socket={sock}",
                "up",
                f"--auth-key={TS_AUTHKEY}",
                f"--hostname={TS_HOSTNAME}",
                "--accept-dns=false",
            ],
            check=True,
            timeout=30,
        )
        ip_res = subprocess.run(
            [str(extracted / "tailscale"), f"--socket={sock}", "ip", "-4"],
            capture_output=True,
            text=True,
            check=True,
        )
        ts_ip = ip_res.stdout.strip().splitlines()[0].strip()
        if ts_ip:
            STATE.ts_url = f"http://{ts_ip}:{RPC_PORT}"
            print(f"[worker] tailscale ready: {STATE.ts_url}", flush=True)
            publish_state("running")
    except Exception as exc:
        print(f"[worker] tailscale setup warning: {exc}", flush=True)


def cloudflared_supervisor() -> None:
    cf_bin = TMP_DIR / "cloudflared"
    if not cf_bin.is_file():
        urllib.request.urlretrieve(
            "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
            cf_bin,
        )
        cf_bin.chmod(0o755)

    while not SHUTDOWN_EVENT.is_set():
        proc = subprocess.Popen(
            [
                str(cf_bin),
                "tunnel",
                "--no-autoupdate",
                "--url",
                f"http://127.0.0.1:{RPC_PORT}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                if SHUTDOWN_EVENT.is_set():
                    break
                m = TRYCF_URL_RE.search(line)
                if m:
                    url = m.group(0)
                    if url != STATE.tunnel_url:
                        STATE.tunnel_url = url
                        print(f"[worker] cloudflared tunnel ready: {url}", flush=True)
                        publish_state("running")
        except Exception as exc:
            print(f"[worker] cloudflared stream error: {exc}", flush=True)
        finally:
            if proc.poll() is None:
                proc.terminate()
        if not SHUTDOWN_EVENT.is_set():
            time.sleep(3.0)


def safe_extract_tar(tar_bytes: bytes, dest_dir: Path) -> int:
    dest_dir = dest_dir.resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tf:
        for member in tf.getmembers():
            target = (dest_dir / member.name).resolve()
            if not str(target).startswith(str(dest_dir)):
                raise ValueError(f"Blocked path traversal in archive: {member.name}")
            if member.issym() or member.islnk():
                continue
            tf.extract(member, path=dest_dir)
            count += 1
    return count


def build_tar_bytes(target_path: Path) -> bytes:
    target_path = target_path.resolve()
    if not target_path.exists():
        raise FileNotFoundError(f"Remote path not found: {target_path}")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        tf.add(str(target_path), arcname=target_path.name)
    return buf.getvalue()


class WorkerHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        return

    def _send_json(self, code: int, data: dict) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _check_auth(self) -> bool:
        auth = self.headers.get("Authorization", "")
        expected = f"Bearer {WORKER_TOKEN}"
        if not hmac.compare_digest(auth, expected):
            self._send_json(401, {"error": "Unauthorized"})
            return False
        STATE.touch_rpc()
        return True

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:
        if self.path == "/ping":
            self._send_json(
                200,
                {
                    "ok": True,
                    "session_id": SESSION_ID,
                    "uptime_sec": round(time.monotonic() - START_MONO, 1),
                },
            )
            return

        if not self._check_auth():
            return

        if self.path == "/status":
            uptime = time.monotonic() - START_MONO
            self._send_json(
                200,
                {
                    "session_id": SESSION_ID,
                    "uptime_sec": round(uptime, 1),
                    "remaining_before_drain_sec": round(max(0.0, MAX_LIFETIME_SEC - uptime), 1),
                    "idle_sec": round(STATE.idle_sec(), 1),
                    "idle_timeout_sec": IDLE_TIMEOUT_SEC,
                    "tunnel_url": STATE.tunnel_url,
                    "ts_url": STATE.ts_url,
                    "gpus": query_gpus(),
                    "ram": query_ram_gb(),
                    "disk_working": query_disk_gb(WORKING_DIR),
                    "disk_tmp": query_disk_gb( Path("/tmp") ),
                    "active_jobs": STATE.job_mgr.active_count(),
                    "pty_sessions": STATE.pty_mgr.names(),
                },
            )
            return

        if self.path == "/jobs/list":
            self._send_json(200, {"jobs": STATE.job_mgr.list_jobs()})
            return

        self._send_json(404, {"error": f"Unknown path: {self.path}"})

    def do_POST(self) -> None:
        if not self._check_auth():
            return

        try:
            if self.path == "/term/exec":
                req = self._read_json()
                res = STATE.pty_mgr.get(req.get("session", "default")).exec_command(
                    command=req["command"],
                    timeout_sec=float(req.get("timeout_sec", 60.0)),
                )
                self._send_json(200, res)
                return

            if self.path == "/jobs/spawn":
                req = self._read_json()
                meta = STATE.job_mgr.spawn(
                    command=req["command"],
                    cwd=req.get("cwd", ""),
                    name=req.get("name", ""),
                )
                self._send_json(200, meta)
                return

            if self.path == "/jobs/logs":
                req = self._read_json()
                res = STATE.job_mgr.read_logs(
                    job_id=req["job_id"],
                    tail_lines=int(req.get("tail_lines", 100)),
                    grep=req.get("grep", ""),
                )
                self._send_json(200, res)
                return

            if self.path == "/jobs/kill":
                req = self._read_json()
                meta = STATE.job_mgr.kill_job(
                    job_id=req["job_id"],
                    sig_name=req.get("signal", "SIGTERM"),
                )
                self._send_json(200, meta)
                return

            if self.path == "/fs/upload":
                remote_dest = Path(self.headers.get("X-Remote-Path", str(WORKING_DIR)))
                length = int(self.headers.get("Content-Length", "0"))
                tar_bytes = self.rfile.read(length)
                extracted = safe_extract_tar(tar_bytes, remote_dest)
                self._send_json(
                    200,
                    {"ok": True, "extracted_entries": extracted, "remote_path": str(remote_dest)},
                )
                return

            if self.path == "/fs/download":
                req = self._read_json()
                target = Path(req["remote_path"])
                tar_bytes = build_tar_bytes(target)
                self.send_response(200)
                self.send_header("Content-Type", "application/gzip")
                self.send_header("Content-Length", str(len(tar_bytes)))
                self.send_header("X-Target-Name", target.name)
                self.end_headers()
                self.wfile.write(tar_bytes)
                return

            if self.path == "/shutdown":
                STATE.job_mgr.terminate_all_running()
                self._send_json(200, {"ok": True, "shutting_down": True})
                threading.Timer(0.5, SHUTDOWN_EVENT.set).start()
                return

            self._send_json(404, {"error": f"Unknown path: {self.path}"})
        except KeyError as exc:
            self._send_json(404, {"error": str(exc)})
        except Exception as exc:
            self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})


def main() -> None:
    fixup_cuda_env()
    print(f"[worker] booting session={SESSION_ID} on port {RPC_PORT}", flush=True)

    # Pre-warm Kaggle SDK upgrade in parallel with cloudflared startup
    sdk_thread = threading.Thread(target=ensure_kaggle_sdk_for_publish, daemon=True)
    sdk_thread.start()

    httpd = ThreadingHTTPServer(("0.0.0.0", RPC_PORT), WorkerHandler)
    srv_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    srv_thread.start()

    cf_thread = threading.Thread(target=cloudflared_supervisor, daemon=True)
    cf_thread.start()

    if TS_AUTHKEY:
        ts_thread = threading.Thread(target=start_tailscale_if_configured, daemon=True)
        ts_thread.start()

    drained = False
    last_gpu_sample = 0.0

    try:
        while not SHUTDOWN_EVENT.wait(5.0):
            now_mono = time.monotonic()
            elapsed = now_mono - START_MONO

            # Sample GPU utilization every 60s for idle quota protection
            if now_mono - last_gpu_sample >= 60.0:
                last_gpu_sample = now_mono
                gpus = query_gpus()
                max_u = max((g["util_pct"] for g in gpus), default=0.0)
                STATE.gpu_util_window.append(max_u)

            # 11h35m checkpoint drain signal
            if not drained and elapsed >= DRAIN_AT_SEC:
                drained = True
                print("[worker] 11h35m reached: sending SIGTERM to running jobs for checkpoint flush", flush=True)
                STATE.job_mgr.terminate_all_running()

            # 11h40m hard exit(0) so Kaggle commits /kaggle/working cleanly before 12h timeout
            if elapsed >= MAX_LIFETIME_SEC:
                print("[worker] 11h40m safe lifetime cap reached; exiting cleanly (0).", flush=True)
                break

            # Idle watchdog: protect weekly 30h GPU quota
            if (
                IDLE_TIMEOUT_SEC > 0
                and STATE.idle_sec() >= IDLE_TIMEOUT_SEC
                and STATE.job_mgr.active_count() == 0
            ):
                avg_gpu = (
                    sum(STATE.gpu_util_window) / len(STATE.gpu_util_window)
                    if STATE.gpu_util_window
                    else 0.0
                )
                if avg_gpu < 3.0:
                    print(
                        f"[worker] idle for {STATE.idle_sec():.0f}s with 0 active jobs & GPU<3%; auto-shutting down.",
                        flush=True,
                    )
                    break
    finally:
        STATE.job_mgr.terminate_all_running()
        STATE.pty_mgr.close_all()
        publish_state("stopped")
        httpd.shutdown()
        print("[worker] clean exit(0).", flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)
