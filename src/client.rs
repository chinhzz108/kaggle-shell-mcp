//! Authenticated async HTTP client to the remote Kaggle worker with auto-healing
//! tunnel endpoint refresh on 502/504/530 or transient transport errors.

use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use reqwest::{Client, Method, Response};
use serde_json::{Value, json};

use crate::config::{ActiveSession, Settings, load_active_session, save_active_session};
use crate::kaggle_api::KaggleApiClient;
use crate::launcher::{fetch_remote_state, probe_worker_ping};
use crate::sync::{pack_tar_gz, unpack_tar_gz};
use crate::utils::expand_tilde;

const RETRY_STATUS_CODES: &[u16] = &[502, 504, 530];

#[derive(Debug, Clone)]
pub struct WorkerClient {
    settings: Settings,
    http: Client,
}

impl WorkerClient {
    pub fn new(settings: Settings) -> Result<Self> {
        let http = Client::builder()
            .build()
            .context("Failed to initialize worker HTTP client")?;
        Ok(Self { settings, http })
    }

    async fn get_ready_session(&self, allow_refresh: bool) -> Result<ActiveSession> {
        let sess = load_active_session(&self.settings.state_dir).context(
            "No active Kaggle worker session found. Call `kaggle_up` first.",
        )?;

        let base = sess.base_url();
        if !base.is_empty() && probe_worker_ping(&base, &sess.session_id).await.is_some() {
            return Ok(sess);
        }

        if allow_refresh
            && let Some(refreshed) = self.refresh_endpoint(&sess).await
            && !refreshed.base_url().is_empty()
        {
            return Ok(refreshed);
        }

        bail!(
            "Could not connect to worker (session={}). Check with `kaggle_status` or call `kaggle_up` to start a new session.",
            sess.session_id
        );
    }

    async fn refresh_endpoint(&self, sess: &ActiveSession) -> Option<ActiveSession> {
        let api = KaggleApiClient::new(&self.settings).ok()?;
        let rstate = fetch_remote_state(&api, &self.settings).await?;
        if rstate.get("session_id").and_then(|v| v.as_str()) != Some(sess.session_id.as_str()) {
            return None;
        }
        if rstate.get("status").and_then(|v| v.as_str()) == Some("stopped") {
            return None;
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

        for url in [ts_url.as_str(), tunnel_url.as_str()]
            .into_iter()
            .filter(|u| !u.is_empty())
        {
            if probe_worker_ping(url, &sess.session_id).await.is_some() {
                let mut updated = sess.clone();
                updated.tunnel_url = tunnel_url.clone();
                updated.ts_url = if url == ts_url {
                    ts_url.clone()
                } else {
                    String::new()
                };
                let _ = save_active_session(&self.settings.state_dir, &updated);
                return Some(updated);
            }
        }
        None
    }

    async fn request(
        &self,
        method: Method,
        path: &str,
        json_body: Option<&Value>,
        raw_body: Option<&[u8]>,
        extra_headers: Option<&[(&str, &str)]>,
        timeout_sec: f64,
    ) -> Result<Response> {
        let mut sess = self.get_ready_session(true).await?;

        let mut headers = HeaderMap::new();
        let auth_val = HeaderValue::from_str(&format!("Bearer {}", sess.worker_token))
            .context("Invalid worker token header")?;
        headers.insert(AUTHORIZATION, auth_val);

        if let Some(extras) = extra_headers {
            for (k, v) in extras {
                let name = HeaderName::from_bytes(k.as_bytes())
                    .with_context(|| format!("Invalid header name: {k}"))?;
                let val = HeaderValue::from_str(v)
                    .with_context(|| format!("Invalid header value for {k}"))?;
                headers.insert(name, val);
            }
        }

        let timeout = Duration::from_secs_f64(timeout_sec.max(1.0));

        for attempt in 0..2 {
            let url = format!("{}{path}", sess.base_url());
            let mut req = self
                .http
                .request(method.clone(), &url)
                .headers(headers.clone())
                .timeout(timeout);

            if let Some(body) = json_body {
                req = req.json(body);
            } else if let Some(bytes) = raw_body {
                req = req.body(bytes.to_vec());
            }

            match req.send().await {
                Ok(resp) => {
                    let status = resp.status();
                    if RETRY_STATUS_CODES.contains(&status.as_u16()) && attempt == 0 {
                        if let Some(refreshed) = self.refresh_endpoint(&sess).await {
                            sess = refreshed;
                            continue;
                        }
                    }
                    if status.as_u16() >= 400 {
                        let text = resp.text().await.unwrap_or_default();
                        let snippet: String = text.chars().take(500).collect();
                        bail!("Worker HTTP {}: {snippet}", status.as_u16());
                    }
                    return Ok(resp);
                }
                Err(err) => {
                    if attempt == 0
                        && (err.is_connect() || err.is_timeout() || err.is_request() || err.is_body())
                        && let Some(refreshed) = self.refresh_endpoint(&sess).await
                    {
                        sess = refreshed;
                        continue;
                    }
                    bail!("Connection error to Kaggle worker ({url}): {err}");
                }
            }
        }

        bail!("Request failed after endpoint refresh.");
    }

    pub async fn status(&self) -> Result<Value> {
        let resp = self
            .request(Method::GET, "/status", None, None, None, 20.0)
            .await?;
        Ok(resp.json().await?)
    }

    pub async fn shutdown(&self) -> Result<Value> {
        let body = json!({ "graceful": true });
        let resp = self
            .request(Method::POST, "/shutdown", Some(&body), None, None, 20.0)
            .await?;
        Ok(resp.json().await?)
    }

    pub async fn term_exec(
        &self,
        command: &str,
        session: &str,
        timeout_sec: f64,
    ) -> Result<Value> {
        let http_timeout = (timeout_sec + 15.0).max(30.0);
        let body = json!({
            "command": command,
            "session": session,
            "timeout_sec": timeout_sec,
        });
        let resp = self
            .request(Method::POST, "/term/exec", Some(&body), None, None, http_timeout)
            .await?;
        Ok(resp.json().await?)
    }

    pub async fn job_spawn(&self, command: &str, cwd: &str, name: &str) -> Result<Value> {
        let body = json!({
            "command": command,
            "cwd": cwd,
            "name": name,
        });
        let resp = self
            .request(Method::POST, "/jobs/spawn", Some(&body), None, None, 30.0)
            .await?;
        Ok(resp.json().await?)
    }

    pub async fn job_list(&self) -> Result<Value> {
        let resp = self
            .request(Method::GET, "/jobs/list", None, None, None, 20.0)
            .await?;
        Ok(resp.json().await?)
    }

    pub async fn job_logs(
        &self,
        job_id: &str,
        tail_lines: usize,
        grep: &str,
    ) -> Result<Value> {
        let body = json!({
            "job_id": job_id,
            "tail_lines": tail_lines,
            "grep": grep,
        });
        let resp = self
            .request(Method::POST, "/jobs/logs", Some(&body), None, None, 30.0)
            .await?;
        Ok(resp.json().await?)
    }

    pub async fn job_kill(&self, job_id: &str, signal: &str) -> Result<Value> {
        let body = json!({
            "job_id": job_id,
            "signal": signal,
        });
        let resp = self
            .request(Method::POST, "/jobs/kill", Some(&body), None, None, 20.0)
            .await?;
        Ok(resp.json().await?)
    }

    pub async fn fs_push(&self, local_path: &Path, remote_path: &str) -> Result<Value> {
        let tar_bytes = pack_tar_gz(local_path)?;
        let uploaded_kb = ((tar_bytes.len() as f64 / 1024.0) * 10.0).round() / 10.0;
        let extras = [
            (CONTENT_TYPE.as_str(), "application/gzip"),
            ("X-Remote-Path", remote_path),
        ];
        let resp = self
            .request(
                Method::POST,
                "/fs/upload",
                None,
                Some(&tar_bytes),
                Some(&extras),
                300.0,
            )
            .await?;
        let mut res: Value = resp.json().await?;
        if let Some(obj) = res.as_object_mut() {
            obj.insert("uploaded_kb".to_string(), json!(uploaded_kb));
        }
        Ok(res)
    }

    pub async fn fs_pull(&self, remote_path: &str, local_path: &Path) -> Result<Value> {
        let body = json!({ "remote_path": remote_path });
        let resp = self
            .request(Method::POST, "/fs/download", Some(&body), None, None, 300.0)
            .await?;
        let raw_bytes = resp.bytes().await?;
        let downloaded_kb = ((raw_bytes.len() as f64 / 1024.0) * 10.0).round() / 10.0;
        let extracted = unpack_tar_gz(&raw_bytes, local_path)?;
        let resolved_dest = std::fs::canonicalize(expand_tilde(local_path))
            .unwrap_or_else(|_| expand_tilde(local_path));
        let sample_files: Vec<String> = extracted.iter().take(20).cloned().collect();

        Ok(json!({
            "ok": true,
            "remote_path": remote_path,
            "local_dest": resolved_dest.to_string_lossy(),
            "downloaded_kb": downloaded_kb,
            "extracted_count": extracted.len(),
            "sample_files": sample_files,
        }))
    }
}
