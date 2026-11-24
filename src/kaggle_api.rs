//! Pure-Rust async client for Kaggle's official API (`https://api.kaggle.com/v1/...`),
//! compatible with both modern `KGAT_` Bearer tokens and legacy username/key Basic auth.

use std::fs;
use std::io::{Cursor, Read};
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use reqwest::{Client, RequestBuilder, StatusCode, redirect::Policy};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use thiserror::Error;
use zip::ZipArchive;

use crate::config::{KaggleAuth, Settings};
use crate::utils::{expand_tilde, normalize_kernel_status, safe_join, split_kaggle_ref};

const USER_AGENT: &str = "kaggle-api/v1.7.0";

#[derive(Debug, Error)]
pub enum KaggleApiError {
    #[error("Kaggle API HTTP {status}: {message}")]
    Http { status: u16, message: String },
    #[error("Kaggle API request failed: {0}")]
    Transport(#[from] reqwest::Error),
    #[error("{0}")]
    Other(#[from] anyhow::Error),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelStatusInfo {
    pub status: String,
    pub failure_message: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SaveKernelRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub slug: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub new_title: Option<String>,
    pub text: String,
    pub language: String,
    pub kernel_type: String,
    pub is_private: bool,
    pub enable_gpu: bool,
    pub enable_tpu: bool,
    pub enable_internet: bool,
    #[serde(default)]
    pub dataset_data_sources: Vec<String>,
    #[serde(default)]
    pub competition_data_sources: Vec<String>,
    #[serde(default)]
    pub kernel_data_sources: Vec<String>,
    #[serde(default)]
    pub model_data_sources: Vec<String>,
    #[serde(default)]
    pub category_ids: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub machine_shape: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SaveKernelResponse {
    #[serde(default)]
    pub r#ref: String,
    #[serde(default)]
    pub url: String,
    pub version_number: Option<i64>,
    pub error: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GetKernelStatusRequest {
    user_name: String,
    kernel_slug: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GetKernelStatusResponse {
    #[serde(default)]
    status: serde_json::Value,
    #[serde(default)]
    failure_message: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ListKernelOutputRequest {
    user_name: String,
    kernel_slug: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    page_size: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    page_token: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct KernelOutputFile {
    url: Option<String>,
    file_name: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ListKernelOutputResponse {
    #[serde(default)]
    files: Vec<KernelOutputFile>,
    log: Option<String>,
    next_page_token: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ListDatasetFilesRequest {
    owner_slug: String,
    dataset_slug: String,
    page_size: i32,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ListDatasetFilesResponse {
    #[serde(default)]
    pub dataset_files: Vec<serde_json::Value>,
    pub error_message: Option<String>,
    pub next_page_token: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct StartBlobUploadRequest {
    r#type: &'static str,
    name: String,
    content_length: u64,
    last_modified_epoch_seconds: u64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct StartBlobUploadResponse {
    token: String,
    create_url: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct DatasetNewFile {
    token: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct CreateDatasetRequest {
    owner_slug: String,
    slug: String,
    title: String,
    license_name: String,
    is_private: bool,
    files: Vec<DatasetNewFile>,
    category_ids: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateDatasetResponse {
    pub r#ref: Option<String>,
    pub url: Option<String>,
    pub status: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct DownloadDatasetRequest {
    owner_slug: String,
    dataset_slug: String,
}

#[derive(Debug, Clone)]
pub struct KaggleApiClient {
    base_url: String,
    auth: KaggleAuth,
    rpc_client: Client,
    no_redirect_client: Client,
    plain_client: Client,
}

impl KaggleApiClient {
    pub fn new(settings: &Settings) -> Result<Self> {
        let rpc_client = Client::builder()
            .user_agent(USER_AGENT)
            .timeout(Duration::from_secs(60))
            .build()
            .context("Failed to initialize Kaggle HTTP client")?;

        let no_redirect_client = Client::builder()
            .user_agent(USER_AGENT)
            .redirect(Policy::none())
            .timeout(Duration::from_secs(60))
            .build()
            .context("Failed to initialize no-redirect HTTP client")?;

        let plain_client = Client::builder()
            .user_agent(USER_AGENT)
            .timeout(Duration::from_secs(300))
            .build()
            .context("Failed to initialize file transfer HTTP client")?;

        Ok(Self {
            base_url: settings.kaggle_api_base.trim_end_matches('/').to_string(),
            auth: settings.auth(),
            rpc_client,
            no_redirect_client,
            plain_client,
        })
    }

    fn rpc_url(&self, service: &str, method: &str) -> String {
        // Prod uses https://api.kaggle.com/v1/{service}/{method}
        // Custom mock bases also serve /v1/{service}/{method}
        format!("{}/v1/{service}/{method}", self.base_url)
    }

    fn apply_auth(&self, req: RequestBuilder) -> RequestBuilder {
        match &self.auth {
            KaggleAuth::Bearer(token) => req.bearer_auth(token),
            KaggleAuth::Basic { username, key } => req.basic_auth(username, Some(key)),
        }
    }

    async fn call_rpc<Req, Resp>(
        &self,
        service: &str,
        method: &str,
        payload: &Req,
    ) -> std::result::Result<Resp, KaggleApiError>
    where
        Req: Serialize,
        Resp: DeserializeOwned,
    {
        let url = self.rpc_url(service, method);
        let resp = self
            .apply_auth(self.rpc_client.post(&url))
            .json(payload)
            .send()
            .await?;

        let status = resp.status();
        let content_type = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_ascii_lowercase();
        let body_text = resp.text().await?;

        if content_type.contains("application/json")
            && let Ok(val) = serde_json::from_str::<serde_json::Value>(&body_text)
        {
            // Kaggle returns either top-level `{"code": 403, "message": "..."}`
            // or Google RPC style `{"error": {"code": 403, "message": "..."}}`.
            let err_obj = val.get("error").filter(|e| e.is_object()).unwrap_or(&val);
            if let Some(code) = err_obj.get("code").and_then(|c| c.as_u64())
                && code >= 400
            {
                let msg = err_obj
                    .get("message")
                    .and_then(|m| m.as_str())
                    .unwrap_or(&body_text)
                    .to_string();
                return Err(KaggleApiError::Http {
                    status: code as u16,
                    message: msg,
                });
            }
        }

        if !status.is_success() {
            return Err(KaggleApiError::Http {
                status: status.as_u16(),
                message: body_text.chars().take(500).collect(),
            });
        }

        serde_json::from_str(&body_text).map_err(|e| {
            KaggleApiError::Other(anyhow::anyhow!(
                "Failed to decode Kaggle API response from {service}/{method}: {e} (body: {})",
                body_text.chars().take(300).collect::<String>()
            ))
        })
    }

    /// Return normalized `{ status, failure_message }` for a Kaggle kernel.
    pub async fn get_kernel_status(&self, kernel_ref: &str) -> KernelStatusInfo {
        let Ok((owner, slug)) = split_kaggle_ref(kernel_ref) else {
            return KernelStatusInfo {
                status: "UNKNOWN".to_string(),
                failure_message: format!("Invalid kernel_ref: {kernel_ref}"),
            };
        };

        let req = GetKernelStatusRequest {
            user_name: owner,
            kernel_slug: slug,
        };

        match self
            .call_rpc::<_, GetKernelStatusResponse>(
                "kernels.KernelsApiService",
                "GetKernelSessionStatus",
                &req,
            )
            .await
        {
            Ok(resp) => {
                let raw_status = match &resp.status {
                    serde_json::Value::String(s) => s.clone(),
                    serde_json::Value::Number(n) => match n.as_i64() {
                        Some(0) => "QUEUED".to_string(),
                        Some(1) => "RUNNING".to_string(),
                        Some(2) => "COMPLETE".to_string(),
                        Some(3) => "ERROR".to_string(),
                        Some(4) => "CANCEL_REQUESTED".to_string(),
                        Some(5) => "CANCEL_ACKNOWLEDGED".to_string(),
                        Some(6) => "NEW_SCRIPT".to_string(),
                        _ => n.to_string(),
                    },
                    _ => "UNKNOWN".to_string(),
                };
                KernelStatusInfo {
                    status: normalize_kernel_status(&raw_status),
                    failure_message: resp.failure_message.unwrap_or_default(),
                }
            }
            Err(err) => {
                let msg = err.to_string();
                let low = msg.to_ascii_lowercase();
                if matches!(err, KaggleApiError::Http { status: 404, .. })
                    || msg.contains("404")
                    || low.contains("notfound")
                    || low.contains("not found")
                    || low.contains("cannot access kernel")
                    || (low.contains("kernels.get") && low.contains("denied"))
                {
                    KernelStatusInfo {
                        status: "NOT_CREATED_YET".to_string(),
                        failure_message: String::new(),
                    }
                } else {
                    KernelStatusInfo {
                        status: "UNKNOWN".to_string(),
                        failure_message: msg,
                    }
                }
            }
        }
    }

    /// Push a kernel script + metadata to Kaggle (`SaveKernel`).
    pub async fn save_kernel(&self, req: &SaveKernelRequest) -> Result<SaveKernelResponse> {
        let resp: SaveKernelResponse = self
            .call_rpc("kernels.KernelsApiService", "SaveKernel", req)
            .await
            .context("Kaggle SaveKernel RPC failed")?;

        if let Some(err) = resp.error.as_deref()
            && !err.is_empty()
        {
            bail!("Kaggle kernels_push failed: {err}");
        }
        Ok(resp)
    }

    /// Check whether a dataset exists by listing its files.
    pub async fn list_dataset_files(&self, dataset_ref: &str) -> Result<ListDatasetFilesResponse> {
        let (owner, slug) = split_kaggle_ref(dataset_ref)?;
        let req = ListDatasetFilesRequest {
            owner_slug: owner,
            dataset_slug: slug,
            page_size: 20,
        };
        let resp: ListDatasetFilesResponse = self
            .call_rpc("datasets.DatasetApiService", "ListDatasetFiles", &req)
            .await?;

        if let Some(err) = resp.error_message.as_deref()
            && !err.is_empty()
        {
            bail!("ListDatasetFiles error: {err}");
        }
        Ok(resp)
    }

    /// Create a brand-new private Kaggle dataset containing a single file via
    /// `StartBlobUpload` -> GCS `PUT` -> `CreateDataset`.
    pub async fn create_private_dataset_with_file(
        &self,
        dataset_ref: &str,
        title: &str,
        file_name: &str,
        content: &[u8],
    ) -> Result<CreateDatasetResponse> {
        let (owner, slug) = split_kaggle_ref(dataset_ref)?;
        let now_epoch = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // 1. Reserve GCS upload slot
        let start_req = StartBlobUploadRequest {
            r#type: "DATASET",
            name: file_name.to_string(),
            content_length: content.len() as u64,
            last_modified_epoch_seconds: now_epoch,
        };
        let start_resp: StartBlobUploadResponse = self
            .call_rpc("blobs.BlobApiService", "StartBlobUpload", &start_req)
            .await
            .context("Kaggle StartBlobUpload failed")?;

        // 2. Upload file bytes directly to the signed GCS URL (no Kaggle auth header!)
        let put_resp = self
            .plain_client
            .put(&start_resp.create_url)
            .body(content.to_vec())
            .send()
            .await
            .context("Failed to PUT blob bytes to signed upload URL")?;

        if !put_resp.status().is_success() {
            let st = put_resp.status();
            let text = put_resp.text().await.unwrap_or_default();
            bail!("Blob upload PUT failed with HTTP {st}: {text}");
        }

        // 3. Finalize private dataset creation
        let create_req = CreateDatasetRequest {
            owner_slug: owner,
            slug,
            title: title.to_string(),
            license_name: "CC0-1.0".to_string(),
            is_private: true,
            files: vec![DatasetNewFile {
                token: start_resp.token,
            }],
            category_ids: Vec::new(),
        };
        let create_resp: CreateDatasetResponse = self
            .call_rpc("datasets.DatasetApiService", "CreateDataset", &create_req)
            .await
            .context("Kaggle CreateDataset failed")?;

        if let Some(err) = create_resp.error.as_deref()
            && !err.is_empty()
        {
            bail!("Kaggle CreateDataset returned error: {err}");
        }
        Ok(create_resp)
    }

    /// Download a dataset archive (or raw file) and extract `target_file` bytes in memory.
    pub async fn download_dataset_file_bytes(
        &self,
        dataset_ref: &str,
        target_file: &str,
    ) -> Result<Option<Vec<u8>>> {
        let (owner, slug) = split_kaggle_ref(dataset_ref)?;
        let url = self.rpc_url("datasets.DatasetApiService", "DownloadDataset");
        let req = DownloadDatasetRequest {
            owner_slug: owner,
            dataset_slug: slug,
        };

        let resp = self
            .apply_auth(self.no_redirect_client.post(&url))
            .json(&req)
            .send()
            .await
            .context("Kaggle DownloadDataset request failed")?;

        let status = resp.status();
        let bytes = if matches!(
            status,
            StatusCode::MOVED_PERMANENTLY
                | StatusCode::FOUND
                | StatusCode::SEE_OTHER
                | StatusCode::TEMPORARY_REDIRECT
                | StatusCode::PERMANENT_REDIRECT
        ) {
            let location = resp
                .headers()
                .get(reqwest::header::LOCATION)
                .and_then(|v| v.to_str().ok())
                .context("DownloadDataset redirect missing Location header")?;

            let is_absolute =
                location.starts_with("http://") || location.starts_with("https://");
            let target_url = if is_absolute {
                location.to_string()
            } else {
                format!("{}{}", self.base_url, location)
            };

            let req_builder = if is_absolute && !target_url.starts_with(&self.base_url) {
                // Cross-origin redirect (e.g. GCS signed URL): never forward Kaggle auth header
                self.plain_client.get(&target_url)
            } else {
                self.apply_auth(self.plain_client.get(&target_url))
            };

            let dl = req_builder
                .send()
                .await
                .context("Failed to follow DownloadDataset redirect")?;
            if !dl.status().is_success() {
                bail!("Dataset redirect download returned HTTP {}", dl.status());
            }
            dl.bytes().await?.to_vec()
        } else if status.is_success() {
            let content_type = resp
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .unwrap_or("")
                .to_ascii_lowercase();
            let raw = resp.bytes().await?.to_vec();
            if content_type.contains("application/json")
                && let Ok(val) = serde_json::from_slice::<serde_json::Value>(&raw)
                && let Some(code) = val.get("code").and_then(|c| c.as_u64())
                && code >= 400
            {
                let msg = val
                    .get("message")
                    .and_then(|m| m.as_str())
                    .unwrap_or("DownloadDataset API error");
                bail!("DownloadDataset error {code}: {msg}");
            }
            raw
        } else {
            let text = resp.text().await.unwrap_or_default();
            bail!("DownloadDataset HTTP {status}: {text}");
        };

        // Check ZIP magic header (`PK\x03\x04` or empty archive `PK\x05\x06`)
        if bytes.starts_with(b"PK\x03\x04") || bytes.starts_with(b"PK\x05\x06") {
            let mut archive = ZipArchive::new(Cursor::new(&bytes))
                .context("Failed to parse downloaded dataset ZIP")?;
            for i in 0..archive.len() {
                let mut entry = archive.by_index(i)?;
                let name = entry.name()?.replace('\\', "/");
                if name == target_file || name.ends_with(&format!("/{target_file}")) {
                    let mut out = Vec::new();
                    entry.read_to_end(&mut out)?;
                    return Ok(Some(out));
                }
            }
            return Ok(None);
        }

        // Otherwise treat bytes directly as the target file content
        Ok(Some(bytes))
    }

    /// Download all committed output files (plus kernel log) for a Kaggle kernel.
    pub async fn pull_kernel_outputs(
        &self,
        kernel_ref: &str,
        local_dest: &Path,
    ) -> Result<Vec<String>> {
        let (owner, slug) = split_kaggle_ref(kernel_ref)?;
        let expanded = expand_tilde(local_dest);
        fs::create_dir_all(&expanded)
            .with_context(|| format!("Failed to create output dir {}", expanded.display()))?;
        let canonical_dest = fs::canonicalize(&expanded)?;

        let mut downloaded = Vec::new();
        let mut page_token: Option<String> = None;
        let mut first_page = true;

        loop {
            let req = ListKernelOutputRequest {
                user_name: owner.clone(),
                kernel_slug: slug.clone(),
                page_size: Some(20),
                page_token: page_token.clone(),
            };
            let resp: ListKernelOutputResponse = self
                .call_rpc(
                    "kernels.KernelsApiService",
                    "ListKernelSessionOutput",
                    &req,
                )
                .await
                .context("Kaggle ListKernelSessionOutput failed")?;

            for item in resp.files {
                let (Some(url), Some(file_name)) = (item.url, item.file_name) else {
                    continue;
                };
                let outfile = safe_join(&canonical_dest, &file_name)?;
                if let Some(parent) = outfile.parent() {
                    fs::create_dir_all(parent)?;
                }

                let dl_resp = self
                    .plain_client
                    .get(&url)
                    .send()
                    .await
                    .with_context(|| format!("Failed to download kernel output '{file_name}'"))?;
                if !dl_resp.status().is_success() {
                    bail!(
                        "Failed to download kernel output '{file_name}': HTTP {}",
                        dl_resp.status()
                    );
                }
                let data = dl_resp.bytes().await?;
                fs::write(&outfile, &data)
                    .with_context(|| format!("Failed to write {}", outfile.display()))?;
                downloaded.push(outfile.to_string_lossy().into_owned());
            }

            if first_page
                && let Some(log_text) = resp.log
                && !log_text.is_empty()
            {
                let log_path = canonical_dest.join(format!("{slug}.log"));
                fs::write(&log_path, log_text)?;
                downloaded.push(log_path.to_string_lossy().into_owned());
            }
            first_page = false;

            match resp.next_page_token {
                Some(tok) if !tok.is_empty() => page_token = Some(tok),
                _ => break,
            }
        }

        Ok(downloaded)
    }
}
