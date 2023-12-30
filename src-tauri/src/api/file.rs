use std::{
    fs::{File, Metadata},
    io::{BufReader, Cursor, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

use anyhow::{bail, Result};
use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use itertools::Itertools;
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{config::Config, ThreadState};

const PERMITTED_PATHS: [&str; 4] = [
    "assets/models",
    "assets/tokenizer",
    "assets/configs",
    "assets/www",
];
const UNZIP_PATHS: [&str; 2] = ["assets/unzip", "assets/temp"];

fn check_path_permitted(path: impl AsRef<Path>, permitted: &[&str]) -> Result<()> {
    let current_path = std::env::current_dir()?;
    for sub in permitted {
        let permitted = std::fs::canonicalize(current_path.join(sub))?;
        let path = std::fs::canonicalize(path.as_ref())?;
        if path.starts_with(permitted) {
            return Ok(());
        }
    }
    bail!("path not valid");
}

fn check_path(path: impl AsRef<Path>) -> Result<()> {
    check_path_permitted(path, &PERMITTED_PATHS)
}

fn compute_sha(path: impl AsRef<Path>, meta: &Metadata) -> Result<String> {
    let file = File::open(path.as_ref())?;
    let mut reader = BufReader::new(file);
    let mut buffer = Vec::new();

    if meta.len() > 10_000_000 {
        let segment_size = meta.len() / 10;
        for i in 0..10 {
            let mut segment_buffer = vec![0; 1_000_000];
            reader.seek(SeekFrom::Start(i * segment_size))?;
            reader.read_exact(&mut segment_buffer)?;
            buffer.extend(segment_buffer);
        }
    } else {
        reader.read_to_end(&mut buffer)?;
    }

    let mut sha = Sha256::new();
    sha.update(&buffer);
    let result = sha.finalize();

    Ok(format!("{:x}", result))
}

#[derive(Debug, Clone, Deserialize)]
pub struct FileInfoRequest {
    path: PathBuf,
    #[serde(default)]
    is_sha: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct FileInfo {
    name: String,
    size: u64,
    sha: String,
}

/// `/api/files/dir`, `/api/files/ls`.
pub async fn dir(
    State(ThreadState(_)): State<ThreadState>,
    Json(request): Json<FileInfoRequest>,
) -> impl IntoResponse {
    if let Err(err) = check_path(&request.path) {
        log::error!("check path failed: {}", err);
        return Err(StatusCode::FORBIDDEN);
    }
    match std::fs::read_dir(request.path) {
        Ok(dir) => {
            let files = dir
                .filter_map(|x| x.ok())
                .filter(|x| x.path().is_file())
                .filter_map(|x| {
                    let path = x.path();
                    let meta = x.metadata().ok()?;

                    let name = x.file_name().to_string_lossy().into();
                    let sha = request
                        .is_sha
                        .then(|| compute_sha(path, &meta).ok())
                        .flatten()
                        .unwrap_or_default();

                    Some(FileInfo {
                        name,
                        size: meta.len(),
                        sha,
                    })
                })
                .collect_vec();
            Ok((StatusCode::OK, Json(files)))
        }
        Err(err) => {
            log::error!("failed to read directory: {}", err);
            Err(StatusCode::NOT_FOUND)
        }
    }
}

/// `/api/models/list`.
pub async fn models(state: State<ThreadState>) -> impl IntoResponse {
    let request = FileInfoRequest {
        path: "assets/models".into(),
        is_sha: true,
    };
    dir(state, Json(request)).await
}

#[derive(Debug, Clone, Deserialize)]
pub struct UnzipRequest {
    #[serde(alias = "zip_path")]
    path: PathBuf,
    #[serde(alias = "target_dir")]
    output: PathBuf,
}

/// `/api/files/unzip`.
pub async fn unzip(
    State(ThreadState(_)): State<ThreadState>,
    Json(request): Json<UnzipRequest>,
) -> StatusCode {
    if let Err(err) = check_path(&request.path) {
        log::error!("check path failed: {}", err);
        return StatusCode::FORBIDDEN;
    }
    if let Err(err) = check_path_permitted(&request.output, &UNZIP_PATHS) {
        log::error!("check path failed: {}", err);
        return StatusCode::FORBIDDEN;
    }

    let unzip = move || -> Result<()> {
        if Path::new(&request.output).exists() {
            std::fs::remove_dir_all(&request.output)?;
        }
        std::fs::create_dir_all(&request.output)?;

        let file = File::open(&request.path)?;
        let map = unsafe { Mmap::map(&file)? };
        zip_extract::extract(Cursor::new(&map), &request.output, false)?;

        Ok(())
    };

    match unzip() {
        Ok(_) => StatusCode::OK,
        Err(err) => {
            log::error!("failed to unzip: {}", err);
            StatusCode::NOT_FOUND
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct LoadRequest {
    path: PathBuf,
}

/// `/api/files/config/load`.
pub async fn load_config(
    State(ThreadState(_)): State<ThreadState>,
    Json(request): Json<LoadRequest>,
) -> impl IntoResponse {
    if let Err(err) = check_path(&request.path) {
        log::error!("check path failed: {}", err);
        return Err(StatusCode::FORBIDDEN);
    }
    match crate::load_config(request.path) {
        Ok(config) => Ok((StatusCode::OK, Json(config))),
        Err(err) => {
            log::error!("failed to load config: {}", err);
            Err(StatusCode::NOT_FOUND)
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct SaveRequest {
    path: PathBuf,
    config: Config,
}

/// `/api/files/config/save`.
pub async fn save_config(
    State(ThreadState(_)): State<ThreadState>,
    Json(request): Json<SaveRequest>,
) -> StatusCode {
    if let Err(err) = check_path(&request.path) {
        log::error!("check path failed: {}", err);
        return StatusCode::FORBIDDEN;
    }

    let write = || -> Result<()> {
        let mut file = File::create(&request.path)?;
        let buf = toml::to_string(&request.config)?.into_bytes();
        file.write_all(&buf)?;
        Ok(())
    };

    match request.path.extension() {
        Some(ext) if ext == "toml" => match write() {
            Ok(_) => StatusCode::OK,
            Err(err) => {
                log::error!("failed to save config: {err}");
                StatusCode::INTERNAL_SERVER_ERROR
            }
        },
        _ => {
            log::error!(
                "failed to save config: file path {} is not toml",
                request.path.to_string_lossy()
            );
            StatusCode::FORBIDDEN
        }
    }
}
