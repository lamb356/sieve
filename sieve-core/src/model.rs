use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use indicatif::{ProgressBar, ProgressStyle};

use crate::{Result, SieveError};

pub const DEFAULT_MODEL_NAME: &str = "bge-small-en-v1.5";
pub const DEFAULT_SPARSE_MODEL_NAME: &str = "splade";
const DENSE_MODEL_URL: &str =
    "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/onnx/model.onnx";
const DENSE_TOKENIZER_URL: &str =
    "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/tokenizer.json";
const MODEL_FILE_NAME: &str = "model.onnx";
const TOKENIZER_FILE_NAME: &str = "tokenizer.json";
const SPARSE_MODEL_FILE_NAME: &str = "splade.onnx";
const SPARSE_MODEL_DATA_FILE_NAME: &str = "splade.onnx.data";
const SPARSE_TOKENIZER_RELATIVE_PATH: &str = "splade-tokenizer/tokenizer.json";

#[derive(Debug, Clone)]
pub struct ModelManager {
    models_dir: PathBuf,
    state: Arc<Mutex<CachedModels>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelRegistry {
    pub dense: Option<DenseModelHandle>,
    pub sparse: Option<SparseModelHandle>,
    pub event_reranker: Option<EventModelHandle>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseModelHandle {
    pub model_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparseModelHandle {
    pub model_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EventModelHandle {
    pub model_path: PathBuf,
    pub name: String,
}

#[derive(Debug, Clone, Default)]
struct CachedModels {
    dense: Option<DenseModelHandle>,
    sparse: Option<SparseModelHandle>,
    event_reranker: Option<EventModelHandle>,
}

impl ModelManager {
    pub fn new(data_dir: &Path) -> Self {
        Self {
            models_dir: data_dir.join("models"),
            state: Arc::new(Mutex::new(CachedModels::default())),
        }
    }

    pub fn ensure_model(&self, model_name: &str) -> Result<PathBuf> {
        self.ensure_artifact(model_name, MODEL_FILE_NAME, dense_model_url(model_name)?)
    }

    pub fn ensure_tokenizer(&self, model_name: &str) -> Result<PathBuf> {
        self.ensure_artifact(model_name, TOKENIZER_FILE_NAME, tokenizer_url(model_name)?)
    }

    pub fn ensure_dense_model(&self) -> Result<DenseModelHandle> {
        let model_path = self.ensure_model(DEFAULT_MODEL_NAME)?;
        let tokenizer_path = self.ensure_tokenizer(DEFAULT_MODEL_NAME)?;
        let handle = DenseModelHandle {
            model_path,
            tokenizer_path,
            name: DEFAULT_MODEL_NAME.to_string(),
        };
        let mut state = self.state.lock().map_err(|_| SieveError::LockPoisoned)?;
        state.dense = Some(handle.clone());
        Ok(handle)
    }

    pub fn ensure_sparse_model(&self) -> Result<SparseModelHandle> {
        let model_path = self.sparse_model_path();
        let tokenizer_path = self.sparse_tokenizer_path();
        let data_path = self.sparse_model_data_path();
        if !model_path.is_file() || !tokenizer_path.is_file() || !data_path.is_file() {
            return Err(SieveError::Message(format!(
                "SPLADE model not available at {}",
                self.model_dir(DEFAULT_SPARSE_MODEL_NAME).display()
            )));
        }
        let handle = SparseModelHandle {
            model_path,
            tokenizer_path,
            name: DEFAULT_SPARSE_MODEL_NAME.to_string(),
        };
        let mut state = self.state.lock().map_err(|_| SieveError::LockPoisoned)?;
        state.sparse = Some(handle.clone());
        Ok(handle)
    }

    pub fn ensure_event_model(&self) -> Result<EventModelHandle> {
        Err(SieveError::Message(
            "Event reranker model download not yet implemented (Phase 4 Batch 3)".to_string(),
        ))
    }

    pub fn registry(&self) -> Result<ModelRegistry> {
        let mut state = self.state.lock().map_err(|_| SieveError::LockPoisoned)?;
        if state.dense.is_none() && self.is_cached(DEFAULT_MODEL_NAME) {
            state.dense = Some(DenseModelHandle {
                model_path: self.model_dir(DEFAULT_MODEL_NAME).join(MODEL_FILE_NAME),
                tokenizer_path: self.model_dir(DEFAULT_MODEL_NAME).join(TOKENIZER_FILE_NAME),
                name: DEFAULT_MODEL_NAME.to_string(),
            });
        }
        if state.sparse.is_none() && self.is_cached(DEFAULT_SPARSE_MODEL_NAME) {
            state.sparse = Some(SparseModelHandle {
                model_path: self.sparse_model_path(),
                tokenizer_path: self.sparse_tokenizer_path(),
                name: DEFAULT_SPARSE_MODEL_NAME.to_string(),
            });
        }
        Ok(ModelRegistry {
            dense: state.dense.clone(),
            sparse: state.sparse.clone(),
            event_reranker: state.event_reranker.clone(),
        })
    }

    pub fn is_cached(&self, model_name: &str) -> bool {
        match model_name {
            DEFAULT_SPARSE_MODEL_NAME => {
                self.sparse_model_path().is_file()
                    && self.sparse_model_data_path().is_file()
                    && self.sparse_tokenizer_path().is_file()
            }
            _ => {
                self.model_dir(model_name).join(MODEL_FILE_NAME).is_file()
                    && self
                        .model_dir(model_name)
                        .join(TOKENIZER_FILE_NAME)
                        .is_file()
            }
        }
    }

    pub fn model_dir(&self, model_name: &str) -> PathBuf {
        self.models_dir.join(model_name)
    }

    pub fn sparse_model_path(&self) -> PathBuf {
        self.model_dir(DEFAULT_SPARSE_MODEL_NAME)
            .join(SPARSE_MODEL_FILE_NAME)
    }

    pub fn sparse_model_data_path(&self) -> PathBuf {
        self.model_dir(DEFAULT_SPARSE_MODEL_NAME)
            .join(SPARSE_MODEL_DATA_FILE_NAME)
    }

    pub fn sparse_tokenizer_path(&self) -> PathBuf {
        self.model_dir(DEFAULT_SPARSE_MODEL_NAME)
            .join(SPARSE_TOKENIZER_RELATIVE_PATH)
    }

    fn ensure_artifact(&self, model_name: &str, file_name: &str, url: &str) -> Result<PathBuf> {
        let model_dir = self.model_dir(model_name);
        fs::create_dir_all(&model_dir)?;
        let destination = model_dir.join(file_name);
        if destination.exists() {
            return Ok(destination);
        }
        download_to_path(url, &destination)
    }
}

fn download_to_path(url: &str, destination: &Path) -> Result<PathBuf> {
    let response = reqwest::blocking::get(url)
        .map_err(|err| io_invalid(format!("failed to download {url}: {err}")))?;
    if !response.status().is_success() {
        return Err(io_invalid(format!(
            "failed to download {url}: HTTP {}",
            response.status()
        )));
    }

    let total = response.content_length().unwrap_or(0);
    let progress = ProgressBar::new(total);
    progress.set_style(
        ProgressStyle::with_template("{msg} [{bar:40.cyan/blue}] {bytes}/{total_bytes}")
            .unwrap_or_else(|_| ProgressStyle::default_bar()),
    );
    progress.set_message(format!("Downloading {}", destination.display()));

    let temp_path = destination.with_extension("partial");
    let mut file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&temp_path)?;
    let mut reader = response;
    let mut buffer = [0_u8; 16 * 1024];
    loop {
        let read = reader
            .read(&mut buffer)
            .map_err(|err| io_invalid(format!("failed while downloading {url}: {err}")))?;
        if read == 0 {
            break;
        }
        file.write_all(&buffer[..read])?;
        progress.inc(read as u64);
    }
    file.flush()?;
    file.sync_data()?;
    drop(file);
    progress.finish_and_clear();
    fs::rename(&temp_path, destination)?;
    sync_dir(destination.parent().unwrap_or_else(|| Path::new(".")))?;
    Ok(destination.to_path_buf())
}

fn sync_dir(path: &Path) -> Result<()> {
    File::open(path)?.sync_all()?;
    Ok(())
}

fn io_invalid(message: impl Into<String>) -> SieveError {
    SieveError::Io(std::io::Error::other(message.into()))
}

fn dense_model_url(model_name: &str) -> Result<&'static str> {
    match model_name {
        DEFAULT_MODEL_NAME => Ok(DENSE_MODEL_URL),
        other => Err(SieveError::Message(format!(
            "unsupported model artifact lookup for {other}"
        ))),
    }
}

fn tokenizer_url(model_name: &str) -> Result<&'static str> {
    match model_name {
        DEFAULT_MODEL_NAME => Ok(DENSE_TOKENIZER_URL),
        other => Err(SieveError::Message(format!(
            "unsupported tokenizer lookup for {other}"
        ))),
    }
}
