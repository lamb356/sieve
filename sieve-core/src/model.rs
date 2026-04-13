use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use indicatif::{ProgressBar, ProgressStyle};

use crate::{Result, SieveError};

pub const DEFAULT_MODEL_NAME: &str = "bge-small-en-v1.5";
pub const DEFAULT_SPARSE_MODEL_NAME: &str = "splade-v3";
const MODEL_URL: &str =
    "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/onnx/model.onnx";
const TOKENIZER_URL: &str =
    "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/tokenizer.json";
const MODEL_FILE_NAME: &str = "model.onnx";
const TOKENIZER_FILE_NAME: &str = "tokenizer.json";

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
        self.ensure_artifact(model_name, MODEL_FILE_NAME, MODEL_URL)
    }

    pub fn ensure_tokenizer(&self, model_name: &str) -> Result<PathBuf> {
        self.ensure_artifact(model_name, TOKENIZER_FILE_NAME, TOKENIZER_URL)
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
        Err(SieveError::Message(
            "SPLADE model download not yet implemented (Phase 4 Batch 2)".to_string(),
        ))
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
        Ok(ModelRegistry {
            dense: state.dense.clone(),
            sparse: state.sparse.clone(),
            event_reranker: state.event_reranker.clone(),
        })
    }

    pub fn is_cached(&self, model_name: &str) -> bool {
        self.model_dir(model_name).join(MODEL_FILE_NAME).is_file()
            && self
                .model_dir(model_name)
                .join(TOKENIZER_FILE_NAME)
                .is_file()
    }

    pub fn model_dir(&self, model_name: &str) -> PathBuf {
        self.models_dir.join(model_name)
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
