use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use ndarray::{Array1, Array2, ArrayViewD, Axis};
use ort::{session::Session, value::TensorRef};

use crate::{Result, SieveError};

pub const QUERY_SEQ_LEN: usize = 256;
pub const DOC_SEQ_LEN: usize = 2048;
const QUERY_OUTPUT_NAME: &str = "query_embedding";
const DOC_OUTPUT_NAME: &str = "doc_embedding";

static DEFAULT_EMBEDDER: Mutex<Option<Arc<Embedder>>> = Mutex::new(None);

pub struct Embedder {
    query_session: Mutex<Session>,
    doc_session: Mutex<Session>,
    dimension: usize,
    query_seq_len: usize,
    doc_seq_len: usize,
}

impl fmt::Debug for Embedder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Embedder")
            .field("dimension", &self.dimension)
            .field("query_seq_len", &self.query_seq_len)
            .field("doc_seq_len", &self.doc_seq_len)
            .finish_non_exhaustive()
    }
}

impl Embedder {
    pub fn load(query_model_path: &Path, doc_model_path: &Path) -> Result<Self> {
        if !query_model_path.is_file() {
            return Err(SieveError::Message(format!(
                "query encoder ONNX not found: {}",
                query_model_path.display()
            )));
        }
        if !doc_model_path.is_file() {
            return Err(SieveError::Message(format!(
                "document encoder ONNX not found: {}",
                doc_model_path.display()
            )));
        }

        let mut query_session = Session::builder()
            .map_err(map_external_error)?
            .commit_from_file(query_model_path)
            .map_err(map_external_error)?;
        let mut doc_session = Session::builder()
            .map_err(map_external_error)?
            .commit_from_file(doc_model_path)
            .map_err(map_external_error)?;

        let query_dimension = infer_dimension(&mut query_session, QUERY_OUTPUT_NAME)?;
        let doc_dimension = infer_dimension(&mut doc_session, DOC_OUTPUT_NAME)?;
        if query_dimension == 0 || doc_dimension == 0 {
            return Err(SieveError::Message(
                "encoder ONNX returned an empty embedding dimension".to_string(),
            ));
        }
        if query_dimension != doc_dimension {
            return Err(SieveError::Message(format!(
                "query/doc encoder dimension mismatch: query={query_dimension}, doc={doc_dimension}"
            )));
        }

        Ok(Self {
            query_session: Mutex::new(query_session),
            doc_session: Mutex::new(doc_session),
            dimension: query_dimension,
            query_seq_len: QUERY_SEQ_LEN,
            doc_seq_len: DOC_SEQ_LEN,
        })
    }

    pub fn load_default() -> Result<Self> {
        let (query_path, doc_path) = default_encoder_paths();
        Self::load(&query_path, &doc_path)
    }

    pub fn encode_query(&self, text: &str) -> Result<Vec<f32>> {
        self.run_texts(
            &self.query_session,
            &[text],
            self.query_seq_len,
            QUERY_OUTPUT_NAME,
        )
        .map(|mut batch| batch.remove(0))
    }

    pub fn encode_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        self.run_texts(&self.doc_session, &refs, self.doc_seq_len, DOC_OUTPUT_NAME)
    }

    pub fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        self.run_texts(
            &self.doc_session,
            &[text],
            self.doc_seq_len,
            DOC_OUTPUT_NAME,
        )
        .map(|mut batch| batch.remove(0))
    }

    pub fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        self.encode_query(text)
    }

    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.run_texts(&self.doc_session, texts, self.doc_seq_len, DOC_OUTPUT_NAME)
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn query_seq_len(&self) -> usize {
        self.query_seq_len
    }

    pub fn doc_seq_len(&self) -> usize {
        self.doc_seq_len
    }

    fn run_texts(
        &self,
        session: &Mutex<Session>,
        texts: &[&str],
        seq_len: usize,
        expected_output_name: &str,
    ) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let (byte_ids, lengths) = byte_batch(texts, seq_len);
        let mut session = session.lock().map_err(|_| SieveError::LockPoisoned)?;
        let outputs = session
            .run(ort::inputs![
                TensorRef::from_array_view(&byte_ids).map_err(map_external_error)?,
                TensorRef::from_array_view(&lengths).map_err(map_external_error)?
            ])
            .map_err(map_external_error)?;
        let output: ArrayViewD<'_, f32> = outputs[0]
            .try_extract_array()
            .map_err(|err| map_external_error(format!("{expected_output_name}: {err}")))?;
        extract_vectors(output, texts.len(), self.dimension)
    }
}

pub fn encode_query(text: &str) -> Result<Vec<f32>> {
    default_embedder()?.encode_query(text)
}

pub fn encode_documents(texts: &[String]) -> Result<Vec<Vec<f32>>> {
    default_embedder()?.encode_documents(texts)
}

pub fn default_encoder_paths() -> (PathBuf, PathBuf) {
    let query_path = std::env::var_os("SIEVE_ENCODER_QUERY_ONNX").map(PathBuf::from);
    let doc_path = std::env::var_os("SIEVE_ENCODER_DOC_ONNX").map(PathBuf::from);
    if let (Some(query), Some(doc)) = (query_path, doc_path) {
        return (query, doc);
    }

    let encoder_dir = std::env::var_os("SIEVE_ENCODER_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".sieve")
                .join("encoder")
        });
    (
        std::env::var_os("SIEVE_ENCODER_QUERY_ONNX")
            .map(PathBuf::from)
            .unwrap_or_else(|| encoder_dir.join("query.onnx")),
        std::env::var_os("SIEVE_ENCODER_DOC_ONNX")
            .map(PathBuf::from)
            .unwrap_or_else(|| encoder_dir.join("doc.onnx")),
    )
}

fn default_embedder() -> Result<Arc<Embedder>> {
    let mut guard = DEFAULT_EMBEDDER
        .lock()
        .map_err(|_| SieveError::LockPoisoned)?;
    if let Some(embedder) = guard.as_ref() {
        return Ok(Arc::clone(embedder));
    }
    let embedder = Arc::new(Embedder::load_default()?);
    *guard = Some(Arc::clone(&embedder));
    Ok(embedder)
}

fn infer_dimension(session: &mut Session, expected_output_name: &str) -> Result<usize> {
    let byte_ids = Array2::<i64>::zeros((1, 1));
    let lengths = Array1::<i64>::from_vec(vec![0]);
    let outputs = session
        .run(ort::inputs![
            TensorRef::from_array_view(&byte_ids).map_err(map_external_error)?,
            TensorRef::from_array_view(&lengths).map_err(map_external_error)?
        ])
        .map_err(map_external_error)?;
    let output: ArrayViewD<'_, f32> = outputs[0]
        .try_extract_array()
        .map_err(|err| map_external_error(format!("{expected_output_name}: {err}")))?;
    let shape = output.shape();
    match shape {
        [_, dim] => Ok(*dim),
        [_, _, dim] => Ok(*dim),
        other => Err(SieveError::Message(format!(
            "unsupported {expected_output_name} output shape: {other:?}"
        ))),
    }
}

fn byte_batch(texts: &[&str], seq_len: usize) -> (Array2<i64>, Array1<i64>) {
    let mut byte_ids = Array2::<i64>::zeros((texts.len(), seq_len));
    let mut lengths = Array1::<i64>::zeros(texts.len());
    for (row, text) in texts.iter().enumerate() {
        let bytes = text.as_bytes();
        let used = bytes.len().min(seq_len);
        lengths[row] = used as i64;
        for (col, byte) in bytes.iter().take(used).enumerate() {
            byte_ids[(row, col)] = i64::from(*byte);
        }
    }
    (byte_ids, lengths)
}

fn extract_vectors(
    output: ArrayViewD<'_, f32>,
    expected_batch: usize,
    expected_dimension: usize,
) -> Result<Vec<Vec<f32>>> {
    let vectors = if output.ndim() == 2 {
        output
            .axis_iter(Axis(0))
            .map(|row| normalize_vector(row.iter().copied().collect::<Vec<f32>>()))
            .collect::<Vec<_>>()
    } else if output.ndim() == 3 {
        let mut pooled = Vec::with_capacity(expected_batch);
        for row in 0..expected_batch {
            let row_view = output.index_axis(Axis(0), row);
            let cls = row_view.index_axis(Axis(0), 0);
            pooled.push(normalize_vector(cls.iter().copied().collect::<Vec<f32>>()));
        }
        pooled
    } else {
        return Err(SieveError::Message(format!(
            "unsupported embedding output rank: {}",
            output.ndim()
        )));
    };

    if vectors.len() != expected_batch {
        return Err(SieveError::Message(format!(
            "embedding batch mismatch: got {}, expected {expected_batch}",
            vectors.len()
        )));
    }
    if vectors.iter().any(|vector| {
        vector.len() != expected_dimension || vector.iter().any(|value| !value.is_finite())
    }) {
        return Err(SieveError::Message(
            "embedding output contains non-finite values or wrong dimension".to_string(),
        ));
    }
    Ok(vectors)
}

fn normalize_vector(mut vector: Vec<f32>) -> Vec<f32> {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut vector {
            *value /= norm;
        }
    }
    vector
}

fn map_external_error(error: impl std::fmt::Display) -> SieveError {
    SieveError::Io(std::io::Error::other(error.to_string()))
}
