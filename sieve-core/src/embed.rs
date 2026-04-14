use std::fmt;
use std::path::Path;
use std::sync::Mutex;

use ndarray::{s, Array2, ArrayViewD, Axis};
use ort::{session::Session, value::TensorRef};
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

use crate::{Result, SieveError};

const MAX_TOKENS: usize = 512;
const QUERY_PREFIX: &str = "Represent this sentence for searching relevant passages: ";
const DEFAULT_DIMENSION: usize = 384;

pub struct Embedder {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    dimension: usize,
}

impl fmt::Debug for Embedder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Embedder")
            .field("dimension", &self.dimension)
            .finish_non_exhaustive()
    }
}

impl Embedder {
    pub fn load(model_path: &Path, tokenizer_path: &Path) -> Result<Self> {
        let session = Session::builder()
            .map_err(map_external_error)?
            .commit_from_file(model_path)
            .map_err(map_external_error)?;
        let mut tokenizer = Tokenizer::from_file(tokenizer_path).map_err(map_external_error)?;
        tokenizer
            .with_padding(Some(PaddingParams {
                strategy: PaddingStrategy::BatchLongest,
                ..Default::default()
            }))
            .with_truncation(Some(TruncationParams {
                max_length: MAX_TOKENS,
                ..Default::default()
            }))
            .map_err(map_external_error)?;
        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            dimension: DEFAULT_DIMENSION,
        })
    }

    pub fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        self.embed_internal(&[text], false)
            .map(|mut batch| batch.remove(0))
    }

    pub fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        self.embed_internal(&[text], true)
            .map(|mut batch| batch.remove(0))
    }

    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.embed_internal(texts, false)
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    fn embed_internal(&self, texts: &[&str], is_query: bool) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let prepared: Vec<String> = texts
            .iter()
            .map(|text| {
                if is_query {
                    format!("{QUERY_PREFIX}{text}")
                } else {
                    (*text).to_string()
                }
            })
            .collect();

        let encodings = self
            .tokenizer
            .encode_batch(prepared.clone(), true)
            .map_err(map_external_error)?;
        let batch = encodings.len();
        let seq_len = encodings
            .iter()
            .map(|encoding| encoding.len())
            .max()
            .unwrap_or(0)
            .min(MAX_TOKENS);
        if seq_len == 0 {
            return Ok(vec![normalize_vector(vec![0.0; self.dimension]); batch]);
        }

        let mut input_ids = Array2::<i64>::zeros((batch, seq_len));
        let mut attention_mask = Array2::<i64>::zeros((batch, seq_len));
        let mut token_type_ids = Array2::<i64>::zeros((batch, seq_len));
        for (row, encoding) in encodings.iter().enumerate() {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();
            let types = encoding.get_type_ids();
            for col in 0..seq_len {
                input_ids[(row, col)] = *ids.get(col).unwrap_or(&0) as i64;
                attention_mask[(row, col)] = *mask.get(col).unwrap_or(&0) as i64;
                token_type_ids[(row, col)] = *types.get(col).unwrap_or(&0) as i64;
            }
        }

        let mut session = self.session.lock().map_err(|_| SieveError::LockPoisoned)?;
        let outputs = session
            .run(ort::inputs![
                TensorRef::from_array_view(&input_ids).map_err(map_external_error)?,
                TensorRef::from_array_view(&attention_mask).map_err(map_external_error)?,
                TensorRef::from_array_view(&token_type_ids).map_err(map_external_error)?
            ])
            .map_err(map_external_error)?;
        let output: ArrayViewD<'_, f32> =
            outputs[0].try_extract_array().map_err(map_external_error)?;
        let vectors = if output.ndim() == 2 {
            output
                .axis_iter(Axis(0))
                .map(|row| normalize_vector(row.iter().copied().collect::<Vec<f32>>()))
                .collect()
        } else if output.ndim() == 3 {
            let mut pooled = Vec::with_capacity(batch);
            for row in 0..batch {
                let cls = output.slice(s![row, 0, ..]);
                pooled.push(normalize_vector(cls.iter().copied().collect::<Vec<f32>>()));
            }
            pooled
        } else {
            return Err(SieveError::Io(std::io::Error::other(format!(
                "unsupported embedding output rank: {}",
                output.ndim()
            ))));
        };
        Ok(vectors)
    }
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
