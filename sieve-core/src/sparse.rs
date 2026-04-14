use std::path::Path;
use std::sync::Mutex;

use ndarray::{Array2, ArrayViewD, Axis};
use ort::{session::Session, value::TensorRef};
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

const MAX_TOKENS: usize = 256;
const SPLADE_VOCAB_SIZE: usize = 30_522;

#[derive(Debug)]
pub struct SpladeEncoder {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    vocab_size: usize,
    vocab: Vec<String>,
}

impl SpladeEncoder {
    pub fn load(model_path: &Path, tokenizer_path: &Path) -> crate::Result<Self> {
        let session = Session::builder()
            .map_err(map_external_error)?
            .commit_from_file(model_path)
            .map_err(map_external_error)?;
        let mut tokenizer = Tokenizer::from_file(tokenizer_path).map_err(map_external_error)?;
        tokenizer
            .with_padding(Some(PaddingParams {
                strategy: PaddingStrategy::Fixed(MAX_TOKENS),
                ..Default::default()
            }))
            .with_truncation(Some(TruncationParams {
                max_length: MAX_TOKENS,
                ..Default::default()
            }))
            .map_err(map_external_error)?;

        let vocab_map = tokenizer.get_vocab(true);
        let vocab_size = vocab_map
            .values()
            .max()
            .map(|id| *id as usize + 1)
            .unwrap_or(SPLADE_VOCAB_SIZE)
            .max(SPLADE_VOCAB_SIZE);
        let mut vocab = vec![String::new(); vocab_size];
        for (piece, id) in vocab_map {
            let slot = id as usize;
            if slot < vocab.len() {
                vocab[slot] = piece;
            }
        }

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            vocab_size,
            vocab,
        })
    }

    pub fn encode(&self, text: &str) -> crate::Result<Vec<(u32, f32)>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(map_external_error)?;
        let seq_len = encoding.len().min(MAX_TOKENS);
        if seq_len == 0 {
            return Ok(Vec::new());
        }

        let mut input_ids = Array2::<i64>::zeros((1, MAX_TOKENS));
        let mut attention_mask = Array2::<i64>::zeros((1, MAX_TOKENS));
        let mut token_type_ids = Array2::<i64>::zeros((1, MAX_TOKENS));
        for col in 0..MAX_TOKENS {
            input_ids[(0, col)] = *encoding.get_ids().get(col).unwrap_or(&0) as i64;
            attention_mask[(0, col)] = *encoding.get_attention_mask().get(col).unwrap_or(&0) as i64;
            token_type_ids[(0, col)] = *encoding.get_type_ids().get(col).unwrap_or(&0) as i64;
        }

        let mut session = self
            .session
            .lock()
            .map_err(|_| crate::SieveError::LockPoisoned)?;
        let outputs = match session.inputs().len() {
            2 => session
                .run(ort::inputs![
                    TensorRef::from_array_view(&input_ids).map_err(map_external_error)?,
                    TensorRef::from_array_view(&attention_mask).map_err(map_external_error)?
                ])
                .map_err(map_external_error)?,
            3 => session
                .run(ort::inputs![
                    TensorRef::from_array_view(&input_ids).map_err(map_external_error)?,
                    TensorRef::from_array_view(&attention_mask).map_err(map_external_error)?,
                    TensorRef::from_array_view(&token_type_ids).map_err(map_external_error)?
                ])
                .map_err(map_external_error)?,
            count => {
                return Err(crate::SieveError::Io(std::io::Error::other(format!(
                    "unsupported SPLADE input count: {count}"
                ))))
            }
        };
        let output: ArrayViewD<'_, f32> =
            outputs[0].try_extract_array().map_err(map_external_error)?;
        if output.ndim() != 3 {
            return Err(crate::SieveError::Io(std::io::Error::other(format!(
                "unsupported SPLADE output rank: {}",
                output.ndim()
            ))));
        }
        let shape = output.shape();
        if shape[0] != 1 {
            return Err(crate::SieveError::Io(std::io::Error::other(format!(
                "unsupported SPLADE batch dimension: {}",
                shape[0]
            ))));
        }

        let mut pooled = vec![0.0f32; shape[2]];
        for row in output.index_axis(Axis(0), 0).axis_iter(Axis(0)) {
            for (index, logit) in row.iter().copied().enumerate() {
                let activation = (1.0 + logit.max(0.0)).ln();
                if activation > pooled[index] {
                    pooled[index] = activation;
                }
            }
        }

        let mut weighted = pooled
            .into_iter()
            .enumerate()
            .filter_map(|(index, weight)| {
                if weight > 0.0 {
                    Some((index as u32, weight))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        weighted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(weighted)
    }

    pub fn vocab_piece(&self, vocab_id: u32) -> Option<&str> {
        self.vocab.get(vocab_id as usize).map(String::as_str)
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

fn map_external_error(error: impl std::fmt::Display) -> crate::SieveError {
    crate::SieveError::Io(std::io::Error::other(error.to_string()))
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::SpladeEncoder;

    #[test]
    #[ignore]
    fn test_splade_encoder_loads_and_encodes() {
        let model_path = Path::new("/home/burba/.sieve/models/splade/splade.onnx");
        let tokenizer_path =
            Path::new("/home/burba/.sieve/models/splade/splade-tokenizer/tokenizer.json");
        let encoder = SpladeEncoder::load(model_path, tokenizer_path).unwrap();
        let encoded = encoder.encode("error handling").unwrap();
        assert!(!encoded.is_empty());
    }

    #[test]
    #[ignore]
    fn test_splade_activation_shape() {
        let model_path = Path::new("/home/burba/.sieve/models/splade/splade.onnx");
        let tokenizer_path =
            Path::new("/home/burba/.sieve/models/splade/splade-tokenizer/tokenizer.json");
        let encoder = SpladeEncoder::load(model_path, tokenizer_path).unwrap();
        let encoded = encoder.encode("failure retry").unwrap();
        assert!(encoded
            .iter()
            .all(|(id, weight)| *id < 30_522 && *weight >= 0.0));
    }
}
