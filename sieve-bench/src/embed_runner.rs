use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Instant;

use anyhow::Result;
use ignore::WalkBuilder;
use sieve_core::chunk::SlidingChunker;
use sieve_core::default_sieve_data_dir;
use sieve_core::embed::Embedder;
use sieve_core::model::ModelManager;

use crate::runner::Runner;
use crate::types::{Episode, Hit};

type FileVector = (PathBuf, Vec<f32>);
type SharedVectors = Arc<Mutex<Vec<FileVector>>>;

#[derive(Default)]
pub struct EmbedKnnRunner {
    stable_vectors: Vec<FileVector>,
    fresh_vectors: SharedVectors,
    background: Option<JoinHandle<()>>,
}

impl Runner for EmbedKnnRunner {
    fn name(&self) -> &'static str {
        "embed-knn"
    }

    fn prepare_stable(&mut self, ep: &Episode) -> Result<()> {
        self.stable_vectors.clear();
        let embedder = load_embedder()?;
        for file in walk_files(&ep.stable_root)? {
            let rel = file
                .strip_prefix(&ep.corpus_root)
                .unwrap_or(&file)
                .to_path_buf();
            let text = std::fs::read_to_string(&file)?;
            for chunk in SlidingChunker::default().chunk_entry(0, &text) {
                let vec = embedder.embed_one(&chunk.text)?;
                self.stable_vectors.push((rel.clone(), vec));
            }
        }
        Ok(())
    }

    fn begin_fresh_arrival(&mut self, ep: &Episode) -> Result<()> {
        self.fresh_vectors = Arc::new(Mutex::new(Vec::new()));
        let target = self.fresh_vectors.clone();
        let file = ep.fresh_live_root.clone();
        let corpus_root = ep.corpus_root.clone();
        self.background = Some(thread::spawn(move || {
            let Ok(embedder) = load_embedder() else {
                return;
            };
            let Ok(text) = std::fs::read_to_string(&file) else {
                return;
            };
            let rel = file
                .strip_prefix(&corpus_root)
                .unwrap_or(&file)
                .to_path_buf();
            let mut local = Vec::new();
            for chunk in SlidingChunker::default().chunk_entry(0, &text) {
                if let Ok(vec) = embedder.embed_one(&chunk.text) {
                    local.push((rel.clone(), vec));
                }
            }
            if let Ok(mut guard) = target.lock() {
                guard.extend(local);
            }
        }));
        Ok(())
    }

    fn search_at_deadline(
        &mut self,
        ep: &Episode,
        deadline: std::time::Duration,
        k: usize,
    ) -> Result<Vec<Hit>> {
        let started = Instant::now();
        let embedder = load_embedder()?;
        let query = embedder.embed_query(&ep.query)?;
        let mut all = self.stable_vectors.clone();
        if !deadline.is_zero() {
            if let Ok(guard) = self.fresh_vectors.lock() {
                all.extend(guard.iter().cloned());
            }
        }
        let mut best = std::collections::BTreeMap::<PathBuf, f32>::new();
        for (path, vec) in all {
            let score = cosine(&query, &vec);
            match best.get_mut(&path) {
                Some(existing) => {
                    if score > *existing {
                        *existing = score;
                    }
                }
                None => {
                    best.insert(path, score);
                }
            }
        }
        let mut hits: Vec<_> = best.into_iter().collect();
        hits.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        Ok(hits
            .into_iter()
            .take(k)
            .enumerate()
            .map(|(idx, (path, score))| Hit {
                path,
                score,
                rank: idx + 1,
                latency: started.elapsed(),
            })
            .collect())
    }

    fn cleanup(&mut self) -> Result<()> {
        if let Some(handle) = self.background.take() {
            let _ = handle.join();
        }
        self.stable_vectors.clear();
        if let Ok(mut guard) = self.fresh_vectors.lock() {
            guard.clear();
        }
        Ok(())
    }
}

fn load_embedder() -> Result<Embedder> {
    let manager = ModelManager::new(&default_sieve_data_dir());
    let handle = manager.ensure_dense_model()?;
    Ok(Embedder::load(&handle.model_path, &handle.tokenizer_path)?)
}

fn walk_files(root: &PathBuf) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for entry in WalkBuilder::new(root)
        .hidden(false)
        .git_ignore(false)
        .git_exclude(false)
        .git_global(false)
        .build()
    {
        let entry = entry?;
        if entry.file_type().is_some_and(|ft| ft.is_file())
            && !entry.path().components().any(|c| c.as_os_str() == ".sieve")
        {
            out.push(entry.path().to_path_buf());
        }
    }
    Ok(out)
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na <= 1e-12 || nb <= 1e-12 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}
