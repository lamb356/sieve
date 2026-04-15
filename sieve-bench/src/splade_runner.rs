use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use ignore::WalkBuilder;
use sieve_core::aliases::AliasLexicon;
use sieve_core::fusion::ResultSource;
use sieve_core::lexical::build_pending_shards;
use sieve_core::model::{ModelManager, DEFAULT_SPARSE_MODEL_NAME};
use sieve_core::semantic_query::{compile_semantic_query_with_options, SemanticCompileOptions};
use sieve_core::{default_sieve_data_dir, Index, SearchOptions};

use crate::runner::Runner;
use crate::types::{Episode, Hit};

pub struct SpladeTantivyRunner {
    index: Option<Index>,
    sparse: Option<Arc<sieve_core::sparse::SpladeEncoder>>,
    aliases: AliasLexicon,
}

impl Default for SpladeTantivyRunner {
    fn default() -> Self {
        Self {
            index: None,
            sparse: None,
            aliases: AliasLexicon::built_in(),
        }
    }
}

impl SpladeTantivyRunner {
    fn load_sparse(&mut self) -> Result<Option<Arc<sieve_core::sparse::SpladeEncoder>>> {
        if self.sparse.is_some() {
            return Ok(self.sparse.clone());
        }
        let manager = ModelManager::new(&default_sieve_data_dir());
        if !manager.is_cached(DEFAULT_SPARSE_MODEL_NAME) {
            return Ok(None);
        }
        let handle = manager.ensure_sparse_model()?;
        let encoder =
            sieve_core::sparse::SpladeEncoder::load(&handle.model_path, &handle.tokenizer_path)
                .ok()
                .map(Arc::new);
        self.sparse = encoder.clone();
        Ok(encoder)
    }
}

impl Runner for SpladeTantivyRunner {
    fn name(&self) -> &'static str {
        "splade-bm25"
    }

    fn prepare_stable(&mut self, ep: &Episode) -> Result<()> {
        let index_root = ep.corpus_root.join(".sieve");
        if index_root.exists() {
            fs::remove_dir_all(&index_root).ok();
        }
        let index = Index::open_or_create(&index_root)?;
        index_directory(&index, &ep.stable_root, &ep.corpus_root)?;
        build_pending_shards(&index)?;
        self.index = Some(index);
        let _ = self.load_sparse()?;
        Ok(())
    }

    fn begin_fresh_arrival(&mut self, ep: &Episode) -> Result<()> {
        let index = self.index.as_ref().context("splade runner index missing")?;
        let rel = ep
            .fresh_live_root
            .strip_prefix(&ep.corpus_root)
            .unwrap_or(&ep.fresh_live_root)
            .to_string_lossy()
            .to_string();
        let text = fs::read_to_string(&ep.fresh_live_root)?;
        index.add_text(rel, text)?;
        index.save_manifest()?;
        Ok(())
    }

    fn wait_for_steady_state(&mut self, _ep: &Episode) -> Result<()> {
        let index = self.index.as_ref().context("splade runner index missing")?;
        let _ = build_pending_shards(index)?;
        Ok(())
    }

    fn search_at_deadline(
        &mut self,
        ep: &Episode,
        _deadline: Duration,
        k: usize,
    ) -> Result<Vec<Hit>> {
        let Some(sparse) = self.load_sparse()? else {
            return Ok(Vec::new());
        };
        let Some(index) = self.index.as_ref() else {
            return Ok(Vec::new());
        };
        let query = compile_semantic_query_with_options(
            &ep.query,
            sparse.as_ref(),
            &self.aliases,
            SemanticCompileOptions::default(),
            sieve_core::semantic_query::ContentType::Code,
        )?;
        let started = Instant::now();
        let outcome = index.search_semantic_query(
            &query,
            SearchOptions {
                top_k: Some(k * 8),
                ..Default::default()
            },
        )?;
        let scored = outcome
            .source_sets
            .into_iter()
            .find(|set| set.source == ResultSource::SpladeBm25)
            .map(|set| set.results)
            .unwrap_or_default();
        Ok(collapse_scored(scored, k, started.elapsed()))
    }

    fn cleanup(&mut self) -> Result<()> {
        self.index = None;
        Ok(())
    }
}

fn index_directory(index: &Index, source_root: &Path, corpus_root: &Path) -> Result<()> {
    let mut seen_paths = std::collections::HashSet::new();
    let index_root = corpus_root.join(".sieve");
    for entry in WalkBuilder::new(source_root)
        .hidden(false)
        .git_ignore(false)
        .git_exclude(false)
        .git_global(false)
        .build()
    {
        let entry = entry?;
        let path = entry.path();
        if path.starts_with(&index_root) || !entry.file_type().is_some_and(|ft| ft.is_file()) {
            continue;
        }
        let rel = path
            .strip_prefix(corpus_root)
            .unwrap_or(path)
            .to_string_lossy()
            .to_string();
        seen_paths.insert(rel.clone());
        let text = fs::read_to_string(path)?;
        index.add_text(rel, text)?;
    }
    index.prune_manifest_to_paths(&seen_paths)?;
    index.save_manifest()?;
    Ok(())
}

fn collapse_scored(
    results: Vec<sieve_core::fusion::ScoredResult>,
    k: usize,
    latency: Duration,
) -> Vec<Hit> {
    let mut best: BTreeMap<PathBuf, f32> = BTreeMap::new();
    for result in results {
        let path = PathBuf::from(result.source_path);
        let score = result.score as f32;
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
    let mut pairs: Vec<_> = best.into_iter().collect();
    pairs.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    pairs
        .into_iter()
        .take(k)
        .enumerate()
        .map(|(idx, (path, score))| Hit {
            path,
            score,
            rank: idx + 1,
            latency,
        })
        .collect()
}
