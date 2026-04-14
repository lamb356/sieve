use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use ignore::WalkBuilder;
use sieve_core::lexical::build_pending_shards;
use sieve_core::{Index, SearchOptions, SearchResult};

use crate::runner::Runner;
use crate::types::{Episode, Hit};

#[derive(Default)]
pub struct SieveRunner {
    index: Option<Index>,
}

impl Runner for SieveRunner {
    fn name(&self) -> &'static str {
        "sieve"
    }

    fn prepare_stable(&mut self, ep: &Episode) -> Result<()> {
        let index_root = ep.corpus_root.join(".sieve");
        if index_root.exists() {
            fs::remove_dir_all(&index_root).ok();
        }
        let index = Index::open_or_create(&index_root)?;
        index_directory(&index, &ep.stable_root, &ep.corpus_root)?;
        build_pending_shards(&index)?;
        let _ = index.embed_pending(32)?;
        self.index = Some(index);
        Ok(())
    }

    fn begin_fresh_arrival(&mut self, ep: &Episode) -> Result<()> {
        let index = self.index.as_ref().context("sieve runner index missing")?;
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

    fn search_at_deadline(
        &mut self,
        ep: &Episode,
        _deadline: std::time::Duration,
        k: usize,
    ) -> Result<Vec<Hit>> {
        let index = self.index.as_ref().context("sieve runner index missing")?;
        let started = Instant::now();
        let results = index.search(&ep.query, SearchOptions { top_k: Some(k * 8) })?;
        let hits = collapse_results(&results, k, started.elapsed());
        Ok(hits)
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

fn collapse_results(results: &[SearchResult], k: usize, latency: std::time::Duration) -> Vec<Hit> {
    let mut best: BTreeMap<PathBuf, f32> = BTreeMap::new();
    for result in results {
        let path = PathBuf::from(&result.source_path);
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
