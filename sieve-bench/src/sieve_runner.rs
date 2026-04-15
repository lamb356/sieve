use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use ignore::WalkBuilder;
use sieve_core::lexical::build_pending_shards;
use sieve_core::{Index, SearchOptions, SearchResult};
use tracing::debug;

use crate::runner::Runner;
use crate::types::{Episode, Hit};

#[derive(Debug, Clone, Copy)]
pub enum SieveMode {
    Full,
    ScanOnly,
    RandomExpansion,
}

pub struct SieveRunner {
    mode: SieveMode,
    index: Option<Index>,
}

impl Default for SieveRunner {
    fn default() -> Self {
        Self::full()
    }
}

impl SieveRunner {
    pub fn full() -> Self {
        Self {
            mode: SieveMode::Full,
            index: None,
        }
    }

    pub fn scan_only() -> Self {
        Self {
            mode: SieveMode::ScanOnly,
            index: None,
        }
    }

    pub fn random_expansion() -> Self {
        Self {
            mode: SieveMode::RandomExpansion,
            index: None,
        }
    }

    fn search_options(&self, k: usize) -> SearchOptions {
        SearchOptions {
            top_k: Some(k * 8),
            fresh_only: matches!(self.mode, SieveMode::ScanOnly | SieveMode::RandomExpansion),
            random_expansion: matches!(self.mode, SieveMode::RandomExpansion),
            ..Default::default()
        }
    }

    fn search_options_for_deadline(&self, k: usize, deadline: Duration) -> SearchOptions {
        let mut options = self.search_options(k);
        if matches!(self.mode, SieveMode::Full) && !crate::eval::is_steady_deadline(deadline) {
            options.allow_delta_fallback = false;
        }
        options
    }

    fn materializes_fresh_on_steady_state(&self) -> bool {
        matches!(self.mode, SieveMode::Full)
    }
}

impl Runner for SieveRunner {
    fn name(&self) -> &'static str {
        match self.mode {
            SieveMode::Full => "sieve",
            SieveMode::ScanOnly => "sieve-scan",
            SieveMode::RandomExpansion => "sieve-random",
        }
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
        index.warm_search_models(false)?;
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

    fn wait_for_steady_state(&mut self, _ep: &Episode) -> Result<()> {
        if !self.materializes_fresh_on_steady_state() {
            return Ok(());
        }
        let index = self.index.as_ref().context("sieve runner index missing")?;
        let _ = build_pending_shards(index)?;
        let _ = index.embed_pending(32)?;
        Ok(())
    }

    fn search_at_deadline(
        &mut self,
        ep: &Episode,
        deadline: Duration,
        k: usize,
    ) -> Result<Vec<Hit>> {
        let index = self.index.as_ref().context("sieve runner index missing")?;
        let started = Instant::now();
        let outcome =
            index.search_with_outcome(&ep.query, self.search_options_for_deadline(k, deadline))?;
        if let Some(debug_info) = &outcome.debug {
            debug!(
                "sieve benchmark query timing: runner={} episode={} query={:?} mode={} splade_expand_ms={} aho_compile_ms={} semantic_scan_ms={} raw_scan_ms={} tantivy_query_ms={} dense_knn_ms={} rrf_fusion_ms={}",
                self.name(),
                ep.id,
                ep.query,
                debug_info.plan_mode,
                debug_info.timings.splade_expand.as_millis(),
                debug_info.timings.aho_compile.as_millis(),
                debug_info.timings.semantic_scan.as_millis(),
                debug_info.timings.raw_scan.as_millis(),
                debug_info.timings.tantivy_query.as_millis(),
                debug_info.timings.dense_knn.as_millis(),
                debug_info.timings.rrf_fusion.as_millis(),
            );
        }
        Ok(collapse_results(&outcome.results, k, started.elapsed()))
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

fn collapse_results(results: &[SearchResult], k: usize, latency: Duration) -> Vec<Hit> {
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

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::SieveRunner;
    use crate::eval::T_STEADY_DEADLINE;

    #[test]
    fn test_scan_only_runner_skips_index_materialization_at_steady_state() {
        let runner = SieveRunner::scan_only();
        assert!(!runner.materializes_fresh_on_steady_state());
    }

    #[test]
    fn test_random_runner_skips_index_materialization_at_steady_state() {
        let runner = SieveRunner::random_expansion();
        assert!(!runner.materializes_fresh_on_steady_state());
    }

    #[test]
    fn test_full_runner_materializes_fresh_on_steady_state() {
        let runner = SieveRunner::full();
        assert!(runner.materializes_fresh_on_steady_state());
    }

    #[test]
    fn test_full_runner_disables_delta_fallback_before_steady_state() {
        let runner = SieveRunner::full();
        let opts = runner.search_options_for_deadline(5, Duration::from_millis(500));
        assert!(!opts.allow_delta_fallback);
    }

    #[test]
    fn test_full_runner_keeps_delta_fallback_at_steady_state() {
        let runner = SieveRunner::full();
        let opts = runner.search_options_for_deadline(5, T_STEADY_DEADLINE);
        assert!(opts.allow_delta_fallback);
    }
}
