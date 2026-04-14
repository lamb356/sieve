use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use anyhow::Result;
use sieve_core::df_prior::static_df_frac;

use crate::runner::Runner;
use crate::types::{Episode, Hit};

#[derive(Default)]
pub struct RipgrepRunner {
    available: bool,
}

impl RipgrepRunner {
    pub fn new() -> Self {
        Self {
            available: Command::new("rg").arg("--version").output().is_ok(),
        }
    }
}

impl Runner for RipgrepRunner {
    fn name(&self) -> &'static str {
        "ripgrep"
    }

    fn prepare_stable(&mut self, _ep: &Episode) -> Result<()> {
        Ok(())
    }

    fn begin_fresh_arrival(&mut self, _ep: &Episode) -> Result<()> {
        Ok(())
    }

    fn search_at_deadline(
        &mut self,
        ep: &Episode,
        _deadline: std::time::Duration,
        k: usize,
    ) -> Result<Vec<Hit>> {
        if !self.available {
            tracing::warn!("ripgrep not installed; skipping runner");
            return Ok(Vec::new());
        }
        let started = Instant::now();
        let tokens = tokenize_query(&ep.query);
        let mut scores: BTreeMap<PathBuf, f32> = BTreeMap::new();
        let mut matched: BTreeMap<PathBuf, BTreeSet<String>> = BTreeMap::new();
        for token in tokens {
            let output = Command::new("rg")
                .arg("-l")
                .arg("-i")
                .arg("-g")
                .arg("*.py")
                .arg(&token)
                .arg(&ep.corpus_root)
                .output()?;
            if !output.status.success() && output.status.code() != Some(1) {
                continue;
            }
            let idf = prior_idf(&token);
            for line in String::from_utf8_lossy(&output.stdout).lines() {
                let path = PathBuf::from(line.trim());
                if path.as_os_str().is_empty() {
                    continue;
                }
                let rel = normalize_rel(&ep.corpus_root, &path);
                let already = matched.entry(rel.clone()).or_default();
                if already.insert(token.clone()) {
                    *scores.entry(rel).or_default() += idf;
                }
            }
        }
        let mut ranked: Vec<_> = scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        Ok(ranked
            .into_iter()
            .enumerate()
            .map(|(idx, (path, score))| Hit {
                path,
                score,
                rank: idx + 1,
                latency: started.elapsed(),
            })
            .take(k)
            .collect())
    }

    fn cleanup(&mut self) -> Result<()> {
        Ok(())
    }
}

pub fn tokenize_query(query: &str) -> Vec<String> {
    let stop = [
        "the", "and", "for", "with", "that", "this", "how", "between", "into", "from", "your",
        "you", "are", "not", "all", "any", "can", "use", "using", "only", "does", "doesnt", "what",
        "when", "where", "why", "which", "have", "has", "had", "was", "were", "to", "of", "in",
        "on", "at", "by", "an", "or",
    ];
    let stop: BTreeSet<_> = stop.into_iter().collect();
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in query.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            cur.push(ch.to_ascii_lowercase());
        } else if !cur.is_empty() {
            if cur.len() >= 3 && !stop.contains(cur.as_str()) {
                out.push(std::mem::take(&mut cur));
            } else {
                cur.clear();
            }
        }
    }
    if !cur.is_empty() && cur.len() >= 3 && !stop.contains(cur.as_str()) {
        out.push(cur);
    }
    out
}

fn prior_idf(token: &str) -> f32 {
    let frac = static_df_frac(token).clamp(1e-6, 0.999999);
    ((1.0 + 1.0) / (frac + 0.5)).ln() + 1.0
}

fn normalize_rel(root: &Path, path: &Path) -> PathBuf {
    path.strip_prefix(root).unwrap_or(path).to_path_buf()
}
