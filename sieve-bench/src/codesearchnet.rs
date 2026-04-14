use std::collections::BTreeSet;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::Deserialize;

use crate::eval::T_STEADY_DEADLINE;
use crate::types::Episode;

const DATASET_NAME: &str = "claudios/code_search_net";
const CONFIG_NAME: &str = "python";
const SPLIT_NAME: &str = "test";
const PAGE_SIZE: usize = 100;

#[derive(Debug, Clone)]
struct CodeExample {
    code: String,
    docstring: String,
}

#[derive(Debug)]
pub struct CodeSearchNetTrack {
    pub base_dir: PathBuf,
    pub stable_dir: PathBuf,
    pub fresh_stage_dir: PathBuf,
    pub episodes: Vec<Episode>,
}

#[derive(Debug, Deserialize)]
struct HfRows {
    rows: Vec<HfRow>,
}

#[derive(Debug, Deserialize)]
struct HfRow {
    row: HfExample,
}

#[derive(Debug, Deserialize)]
struct HfExample {
    #[serde(default)]
    func_code_string: String,
    #[serde(default)]
    func_documentation_string: String,
}

impl CodeSearchNetTrack {
    pub fn setup(cache_dir: &Path, n_stable: usize, n_fresh: usize) -> Result<Self> {
        fs::create_dir_all(cache_dir)?;
        let dataset_cache_dir = cache_dir.join("dataset-cache");
        fs::create_dir_all(&dataset_cache_dir)?;
        let cache_file = dataset_cache_dir.join("codesearchnet_python_test.jsonl");
        let needed = n_stable + n_fresh;
        let examples = load_or_fetch_examples(&cache_file, needed)?;
        Self::setup_from_examples(
            &cache_dir.join("materialized"),
            &examples,
            n_stable,
            n_fresh,
        )
    }

    pub fn setup_from_examples(
        base_dir: &Path,
        examples: &[(String, String)],
        n_stable: usize,
        n_fresh: usize,
    ) -> Result<Self> {
        let needed = n_stable + n_fresh;
        anyhow::ensure!(
            examples.len() >= needed,
            "not enough examples: need {needed}, got {}",
            examples.len()
        );

        if base_dir.exists() {
            fs::remove_dir_all(base_dir).ok();
        }
        fs::create_dir_all(base_dir)?;
        let stable_dir = base_dir.join("stable-source");
        let fresh_stage_dir = base_dir.join("fresh-source");
        fs::create_dir_all(&stable_dir)?;
        fs::create_dir_all(&fresh_stage_dir)?;

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut items: Vec<CodeExample> = examples
            .iter()
            .map(|(code, doc)| CodeExample {
                code: code.clone(),
                docstring: doc.clone(),
            })
            .collect();
        items.shuffle(&mut rng);

        let stable_items = &items[..n_stable];
        let fresh_items = &items[n_stable..needed];

        for (idx, item) in stable_items.iter().enumerate() {
            fs::write(stable_dir.join(format!("stable_{idx:05}.py")), &item.code)?;
        }
        for (idx, item) in fresh_items.iter().enumerate() {
            fs::write(
                fresh_stage_dir.join(format!("fresh_{idx:05}.py")),
                &item.code,
            )?;
        }

        let mut episodes = Vec::with_capacity(fresh_items.len());
        for (idx, item) in fresh_items.iter().enumerate() {
            let episode_root = base_dir.join("episodes").join(format!("ep_{idx:05}"));
            let corpus_root = episode_root.join("corpus");
            let stage_root = episode_root.join("stage");
            fs::create_dir_all(&corpus_root)?;
            fs::create_dir_all(&stage_root)?;
            for stable in fs::read_dir(&stable_dir)? {
                let stable = stable?;
                let dst = corpus_root.join(stable.file_name());
                hard_link_or_copy(&stable.path(), &dst)?;
            }
            let fresh_name = format!("fresh_{idx:05}.py");
            let source_fresh = fresh_stage_dir.join(&fresh_name);
            let staged_fresh = stage_root.join(&fresh_name);
            hard_link_or_copy(&source_fresh, &staged_fresh)?;
            episodes.push(Episode {
                id: format!("codesearchnet-python-{idx:05}"),
                query: item.docstring.clone(),
                corpus_root: corpus_root.clone(),
                stable_root: corpus_root.clone(),
                fresh_stage_root: staged_fresh,
                fresh_live_root: corpus_root.join(&fresh_name),
                relevant_paths: BTreeSet::from([PathBuf::from(fresh_name)]),
                deadlines: vec![
                    Duration::from_millis(0),
                    Duration::from_millis(100),
                    Duration::from_millis(500),
                    Duration::from_secs(1),
                    Duration::from_secs(5),
                    T_STEADY_DEADLINE,
                ],
            });
        }

        Ok(Self {
            base_dir: base_dir.to_path_buf(),
            stable_dir,
            fresh_stage_dir,
            episodes,
        })
    }

    pub fn episodes(&self) -> &[Episode] {
        &self.episodes
    }

    pub fn stable_dir(&self) -> &Path {
        &self.stable_dir
    }

    pub fn fresh_stage_dir(&self) -> &Path {
        &self.fresh_stage_dir
    }
}

fn load_or_fetch_examples(cache_file: &Path, needed: usize) -> Result<Vec<(String, String)>> {
    if cache_file.exists() {
        let cached = read_jsonl_examples(cache_file)?;
        if cached.len() >= needed {
            return Ok(cached);
        }
    }

    let client = reqwest::blocking::Client::builder()
        .user_agent("sieve-bench/0.1")
        .build()?;
    let mut out = Vec::new();
    let mut offset = 0usize;
    while out.len() < needed {
        let url = format!(
            "https://datasets-server.huggingface.co/rows?dataset={}&config={}&split={}&offset={}&length={}",
            DATASET_NAME, CONFIG_NAME, SPLIT_NAME, offset, PAGE_SIZE
        );
        let page: HfRows = client
            .get(&url)
            .send()
            .with_context(|| format!("failed GET {url}"))?
            .error_for_status()
            .with_context(|| format!("bad status for {url}"))?
            .json()
            .with_context(|| format!("failed to decode rows response from {url}"))?;
        if page.rows.is_empty() {
            break;
        }
        for row in page.rows {
            let code = row.row.func_code_string.trim().to_string();
            let doc = row.row.func_documentation_string.trim().to_string();
            if code.is_empty() || doc.is_empty() {
                continue;
            }
            out.push((code, doc));
            if out.len() >= needed {
                break;
            }
        }
        offset += PAGE_SIZE;
    }
    anyhow::ensure!(
        out.len() >= needed,
        "dataset fetch returned only {} usable examples, need {needed}",
        out.len()
    );
    write_jsonl_examples(cache_file, &out)?;
    Ok(out)
}

fn read_jsonl_examples(path: &Path) -> Result<Vec<(String, String)>> {
    let text = fs::read_to_string(path)?;
    let mut out = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(line)?;
        let code = value
            .get("code")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let doc = value
            .get("docstring")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        if !code.is_empty() && !doc.is_empty() {
            out.push((code, doc));
        }
    }
    Ok(out)
}

fn write_jsonl_examples(path: &Path, examples: &[(String, String)]) -> Result<()> {
    let file = fs::File::create(path)?;
    let mut writer = BufWriter::new(file);
    for (code, doc) in examples {
        let value = serde_json::json!({"code": code, "docstring": doc});
        writeln!(writer, "{}", serde_json::to_string(&value)?)?;
    }
    Ok(())
}

fn hard_link_or_copy(src: &Path, dst: &Path) -> Result<()> {
    if let Some(parent) = dst.parent() {
        fs::create_dir_all(parent)?;
    }
    match fs::hard_link(src, dst) {
        Ok(()) => Ok(()),
        Err(_) => {
            fs::copy(src, dst)?;
            Ok(())
        }
    }
}
