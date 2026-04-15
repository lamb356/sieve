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
const SEMANTIC_HARD_MIN_FETCH: usize = 2048;

#[derive(Debug, Clone)]
struct CodeExample {
    code: String,
    docstring: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryMode {
    Original,
    SemanticHard,
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
        Self::setup_with_mode(cache_dir, n_stable, n_fresh, QueryMode::Original)
    }

    pub fn setup_semantic_hard(cache_dir: &Path, n_stable: usize, n_fresh: usize) -> Result<Self> {
        Self::setup_with_mode(cache_dir, n_stable, n_fresh, QueryMode::SemanticHard)
    }

    fn setup_with_mode(
        cache_dir: &Path,
        n_stable: usize,
        n_fresh: usize,
        query_mode: QueryMode,
    ) -> Result<Self> {
        fs::create_dir_all(cache_dir)?;
        let dataset_cache_dir = cache_dir.join("dataset-cache");
        fs::create_dir_all(&dataset_cache_dir)?;
        let cache_file = dataset_cache_dir.join("codesearchnet_python_test.jsonl");
        let needed = required_example_count(n_stable, n_fresh, query_mode);
        let examples = load_or_fetch_examples(&cache_file, needed)?;
        Self::setup_from_examples_with_mode(
            &cache_dir.join("materialized"),
            &examples,
            n_stable,
            n_fresh,
            query_mode,
        )
    }

    pub fn setup_from_examples(
        base_dir: &Path,
        examples: &[(String, String)],
        n_stable: usize,
        n_fresh: usize,
    ) -> Result<Self> {
        Self::setup_from_examples_with_mode(
            base_dir,
            examples,
            n_stable,
            n_fresh,
            QueryMode::Original,
        )
    }

    pub fn setup_from_examples_with_mode(
        base_dir: &Path,
        examples: &[(String, String)],
        n_stable: usize,
        n_fresh: usize,
        query_mode: QueryMode,
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
        let fresh_items: Vec<(String, &CodeExample)> = match query_mode {
            QueryMode::Original => items[n_stable..needed]
                .iter()
                .map(|item| (item.docstring.clone(), item))
                .collect(),
            QueryMode::SemanticHard => items[n_stable..]
                .iter()
                .filter_map(|item| {
                    semantic_hard_query_for_docstring(&item.docstring).map(|query| (query, item))
                })
                .take(n_fresh)
                .collect(),
        };
        anyhow::ensure!(
            fresh_items.len() >= n_fresh,
            "semantic-hard selection returned only {} usable examples, need {}",
            fresh_items.len(),
            n_fresh
        );

        for (idx, item) in stable_items.iter().enumerate() {
            fs::write(stable_dir.join(format!("stable_{idx:05}.py")), &item.code)?;
        }
        for (idx, (_query, item)) in fresh_items.iter().enumerate() {
            fs::write(
                fresh_stage_dir.join(format!("fresh_{idx:05}.py")),
                &item.code,
            )?;
        }

        let mut episodes = Vec::with_capacity(fresh_items.len());
        for (idx, (query, _item)) in fresh_items.iter().enumerate() {
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
                query: query.clone(),
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

fn required_example_count(n_stable: usize, n_fresh: usize, query_mode: QueryMode) -> usize {
    match query_mode {
        QueryMode::Original => n_stable + n_fresh,
        QueryMode::SemanticHard => (n_stable + n_fresh).max(SEMANTIC_HARD_MIN_FETCH),
    }
}

fn load_or_fetch_examples(cache_file: &Path, needed: usize) -> Result<Vec<(String, String)>> {
    if cache_file.exists() {
        let cached = read_jsonl_examples(cache_file)?;
        if cached.len() >= needed {
            return Ok(cached.into_iter().take(needed).collect());
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

fn semantic_hard_query_for_docstring(docstring: &str) -> Option<String> {
    let normalized = docstring.split_whitespace().collect::<Vec<_>>().join(" ");
    let trimmed = normalized.trim();
    let query = if trimmed.starts_with("Extracts video ID from URL.") {
        "recover clip identifier from address"
    } else if trimmed
        .starts_with("List all objects from the bucket with the give string prefix in name")
    {
        "show stored entries whose keys start with text"
    } else if trimmed.starts_with("Returns a cassandra Session object") {
        "open database client handle"
    } else if trimmed.starts_with(
        "Takes a cursor, and writes the BigQuery schema for the results to a local file system.",
    ) {
        "store warehouse column blueprint on disk"
    } else if trimmed.starts_with("Executes the sql and returns a set of records.") {
        "retrieve tuples from relational backend"
    } else if trimmed.starts_with("Call the SparkSqlHook to run the provided sql query") {
        "launch warehouse statement via cluster adapter"
    } else if trimmed.starts_with("Establish a connection to druid broker.") {
        "open analytics service channel"
    } else if trimmed.starts_with("generate HTML div") {
        "build markup wrapper block"
    } else if trimmed.starts_with("Create X-axis") {
        "define horizontal plotting guide"
    } else if trimmed.starts_with("Print a log message to standard error.") {
        "emit diagnostic note to err stream"
    } else if trimmed.starts_with("Visualizes a qualitative analysis of a given model.") {
        "display example reconstructions for variational system"
    } else {
        return None;
    };
    Some(query.to_string())
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::{
        required_example_count, semantic_hard_query_for_docstring, CodeSearchNetTrack, QueryMode,
        SEMANTIC_HARD_MIN_FETCH,
    };

    #[test]
    fn test_required_example_count_preserves_exact_need_for_original_mode() {
        assert_eq!(required_example_count(20, 11, QueryMode::Original), 31);
    }

    #[test]
    fn test_required_example_count_fetches_extra_rows_for_semantic_hard_mode() {
        assert_eq!(
            required_example_count(20, 11, QueryMode::SemanticHard),
            SEMANTIC_HARD_MIN_FETCH
        );
    }

    #[test]
    fn test_load_or_fetch_examples_truncates_cached_rows_to_requested_count() {
        let dir = tempdir().unwrap();
        let cache_file = dir.path().join("examples.jsonl");
        std::fs::write(
            &cache_file,
            [
                serde_json::json!({"code":"a","docstring":"one"}).to_string(),
                serde_json::json!({"code":"b","docstring":"two"}).to_string(),
                serde_json::json!({"code":"c","docstring":"three"}).to_string(),
            ]
            .join("\n"),
        )
        .unwrap();
        let examples = super::load_or_fetch_examples(&cache_file, 2).unwrap();
        assert_eq!(examples.len(), 2);
        assert_eq!(examples[0].1, "one");
        assert_eq!(examples[1].1, "two");
    }

    #[test]
    fn test_semantic_hard_query_override_rewrites_known_docstring() {
        let query = semantic_hard_query_for_docstring("Extracts video ID from URL.");
        assert_eq!(
            query.as_deref(),
            Some("recover clip identifier from address")
        );
    }

    #[test]
    fn test_semantic_hard_query_override_returns_none_for_unknown_docstring() {
        assert!(semantic_hard_query_for_docstring("unknown helper behavior").is_none());
    }

    #[test]
    fn test_setup_from_examples_with_semantic_hard_mode_filters_and_rewrites_queries() {
        let dir = tempdir().unwrap();
        let examples = vec![
            (
                "def keep():\n    return 1\n".to_string(),
                "unknown helper behavior".to_string(),
            ),
            (
                "def get_vid_from_url(url):\n    return url\n".to_string(),
                "Extracts video ID from URL.".to_string(),
            ),
        ];
        let track = CodeSearchNetTrack::setup_from_examples_with_mode(
            dir.path(),
            &examples,
            0,
            1,
            QueryMode::SemanticHard,
        )
        .unwrap();
        assert_eq!(track.episodes().len(), 1);
        assert_eq!(
            track.episodes()[0].query,
            "recover clip identifier from address"
        );
    }
}
