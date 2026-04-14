use std::collections::BTreeSet;
use std::path::PathBuf;
use std::time::Duration;

use sieve_bench::codesearchnet::CodeSearchNetTrack;
use sieve_bench::eval::{
    compute_hit_at_k, compute_mrr_at_k, compute_recall_at_k, compute_zero_prep_retention,
    run_benchmark, T_STEADY_DEADLINE,
};
use sieve_bench::ripgrep_runner::tokenize_query;
use sieve_bench::runner::Runner;
use sieve_bench::types::{Episode, Hit};
use tempfile::tempdir;

#[test]
fn test_episode_creation() {
    let dir = tempdir().unwrap();
    let samples = (0..10)
        .map(|i| {
            (
                format!("def f{i}():\n    return {i}\n"),
                format!("function {i}"),
            )
        })
        .collect::<Vec<_>>();
    let track = CodeSearchNetTrack::setup_from_examples(dir.path(), &samples, 5, 5).unwrap();
    assert_eq!(track.episodes().len(), 5);
    for ep in track.episodes() {
        assert!(!ep.query.is_empty());
        assert!(!ep.relevant_paths.is_empty());
        assert!(!ep.deadlines.is_empty());
    }
}

#[test]
fn test_recall_computation() {
    let relevant = BTreeSet::from([PathBuf::from("b.py"), PathBuf::from("d.py")]);
    let hits = vec![
        Hit {
            path: PathBuf::from("a.py"),
            score: 1.0,
            rank: 1,
            latency: Duration::from_millis(1),
        },
        Hit {
            path: PathBuf::from("b.py"),
            score: 0.9,
            rank: 2,
            latency: Duration::from_millis(1),
        },
        Hit {
            path: PathBuf::from("c.py"),
            score: 0.8,
            rank: 3,
            latency: Duration::from_millis(1),
        },
    ];
    assert!((compute_recall_at_k(&hits, &relevant, 5) - 0.5).abs() < 1e-6);
    assert!(compute_hit_at_k(&hits, &relevant, 5));
    assert!((compute_mrr_at_k(&hits, &relevant, 5) - 0.5).abs() < 1e-6);
}

#[test]
fn test_zero_prep_retention() {
    let zpr = compute_zero_prep_retention(0.4, 0.6);
    assert!((zpr - 0.6666667).abs() < 1e-4);
}

#[test]
fn test_ripgrep_runner_tokenizes_query() {
    let toks = tokenize_query("error handling retry");
    assert_eq!(toks, vec!["error", "handling", "retry"]);
}

#[test]
fn test_fresh_files_not_in_stable() {
    let dir = tempdir().unwrap();
    let samples = (0..12)
        .map(|i| {
            (
                format!("def f{i}():\n    return {i}\n"),
                format!("function {i}"),
            )
        })
        .collect::<Vec<_>>();
    let track = CodeSearchNetTrack::setup_from_examples(dir.path(), &samples, 7, 5).unwrap();
    let stable: BTreeSet<_> = std::fs::read_dir(track.stable_dir())
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().to_string())
        .collect();
    let fresh: BTreeSet<_> = std::fs::read_dir(track.fresh_stage_dir())
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().to_string())
        .collect();
    assert!(stable.is_disjoint(&fresh));
}

#[test]
fn test_cached_dataset_survives_materialization() {
    let dir = tempdir().unwrap();
    let dataset_cache = dir.path().join("dataset-cache");
    std::fs::create_dir_all(&dataset_cache).unwrap();
    let cache_file = dataset_cache.join("codesearchnet_python_test.jsonl");
    let mut lines = String::new();
    for i in 0..12 {
        lines.push_str(
            &serde_json::json!({
                "code": format!("def f{i}():\n    return {i}\n"),
                "docstring": format!("function {i}"),
            })
            .to_string(),
        );
        lines.push('\n');
    }
    std::fs::write(&cache_file, lines).unwrap();
    let _track = CodeSearchNetTrack::setup(dir.path(), 7, 5).unwrap();
    assert!(cache_file.exists());
}

#[derive(Default)]
struct FastRelevantRunner;

impl Runner for FastRelevantRunner {
    fn name(&self) -> &'static str {
        "fast-relevant"
    }

    fn prepare_stable(&mut self, _ep: &Episode) -> anyhow::Result<()> {
        Ok(())
    }

    fn begin_fresh_arrival(&mut self, _ep: &Episode) -> anyhow::Result<()> {
        Ok(())
    }

    fn search_at_deadline(
        &mut self,
        ep: &Episode,
        _deadline: Duration,
        _k: usize,
    ) -> anyhow::Result<Vec<Hit>> {
        std::thread::sleep(Duration::from_millis(10));
        Ok(vec![Hit {
            path: ep.relevant_paths.iter().next().unwrap().clone(),
            score: 1.0,
            rank: 1,
            latency: Duration::from_millis(10),
        }])
    }

    fn cleanup(&mut self) -> anyhow::Result<()> {
        Ok(())
    }
}

#[derive(Default)]
struct SteadyOnlyRunner {
    waited: bool,
}

impl Runner for SteadyOnlyRunner {
    fn name(&self) -> &'static str {
        "steady-only"
    }

    fn prepare_stable(&mut self, _ep: &Episode) -> anyhow::Result<()> {
        Ok(())
    }

    fn begin_fresh_arrival(&mut self, _ep: &Episode) -> anyhow::Result<()> {
        self.waited = false;
        Ok(())
    }

    fn wait_for_steady_state(&mut self, _ep: &Episode) -> anyhow::Result<()> {
        self.waited = true;
        Ok(())
    }

    fn search_at_deadline(
        &mut self,
        ep: &Episode,
        deadline: Duration,
        _k: usize,
    ) -> anyhow::Result<Vec<Hit>> {
        if self.waited && deadline == T_STEADY_DEADLINE {
            Ok(vec![Hit {
                path: ep.relevant_paths.iter().next().unwrap().clone(),
                score: 1.0,
                rank: 1,
                latency: Duration::from_millis(1),
            }])
        } else {
            Ok(Vec::new())
        }
    }

    fn cleanup(&mut self) -> anyhow::Result<()> {
        Ok(())
    }
}

#[test]
fn test_run_benchmark_accepts_fast_results_before_deadline() {
    let dir = tempdir().unwrap();
    let corpus_root = dir.path().join("corpus");
    let stage_dir = dir.path().join("stage");
    std::fs::create_dir_all(&corpus_root).unwrap();
    std::fs::create_dir_all(&stage_dir).unwrap();
    let staged = stage_dir.join("fresh.py");
    std::fs::write(&staged, "def answer():\n    return 42\n").unwrap();
    let ep = Episode {
        id: "ep-1".to_string(),
        query: "answer function".to_string(),
        corpus_root: corpus_root.clone(),
        stable_root: corpus_root.clone(),
        fresh_stage_root: staged,
        fresh_live_root: corpus_root.join("fresh.py"),
        relevant_paths: BTreeSet::from([PathBuf::from("fresh.py")]),
        deadlines: vec![Duration::from_millis(50), Duration::from_millis(100)],
    };
    let mut runners: Vec<Box<dyn Runner>> = vec![Box::new(FastRelevantRunner)];
    let metrics = run_benchmark(&[ep], &mut runners, 5);
    assert!(metrics.iter().all(|m| m.hit_at_5));
    assert!(metrics.iter().all(|m| (m.recall_at_5 - 1.0).abs() < 1e-6));
}

#[test]
fn test_run_benchmark_waits_for_t_steady() {
    let dir = tempdir().unwrap();
    let corpus_root = dir.path().join("corpus");
    let stage_dir = dir.path().join("stage");
    std::fs::create_dir_all(&corpus_root).unwrap();
    std::fs::create_dir_all(&stage_dir).unwrap();
    let staged = stage_dir.join("fresh.py");
    std::fs::write(&staged, "def answer():\n    return 42\n").unwrap();
    let ep = Episode {
        id: "ep-steady".to_string(),
        query: "answer function".to_string(),
        corpus_root: corpus_root.clone(),
        stable_root: corpus_root.clone(),
        fresh_stage_root: staged,
        fresh_live_root: corpus_root.join("fresh.py"),
        relevant_paths: BTreeSet::from([PathBuf::from("fresh.py")]),
        deadlines: vec![Duration::from_millis(0), T_STEADY_DEADLINE],
    };
    let mut runners: Vec<Box<dyn Runner>> = vec![Box::new(SteadyOnlyRunner::default())];
    let metrics = run_benchmark(&[ep], &mut runners, 5);
    assert_eq!(metrics.len(), 2);
    assert!(!metrics[0].hit_at_5);
    assert!(metrics[1].hit_at_5);
}
