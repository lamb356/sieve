use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;
use std::thread;
use std::time::{Duration, Instant};

use crate::runner::Runner;
use crate::types::{AggregateMetrics, Episode, EpisodeMetrics, Hit};

pub const T_STEADY_DEADLINE: Duration = Duration::from_secs(30);

pub fn is_steady_deadline(deadline: Duration) -> bool {
    deadline == T_STEADY_DEADLINE
}

pub fn is_zero_prep_deadline(deadline: Duration) -> bool {
    deadline.is_zero()
}

pub fn run_benchmark(
    episodes: &[Episode],
    runners: &mut [Box<dyn Runner>],
    k: usize,
) -> Vec<EpisodeMetrics> {
    let mut out = Vec::new();
    for ep in episodes {
        for runner in runners.iter_mut() {
            let _ = runner.cleanup();
            reset_episode(ep).expect("reset episode failed");
            runner.prepare_stable(ep).expect("prepare_stable failed");
            std::fs::rename(&ep.fresh_stage_root, &ep.fresh_live_root)
                .expect("fresh rename failed");
            let t0 = Instant::now();
            runner
                .begin_fresh_arrival(ep)
                .expect("begin_fresh_arrival failed");
            for deadline in &ep.deadlines {
                if is_steady_deadline(*deadline) {
                    runner
                        .wait_for_steady_state(ep)
                        .expect("wait_for_steady_state failed");
                } else {
                    let elapsed = t0.elapsed();
                    if *deadline > elapsed {
                        thread::sleep(*deadline - elapsed);
                    }
                }
                let started = Instant::now();
                let mut hits = runner
                    .search_at_deadline(ep, *deadline, k)
                    .unwrap_or_default();
                let latency = started.elapsed();
                if !is_steady_deadline(*deadline)
                    && !is_zero_prep_deadline(*deadline)
                    && latency > *deadline
                {
                    hits.clear();
                }
                out.push(EpisodeMetrics {
                    episode_id: ep.id.clone(),
                    runner_name: runner.name().to_string(),
                    deadline: *deadline,
                    recall_at_5: compute_recall_at_k(&hits, &ep.relevant_paths, k),
                    hit_at_5: compute_hit_at_k(&hits, &ep.relevant_paths, k),
                    mrr_at_5: compute_mrr_at_k(&hits, &ep.relevant_paths, k),
                    query_latency: latency,
                });
            }
            let _ = runner.cleanup();
            reset_episode(ep).expect("reset episode failed after runner");
        }
    }
    out
}

fn reset_episode(ep: &Episode) -> std::io::Result<()> {
    if ep.fresh_live_root.exists() {
        if ep.fresh_stage_root.exists() {
            let _ = fs::remove_file(&ep.fresh_stage_root);
        }
        if let Some(parent) = ep.fresh_stage_root.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::rename(&ep.fresh_live_root, &ep.fresh_stage_root)?;
    }
    let sieve_dir = ep.corpus_root.join(".sieve");
    if sieve_dir.exists() {
        fs::remove_dir_all(sieve_dir)?;
    }
    Ok(())
}

pub fn compute_recall_at_k(hits: &[Hit], relevant: &BTreeSet<PathBuf>, k: usize) -> f32 {
    let top_k: BTreeSet<_> = hits.iter().take(k).map(|h| &h.path).collect();
    let found = relevant.iter().filter(|r| top_k.contains(r)).count();
    found as f32 / relevant.len().max(1) as f32
}

pub fn compute_hit_at_k(hits: &[Hit], relevant: &BTreeSet<PathBuf>, k: usize) -> bool {
    hits.iter().take(k).any(|h| relevant.contains(&h.path))
}

pub fn compute_mrr_at_k(hits: &[Hit], relevant: &BTreeSet<PathBuf>, k: usize) -> f32 {
    for (i, hit) in hits.iter().take(k).enumerate() {
        if relevant.contains(&hit.path) {
            return 1.0 / (i as f32 + 1.0);
        }
    }
    0.0
}

pub fn compute_zero_prep_retention(sieve_recall_at_0: f32, embed_recall_final: f32) -> f32 {
    if embed_recall_final < 1e-6 {
        return 0.0;
    }
    sieve_recall_at_0 / embed_recall_final
}

pub fn aggregate_metrics(metrics: &[EpisodeMetrics]) -> Vec<AggregateMetrics> {
    let mut groups = std::collections::BTreeMap::<(String, Duration), Vec<&EpisodeMetrics>>::new();
    for metric in metrics {
        groups
            .entry((metric.runner_name.clone(), metric.deadline))
            .or_default()
            .push(metric);
    }
    groups
        .into_iter()
        .map(|((runner_name, deadline), items)| {
            let mut lats: Vec<_> = items.iter().map(|m| m.query_latency).collect();
            lats.sort();
            let n = lats.len().max(1);
            let p50 = lats[(n - 1) / 2];
            let p95 = lats[((n - 1) * 95) / 100];
            AggregateMetrics {
                runner_name,
                deadline,
                mean_recall_at_5: items.iter().map(|m| m.recall_at_5).sum::<f32>()
                    / items.len().max(1) as f32,
                mean_hit_at_5: items
                    .iter()
                    .map(|m| if m.hit_at_5 { 1.0 } else { 0.0 })
                    .sum::<f32>()
                    / items.len().max(1) as f32,
                mean_mrr_at_5: items.iter().map(|m| m.mrr_at_5).sum::<f32>()
                    / items.len().max(1) as f32,
                p50_latency: p50,
                p95_latency: p95,
            }
        })
        .collect()
}
