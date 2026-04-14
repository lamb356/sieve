use std::collections::BTreeSet;
use std::time::Duration;

use serde::Serialize;

use crate::eval::{aggregate_metrics, compute_zero_prep_retention, is_steady_deadline};
use crate::types::{AggregateMetrics, EpisodeMetrics};

#[derive(Debug, Serialize)]
pub struct JsonAggregate {
    pub runner_name: String,
    pub deadline_ms: u64,
    pub mean_recall_at_5: f32,
    pub mean_hit_at_5: f32,
    pub mean_mrr_at_5: f32,
    pub p50_latency_ms: u128,
    pub p95_latency_ms: u128,
}

#[derive(Debug, Serialize)]
pub struct JsonEpisodeMetric {
    pub episode_id: String,
    pub runner_name: String,
    pub deadline_ms: u64,
    pub recall_at_5: f32,
    pub hit_at_5: bool,
    pub mrr_at_5: f32,
    pub query_latency_ms: u128,
}

#[derive(Debug, Serialize)]
pub struct JsonReport {
    pub track: String,
    pub n_stable: usize,
    pub n_fresh: usize,
    pub aggregates: Vec<JsonAggregate>,
    pub episode_metrics: Vec<JsonEpisodeMetric>,
    pub zero_prep_retention_at_0ms: f32,
    pub semantic_lift_over_rg_at_0ms: Option<f32>,
}

pub fn print_report(track_name: &str, n_stable: usize, n_fresh: usize, metrics: &[EpisodeMetrics]) {
    let aggregates = aggregate_metrics(metrics);
    let deadlines = collect_deadlines(&aggregates);
    let runners = collect_runners(&aggregates);

    println!("=== SIEVE Benchmark Results ===");
    println!("Track: {track_name} ({n_fresh} fresh / {n_stable} stable)");
    println!();
    print_table(
        "Fresh Recall@5 by Deadline",
        &aggregates,
        &runners,
        &deadlines,
        |m| m.mean_recall_at_5,
    );
    println!();
    print_table(
        "Fresh Hit@5 by Deadline",
        &aggregates,
        &runners,
        &deadlines,
        |m| m.mean_hit_at_5,
    );
    println!();

    let zero_prep = metric_value(&aggregates, "sieve", Duration::from_millis(0), |m| {
        m.mean_recall_at_5
    })
    .zip(metric_value(
        &aggregates,
        "embed-knn",
        max_deadline(&deadlines),
        |m| m.mean_recall_at_5,
    ))
    .map(|(s, e)| compute_zero_prep_retention(s, e))
    .unwrap_or(0.0);
    let semantic_lift = metric_value(&aggregates, "sieve", Duration::from_millis(0), |m| {
        m.mean_recall_at_5
    })
    .zip(metric_value(
        &aggregates,
        "ripgrep",
        Duration::from_millis(0),
        |m| m.mean_recall_at_5,
    ))
    .map(|(s, r)| s - r);
    println!("--- Key Metrics ---");
    println!("ZeroPrepRetention@5:     {:.2}", zero_prep);
    if let Some(lift) = semantic_lift {
        println!("Semantic Lift over rg:   {:+.2} Recall@5 at t=0", lift);
    }
    println!("Time-to-Searchable (sieve):  0ms");
    if let Some(embed_final) =
        metric_value(&aggregates, "embed-knn", max_deadline(&deadlines), |m| {
            m.mean_recall_at_5
        })
    {
        println!(
            "Time-to-Searchable (embed):  measured by final deadline; final Recall@5 = {:.2}",
            embed_final
        );
    }
    println!();

    println!("--- Latency ---");
    println!("{:14} {:>8} {:>8}", "", "p50", "p95");
    for runner in &runners {
        if let Some(last) = aggregates
            .iter()
            .filter(|m| &m.runner_name == runner)
            .max_by_key(|m| m.deadline)
        {
            println!(
                "{:<14} {:>8.1}s {:>8.1}s",
                runner,
                last.p50_latency.as_secs_f32(),
                last.p95_latency.as_secs_f32()
            );
        }
    }
    println!();

    if let Some((n, sieve_recall, embed_recall, zpr)) = semantic_hard_subset(metrics) {
        println!("--- Semantic-Hard Subset ---");
        println!("(queries where ripgrep misses all relevant files in top 5)");
        println!("N = {n} queries");
        println!("sieve Recall@5:     {:.2}", sieve_recall);
        if let Some(embed) = embed_recall {
            println!("embed-knn Recall@5: {:.2}", embed);
        }
        println!("ZeroPrepRetention:  {:.2}", zpr);
    }
}

pub fn json_report(
    track_name: &str,
    n_stable: usize,
    n_fresh: usize,
    metrics: &[EpisodeMetrics],
) -> JsonReport {
    let aggregates = aggregate_metrics(metrics);
    let zero_prep = metric_value(&aggregates, "sieve", Duration::from_millis(0), |m| {
        m.mean_recall_at_5
    })
    .zip(metric_value(
        &aggregates,
        "embed-knn",
        max_deadline(&collect_deadlines(&aggregates)),
        |m| m.mean_recall_at_5,
    ))
    .map(|(s, e)| compute_zero_prep_retention(s, e))
    .unwrap_or(0.0);
    let semantic_lift = metric_value(&aggregates, "sieve", Duration::from_millis(0), |m| {
        m.mean_recall_at_5
    })
    .zip(metric_value(
        &aggregates,
        "ripgrep",
        Duration::from_millis(0),
        |m| m.mean_recall_at_5,
    ))
    .map(|(s, r)| s - r);
    JsonReport {
        track: track_name.to_string(),
        n_stable,
        n_fresh,
        aggregates: aggregates
            .into_iter()
            .map(|m| JsonAggregate {
                runner_name: m.runner_name,
                deadline_ms: m.deadline.as_millis() as u64,
                mean_recall_at_5: m.mean_recall_at_5,
                mean_hit_at_5: m.mean_hit_at_5,
                mean_mrr_at_5: m.mean_mrr_at_5,
                p50_latency_ms: m.p50_latency.as_millis(),
                p95_latency_ms: m.p95_latency.as_millis(),
            })
            .collect(),
        episode_metrics: metrics
            .iter()
            .map(|m| JsonEpisodeMetric {
                episode_id: m.episode_id.clone(),
                runner_name: m.runner_name.clone(),
                deadline_ms: m.deadline.as_millis() as u64,
                recall_at_5: m.recall_at_5,
                hit_at_5: m.hit_at_5,
                mrr_at_5: m.mrr_at_5,
                query_latency_ms: m.query_latency.as_millis(),
            })
            .collect(),
        zero_prep_retention_at_0ms: zero_prep,
        semantic_lift_over_rg_at_0ms: semantic_lift,
    }
}

fn semantic_hard_subset(metrics: &[EpisodeMetrics]) -> Option<(usize, f32, Option<f32>, f32)> {
    let final_deadline = metrics.iter().map(|m| m.deadline).max()?;
    let mut hard_ids = BTreeSet::new();
    for metric in metrics {
        if metric.runner_name == "ripgrep" && metric.deadline == final_deadline && !metric.hit_at_5
        {
            hard_ids.insert(metric.episode_id.clone());
        }
    }
    if hard_ids.is_empty() {
        return None;
    }
    let sieve_mean = mean_for_subset(metrics, &hard_ids, "sieve", final_deadline)?;
    let embed_mean = mean_for_subset(metrics, &hard_ids, "embed-knn", final_deadline);
    let zpr = embed_mean
        .map(|embed| compute_zero_prep_retention(sieve_mean, embed))
        .unwrap_or(0.0);
    Some((hard_ids.len(), sieve_mean, embed_mean, zpr))
}

fn mean_for_subset(
    metrics: &[EpisodeMetrics],
    ids: &BTreeSet<String>,
    runner: &str,
    deadline: Duration,
) -> Option<f32> {
    let vals: Vec<f32> = metrics
        .iter()
        .filter(|m| {
            m.runner_name == runner && m.deadline == deadline && ids.contains(&m.episode_id)
        })
        .map(|m| m.recall_at_5)
        .collect();
    if vals.is_empty() {
        None
    } else {
        Some(vals.iter().sum::<f32>() / vals.len() as f32)
    }
}

fn print_table<F: Fn(&AggregateMetrics) -> f32>(
    title: &str,
    metrics: &[AggregateMetrics],
    runners: &[String],
    deadlines: &[Duration],
    value: F,
) {
    println!("--- {title} ---");
    print!("{:14}", "");
    for deadline in deadlines {
        print!(" {:>7}", format_deadline(*deadline));
    }
    println!();
    for runner in runners {
        print!("{:<14}", runner);
        for deadline in deadlines {
            let v = metric_value(metrics, runner, *deadline, &value).unwrap_or(0.0);
            print!(" {:>7.2}", v);
        }
        println!();
    }
}

fn metric_value<F: Fn(&AggregateMetrics) -> f32>(
    metrics: &[AggregateMetrics],
    runner: &str,
    deadline: Duration,
    value: F,
) -> Option<f32> {
    metrics
        .iter()
        .find(|m| m.runner_name == runner && m.deadline == deadline)
        .map(value)
}

fn collect_deadlines(metrics: &[AggregateMetrics]) -> Vec<Duration> {
    let mut set = BTreeSet::new();
    for metric in metrics {
        set.insert(metric.deadline);
    }
    set.into_iter().collect()
}

fn collect_runners(metrics: &[AggregateMetrics]) -> Vec<String> {
    let mut set = BTreeSet::new();
    for metric in metrics {
        set.insert(metric.runner_name.clone());
    }
    set.into_iter().collect()
}

fn format_deadline(deadline: Duration) -> String {
    if is_steady_deadline(deadline) {
        "T+steady".to_string()
    } else if deadline.as_millis() < 1000 {
        format!("{}ms", deadline.as_millis())
    } else {
        format!("{}s", deadline.as_secs())
    }
}

fn max_deadline(deadlines: &[Duration]) -> Duration {
    deadlines
        .iter()
        .copied()
        .max()
        .unwrap_or(Duration::from_secs(0))
}
