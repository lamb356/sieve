use std::collections::BTreeSet;
use std::path::PathBuf;
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct Episode {
    pub id: String,
    pub query: String,
    pub corpus_root: PathBuf,
    pub stable_root: PathBuf,
    pub fresh_stage_root: PathBuf,
    pub fresh_live_root: PathBuf,
    pub relevant_paths: BTreeSet<PathBuf>,
    pub deadlines: Vec<Duration>,
}

#[derive(Debug, Clone)]
pub struct Hit {
    pub path: PathBuf,
    pub score: f32,
    pub rank: usize,
    pub latency: Duration,
}

#[derive(Debug, Clone)]
pub struct EpisodeMetrics {
    pub episode_id: String,
    pub runner_name: String,
    pub deadline: Duration,
    pub recall_at_5: f32,
    pub hit_at_5: bool,
    pub mrr_at_5: f32,
    pub query_latency: Duration,
}

#[derive(Debug, Clone)]
pub struct AggregateMetrics {
    pub runner_name: String,
    pub deadline: Duration,
    pub mean_recall_at_5: f32,
    pub mean_hit_at_5: f32,
    pub mean_mrr_at_5: f32,
    pub p50_latency: Duration,
    pub p95_latency: Duration,
}
