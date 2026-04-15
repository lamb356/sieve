use std::fs;
use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use sieve_bench::codesearchnet::CodeSearchNetTrack;
use sieve_bench::embed_runner::EmbedKnnRunner;
use sieve_bench::eval::run_benchmark;
use sieve_bench::report::{json_report, print_report};
use sieve_bench::ripgrep_runner::RipgrepRunner;
use sieve_bench::runner::Runner;
use sieve_bench::sieve_runner::SieveRunner;
use sieve_bench::splade_runner::SpladeTantivyRunner;
use tracing_subscriber::EnvFilter;

#[derive(Debug, Clone, Parser)]
struct Cli {
    #[arg(long, default_value = "codesearchnet")]
    track: String,
    #[arg(long, default_value_t = 1000)]
    n_stable: usize,
    #[arg(long, default_value_t = 100)]
    n_fresh: usize,
    #[arg(long)]
    output: Option<PathBuf>,
    #[arg(long)]
    skip_embed: bool,
    #[arg(long)]
    skip_ripgrep: bool,
    #[arg(long)]
    cache_dir: Option<PathBuf>,
    #[arg(long)]
    semantic_hard: bool,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .try_init()
        .ok();

    let cli = Cli::parse();
    anyhow::ensure!(
        cli.track == "codesearchnet",
        "unsupported track: {}",
        cli.track
    );
    let cache_dir = cli.cache_dir.unwrap_or_else(default_cache_dir);
    fs::create_dir_all(&cache_dir)?;
    let track = if cli.semantic_hard {
        CodeSearchNetTrack::setup_semantic_hard(
            &cache_dir.join("codesearchnet-stream"),
            cli.n_stable,
            cli.n_fresh,
        )?
    } else {
        CodeSearchNetTrack::setup(
            &cache_dir.join("codesearchnet-stream"),
            cli.n_stable,
            cli.n_fresh,
        )?
    };

    let mut runners: Vec<Box<dyn Runner>> = vec![
        Box::new(SieveRunner::full()),
        Box::new(SieveRunner::scan_only()),
        Box::new(SieveRunner::random_expansion()),
        Box::new(SpladeTantivyRunner::default()),
    ];
    if !cli.skip_ripgrep {
        runners.push(Box::new(RipgrepRunner::new()));
    }
    if !cli.skip_embed {
        runners.push(Box::new(EmbedKnnRunner::default()));
    }

    let metrics = run_benchmark(track.episodes(), &mut runners, 5);
    let track_name = if cli.semantic_hard {
        "CodeSearchNet Stream-1K (semantic-hard subset)"
    } else {
        "CodeSearchNet Stream-1K"
    };
    print_report(track_name, cli.n_stable, cli.n_fresh, &metrics);

    if let Some(output) = cli.output {
        let report = json_report(track_name, cli.n_stable, cli.n_fresh, &metrics);
        fs::write(output, serde_json::to_vec_pretty(&report)?)?;
    }
    Ok(())
}

fn default_cache_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".sieve")
        .join("bench-cache")
}
