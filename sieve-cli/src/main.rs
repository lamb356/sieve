use std::fs;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use ignore::WalkBuilder;
use serde_json::json;
#[cfg(feature = "semantic")]
use sieve_core::default_sieve_data_dir;
use sieve_core::lexical::{build_pending_shards, load_indexed_entries};
#[cfg(feature = "semantic")]
use sieve_core::model::{ModelManager, DEFAULT_MODEL_NAME};
use sieve_core::{blake3_hex, Index, SearchOptions, SearchResult};

#[derive(Debug, Parser)]
#[command(name = "sieve-cli")]
#[command(about = "SIEVE semantic grep engine CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    Index {
        path: PathBuf,
    },
    Search {
        query: String,
        #[arg(long)]
        index: Option<PathBuf>,
        #[arg(long)]
        top: Option<usize>,
        #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
        format: OutputFormat,
        #[arg(long, default_value_t = 0)]
        context: usize,
    },
    Status {
        #[arg(long)]
        index: Option<PathBuf>,
    },
    #[cfg(feature = "semantic")]
    DownloadModel {
        #[arg(long)]
        sparse: bool,
        #[arg(long)]
        all: bool,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
enum OutputFormat {
    Text,
    Json,
}

fn main() {
    init_tracing();
    if let Err(err) = run() {
        eprintln!("error: {err:#}");
        std::process::exit(1);
    }
}

fn init_tracing() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .try_init();
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Index { path } => run_index(&path),
        Commands::Search {
            query,
            index,
            top,
            format,
            context,
        } => run_search(&query, index.as_deref(), top, format, context),
        Commands::Status { index } => run_status(index.as_deref()),
        #[cfg(feature = "semantic")]
        Commands::DownloadModel { sparse, all } => run_download_model(sparse, all),
    }
}

fn run_index(path: &Path) -> Result<()> {
    let source_root = path
        .canonicalize()
        .with_context(|| format!("failed to resolve source path {}", path.display()))?;
    if !source_root.is_dir() {
        bail!("index path must be a directory: {}", source_root.display());
    }

    let index_root = source_root.join(".sieve");
    let index = Index::open_or_create(&index_root)?;

    let mut seen_paths = std::collections::HashSet::new();
    for entry in WalkBuilder::new(&source_root)
        .hidden(false)
        .git_ignore(true)
        .git_exclude(true)
        .git_global(true)
        .build()
    {
        let entry = entry?;
        let entry_path = entry.path();
        if entry_path.starts_with(&index_root) || !entry.file_type().is_some_and(|ft| ft.is_file())
        {
            continue;
        }

        let metadata = fs::metadata(entry_path)
            .with_context(|| format!("failed to stat {}", entry_path.display()))?;
        let mtime_ms = metadata
            .modified()
            .ok()
            .and_then(|mtime| mtime.duration_since(UNIX_EPOCH).ok())
            .map(|dur| dur.as_nanos() as u64)
            .unwrap_or_default();
        let size = metadata.len();
        let relative = entry_path
            .strip_prefix(&source_root)
            .unwrap_or(entry_path)
            .to_string_lossy()
            .to_string();

        seen_paths.insert(relative.clone());
        if index.should_skip_source(&relative, mtime_ms, size)? {
            continue;
        }

        let content = fs::read(entry_path)
            .with_context(|| format!("failed to read {}", entry_path.display()))?;
        let text = String::from_utf8_lossy(&content).into_owned();
        let wal_entry_id = index.add_text(relative.clone(), text)?;
        index.record_source_entry(relative, mtime_ms, size, blake3_hex(&content), wal_entry_id)?;
    }

    index.prune_manifest_to_paths(&seen_paths)?;
    index.save_manifest()?;
    let built_shards = build_pending_shards(&index)?;

    println!("Indexing {}...", source_root.display());
    println!("WAL: {} chunks", index.chunk_count()?);
    println!(
        "Shards: {built_shards} built ({} entries indexed)",
        load_indexed_entries(&index.root().join("segments"))?.len()
    );
    #[cfg(feature = "semantic")]
    {
        let _embedded = index.embed_pending(32)?;
        let semantic = index.semantic_status()?;
        println!(
            "Embedding: {}/{} chunks done",
            semantic.vectors, semantic.total_chunks
        );
    }
    Ok(())
}

fn run_search(
    query: &str,
    index_override: Option<&Path>,
    top: Option<usize>,
    format: OutputFormat,
    context: usize,
) -> Result<()> {
    let index_root = resolve_index_root(index_override)?;
    if !is_index_root(&index_root) {
        bail!("index does not exist: {}", index_root.display());
    }
    let index = Index::open_or_create(&index_root)
        .with_context(|| format!("failed to open index at {}", index_root.display()))?;

    let results = index.search(query, SearchOptions { top_k: top })?;
    match format {
        OutputFormat::Text => {
            for result in &results {
                let snippet = if context > 0 {
                    render_with_context(&index, result, context)?
                } else {
                    result.snippet.clone()
                };
                println!(
                    "[{}] {}:{}:{}-{}:{}",
                    result.source_layer.as_str(),
                    result.source_path,
                    result.line_number,
                    result.byte_range.0,
                    result.byte_range.1,
                    snippet.replace('\n', "\\n")
                );
            }
        }
        OutputFormat::Json => {
            let rendered: Vec<_> = results
                .iter()
                .map(|result| {
                    let snippet = if context > 0 {
                        render_with_context(&index, result, context)
                            .unwrap_or_else(|_| result.snippet.clone())
                    } else {
                        result.snippet.clone()
                    };
                    json!({
                        "path": result.source_path,
                        "line": result.line_number,
                        "chunk_id": result.chunk_id,
                        "byte_range": [result.byte_range.0, result.byte_range.1],
                        "snippet": snippet,
                        "score": result.score,
                        "layer": result.source_layer.as_str(),
                    })
                })
                .collect();
            println!("{}", serde_json::to_string_pretty(&rendered)?);
        }
    }
    Ok(())
}

fn run_status(index_override: Option<&Path>) -> Result<()> {
    let index_root = resolve_index_root(index_override)?;
    if !is_index_root(&index_root) {
        bail!("index does not exist: {}", index_root.display());
    }
    let index = Index::open_or_create(&index_root)
        .with_context(|| format!("failed to open index at {}", index_root.display()))?;
    let wal_entries = index.wal_entries_count()?;
    let chunk_count = index.chunk_count()?;
    let segments_dir = index.root().join("segments");
    let shard_count = if segments_dir.exists() {
        fs::read_dir(&segments_dir)?
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .filter(|path| path.is_dir())
            .count()
    } else {
        0
    };
    let indexed_entries = load_indexed_entries(&segments_dir)?.len() as usize;
    let unindexed_entries = wal_entries.saturating_sub(indexed_entries);

    println!("Index: {}", index.root().display());
    println!("WAL entries: {wal_entries}");
    println!("Chunks: {chunk_count}");
    println!(
        "Shards: {shard_count} ({indexed_entries} entries indexed, {unindexed_entries} unindexed)"
    );
    #[cfg(feature = "semantic")]
    {
        let semantic = index.semantic_status()?;
        if semantic.vectors == 0 {
            println!("Vectors: none (run `sieve download-model`)");
        } else {
            println!(
                "Vectors: {} ({}-dim, {})",
                semantic.vectors, semantic.dimension, DEFAULT_MODEL_NAME
            );
        }
        let percent = if semantic.total_chunks == 0 {
            0
        } else {
            (semantic.vectors * 100) / semantic.total_chunks
        };
        println!(
            "Semantic coverage: {}/{} chunks ({}%)",
            semantic.vectors, semantic.total_chunks, percent
        );
        if semantic.model_cached {
            println!("Model: cached at {}", semantic.model_dir.display());
        }
    }
    Ok(())
}

#[cfg(feature = "semantic")]
fn run_download_model(sparse: bool, all: bool) -> Result<()> {
    let manager = ModelManager::new(&default_sieve_data_dir());
    if sparse {
        println!("SPLADE model download not yet implemented (Phase 4 Batch 2)");
        return Ok(());
    }
    let dense = manager.ensure_dense_model()?;
    println!(
        "Model cached at {}",
        dense.model_path.parent().unwrap_or(dense.model_path.as_path()).display()
    );
    println!("Model file: {}", dense.model_path.display());
    println!("Tokenizer file: {}", dense.tokenizer_path.display());
    if all {
        println!("SPLADE model download not yet implemented (Phase 4 Batch 2)");
    }
    Ok(())
}

fn resolve_index_root(index_override: Option<&Path>) -> Result<PathBuf> {
    Ok(match index_override {
        Some(path) => path.to_path_buf(),
        None => std::env::current_dir()?.join(".sieve"),
    })
}

fn is_index_root(path: &Path) -> bool {
    path.join("wal").join("wal.meta").is_file() && path.join("wal").join("wal.content").is_file()
}

fn render_with_context(index: &Index, result: &SearchResult, context: usize) -> Result<String> {
    let content = index.read_entry_content(result.wal_entry_id)?;
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return Ok(result.snippet.clone());
    }
    let start = result.line_range.0.saturating_sub(context + 1);
    let end = (result.line_range.1 + context).min(lines.len());
    Ok(lines[start..end].join("\n"))
}
