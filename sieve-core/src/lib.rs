pub mod aliases;
pub mod chunk;
#[cfg(feature = "semantic")]
pub mod default_queries;
pub mod df_prior;
#[cfg(feature = "semantic")]
pub mod embed;
#[cfg(feature = "semantic")]
pub mod event_rerank;
pub mod fusion;
pub mod lexical;
#[cfg(feature = "semantic")]
pub mod model;
#[cfg(feature = "semantic")]
pub mod semantic_query;
#[cfg(feature = "semantic")]
pub mod semantic_scan;
#[cfg(feature = "semantic")]
pub mod sparse;
#[cfg(feature = "semantic")]
pub mod surface;
#[cfg(feature = "semantic")]
pub mod training_export;
#[cfg(feature = "semantic")]
pub mod vectors;
#[cfg(feature = "semantic")]
pub mod window_score;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use blake3::Hasher;
use fs2::FileExt;
use memchr::{memchr_iter, memmem, memrchr};
use memmap2::Mmap;
use rayon::join;
use regex::bytes::Regex;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[cfg(feature = "semantic")]
use crate::aliases::AliasLexicon;
use crate::chunk::{Chunk, SlidingChunker};
#[cfg(feature = "semantic")]
use crate::df_prior::static_df_frac;
use crate::fusion::{
    coverage_aware_rrf_fuse, CoverageState, LayerResults, ResultId, ResultSource, ScoredResult,
};
#[cfg(feature = "semantic")]
use crate::lexical::{load_indexed_entries, search_semantic_lexical};
use crate::lexical::{search_lexical_with_fallback, LexicalMatch};
#[cfg(feature = "semantic")]
use crate::model::DEFAULT_SPARSE_MODEL_NAME;
#[cfg(feature = "semantic")]
use crate::model::{ModelManager, DEFAULT_MODEL_NAME};
#[cfg(feature = "semantic")]
use crate::semantic_query::{ContentType, SemanticQuery};
#[cfg(feature = "semantic")]
use crate::semantic_scan::{compile_scan_query_with_options, SemanticScanOptions};
#[cfg(feature = "semantic")]
use crate::surface::realize_surfaces;
#[cfg(feature = "semantic")]
use crate::vectors::{snippet_from_byte_range, HotVectorStore, VectorMatch, VectorMeta};

const WAL_META_FILE_NAME: &str = "wal.meta";
const WAL_CONTENT_FILE_NAME: &str = "wal.content";
const MANIFEST_FILE_NAME: &str = "manifest.json";

#[derive(Debug, Error)]
pub enum SieveError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("tantivy error: {0}")]
    Tantivy(#[from] tantivy::TantivyError),
    #[error("tantivy directory error: {0}")]
    TantivyDirectory(#[from] tantivy::directory::error::OpenDirectoryError),
    #[error("regex error: {0}")]
    Regex(#[from] regex::Error),
    #[error("{0}")]
    Message(String),
    #[error("lock poisoned")]
    LockPoisoned,
}

pub type Result<T> = std::result::Result<T, SieveError>;

#[derive(Debug, Clone)]
pub struct Index {
    root: PathBuf,
    wal_dir: PathBuf,
    wal_meta_path: PathBuf,
    wal_content_path: PathBuf,
    sources_dir: PathBuf,
    manifest_path: PathBuf,
    metadata: Arc<RwLock<Vec<WalMetaRecord>>>,
    manifest: Arc<RwLock<HashMap<String, SourceManifestEntry>>>,
    #[cfg(feature = "semantic")]
    dense_embedder: Arc<RwLock<Option<Arc<crate::embed::Embedder>>>>,
    #[cfg(feature = "semantic")]
    sparse_encoder:
        Arc<RwLock<HashMap<crate::model::SparseRoute, Arc<crate::sparse::SpladeEncoder>>>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryPromotedDense {
    pub max_promoted_chunks: usize,
    pub max_promoted_ms: u64,
}

impl Default for QueryPromotedDense {
    fn default() -> Self {
        Self {
            max_promoted_chunks: 10,
            max_promoted_ms: 300,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SearchOptions {
    pub top_k: Option<usize>,
    #[cfg(feature = "semantic")]
    pub fresh_only: bool,
    #[cfg(feature = "semantic")]
    pub experimental_rerank: bool,
    #[cfg(feature = "semantic")]
    pub no_expand: bool,
    #[cfg(feature = "semantic")]
    pub no_window_scoring: bool,
    #[cfg(feature = "semantic")]
    pub no_df_filter: bool,
    #[cfg(feature = "semantic")]
    pub random_expansion: bool,
    #[cfg(feature = "semantic")]
    pub allow_delta_fallback: bool,
    #[cfg(feature = "semantic")]
    pub query_promoted_dense: QueryPromotedDense,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            top_k: None,
            #[cfg(feature = "semantic")]
            fresh_only: false,
            #[cfg(feature = "semantic")]
            experimental_rerank: false,
            #[cfg(feature = "semantic")]
            no_expand: false,
            #[cfg(feature = "semantic")]
            no_window_scoring: false,
            #[cfg(feature = "semantic")]
            no_df_filter: false,
            #[cfg(feature = "semantic")]
            random_expansion: false,
            #[cfg(feature = "semantic")]
            allow_delta_fallback: true,
            #[cfg(feature = "semantic")]
            query_promoted_dense: QueryPromotedDense::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum QueryPlan {
    Exact(String),
    Regex(String),
    #[cfg(feature = "semantic")]
    Semantic(Arc<SemanticQuery>),
    Lexical(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    pub source_path: String,
    pub line_number: usize,
    pub line_range: (usize, usize),
    pub chunk_id: u32,
    pub byte_range: (u32, u32),
    pub snippet: String,
    pub score: f64,
    pub source_layer: ResultSource,
    pub wal_entry_id: u64,
}

#[cfg(feature = "semantic")]
pub fn plan_query(
    raw: &str,
    sparse: Option<&crate::sparse::SpladeEncoder>,
    aliases: &crate::aliases::AliasLexicon,
    options: &SearchOptions,
    content_type: ContentType,
) -> QueryPlan {
    if raw.starts_with('/') && raw.ends_with('/') && raw.len() >= 2 {
        return QueryPlan::Regex(raw[1..raw.len() - 1].to_string());
    }
    if raw.starts_with('"') && raw.ends_with('"') && raw.len() >= 2 {
        return QueryPlan::Exact(raw[1..raw.len() - 1].to_string());
    }
    let Some(sparse) = sparse else {
        return QueryPlan::Lexical(raw.to_string());
    };
    match crate::semantic_query::compile_semantic_query_with_options(
        raw,
        sparse,
        aliases,
        crate::semantic_query::SemanticCompileOptions {
            no_expand: options.no_expand,
            random_expansion: options.random_expansion,
        },
        content_type,
    ) {
        Ok(query) if query.terms.iter().any(|term| term.is_anchor) => {
            QueryPlan::Semantic(Arc::new(query))
        }
        _ => QueryPlan::Lexical(raw.to_string()),
    }
}

#[cfg(feature = "semantic")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticStatus {
    pub model_cached: bool,
    pub model_dir: PathBuf,
    pub vectors: usize,
    pub dimension: usize,
    pub total_chunks: usize,
}

#[cfg(feature = "semantic")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchCoverage {
    pub total_chunks: usize,
    pub embedded_chunks: usize,
    pub delta_chunks: usize,
    pub skipped_due_to_budget: bool,
}

#[cfg(feature = "semantic")]
#[derive(Debug, Clone, PartialEq)]
pub struct SourceResultSet {
    pub source: ResultSource,
    pub weight: f64,
    pub coverage: CoverageState,
    pub results: Vec<ScoredResult>,
}

#[cfg(feature = "semantic")]
pub fn filter_stale_only_source_sets(
    source_sets: Vec<SourceResultSet>,
    fresh_ids: &HashSet<u64>,
    allow_stale_only_fusion: bool,
) -> Vec<SourceResultSet> {
    if allow_stale_only_fusion || fresh_ids.is_empty() {
        return source_sets;
    }
    let has_any_fresh = source_sets.iter().any(|set| {
        set.results
            .iter()
            .any(|result| fresh_ids.contains(&result.wal_entry_id))
    });
    if !has_any_fresh {
        return source_sets;
    }
    source_sets
        .into_iter()
        .filter_map(|mut set| {
            set.results
                .retain(|result| fresh_ids.contains(&result.wal_entry_id));
            if set.results.is_empty() {
                None
            } else {
                Some(set)
            }
        })
        .collect()
}

#[cfg(feature = "semantic")]
fn fuse_source_sets(source_sets: &[SourceResultSet], k: f64) -> Vec<ScoredResult> {
    let layers: Vec<LayerResults> = source_sets
        .iter()
        .map(|set| LayerResults {
            source: set.source,
            weight: set.weight,
            results: collapse_scored_results_by_file(set.results.clone(), set.results.len()),
            coverage: set.coverage,
        })
        .collect();
    coverage_aware_rrf_fuse(layers, k)
}

#[cfg(feature = "semantic")]
fn coverage_state_from_entry_counts(covered_entries: usize, total_entries: usize) -> CoverageState {
    if total_entries == 0 || covered_entries == 0 {
        CoverageState::Unavailable
    } else if covered_entries >= total_entries {
        CoverageState::Complete
    } else {
        CoverageState::Partial((covered_entries as f32 / total_entries as f32).clamp(0.0, 1.0))
    }
}

#[cfg(feature = "semantic")]
#[derive(Debug, Clone, PartialEq)]
pub struct SearchOutcome {
    pub results: Vec<SearchResult>,
    pub coverage: SearchCoverage,
    pub source_sets: Vec<SourceResultSet>,
    pub debug: Option<SearchDebugInfo>,
}

#[cfg(feature = "semantic")]
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SearchTimingBreakdown {
    pub splade_expand: Duration,
    pub aho_compile: Duration,
    pub semantic_scan: Duration,
    pub raw_scan: Duration,
    pub tantivy_query: Duration,
    pub dense_knn: Duration,
    pub rrf_fusion: Duration,
}

#[cfg(feature = "semantic")]
impl SearchTimingBreakdown {
    fn add(&mut self, other: &Self) {
        self.splade_expand += other.splade_expand;
        self.aho_compile += other.aho_compile;
        self.semantic_scan += other.semantic_scan;
        self.raw_scan += other.raw_scan;
        self.tantivy_query += other.tantivy_query;
        self.dense_knn += other.dense_knn;
        self.rrf_fusion += other.rrf_fusion;
    }
}

#[cfg(feature = "semantic")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchDebugInfo {
    pub plan_mode: String,
    pub timings: SearchTimingBreakdown,
}

#[cfg(feature = "semantic")]
#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct SemanticScanTiming {
    aho_compile: Duration,
    semantic_scan: Duration,
}

#[cfg(feature = "semantic")]
#[derive(Debug, Clone, PartialEq)]
pub struct SearchSnapshot {
    pub indexed_ids: roaring::RoaringTreemap,
    pub fresh_ids: roaring::RoaringTreemap,
    pub active_ids: roaring::RoaringTreemap,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct WalMetaRecord {
    pub(crate) wal_entry_id: u64,
    pub(crate) source_path: String,
    pub(crate) content_hash: String,
    pub(crate) byte_offset: u64,
    pub(crate) byte_length: u64,
    pub(crate) committed_at: u64,
    #[serde(default = "default_line_start")]
    pub(crate) line_range_start: usize,
    #[serde(default = "default_line_end")]
    pub(crate) line_range_end: usize,
    #[serde(skip, default)]
    pub(crate) content_type: ContentType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceManifestEntry {
    pub path: String,
    pub mtime_ms: u64,
    pub size: u64,
    pub blake3_hash: String,
    pub wal_entry_id: u64,
}

#[cfg(feature = "semantic")]
fn dominant_content_type(metadata: &[WalMetaRecord]) -> ContentType {
    let mut counts = [0usize; 6];
    for entry in metadata {
        match entry.content_type {
            ContentType::Code => counts[0] += 1,
            ContentType::Prose => counts[1] += 1,
            ContentType::Config => counts[2] += 1,
            ContentType::Log => counts[3] += 1,
            ContentType::Mixed => counts[4] += 1,
            ContentType::Unknown => counts[5] += 1,
        }
    }
    let (idx, _) = counts
        .iter()
        .enumerate()
        .max_by_key(|(_, count)| **count)
        .unwrap_or((5, &0));
    match idx {
        0 => ContentType::Code,
        1 => ContentType::Prose,
        2 => ContentType::Config,
        3 => ContentType::Log,
        4 => ContentType::Mixed,
        _ => ContentType::Unknown,
    }
}

impl Index {
    pub fn open_or_create(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        let root_created = ensure_dir(&root)?;

        let wal_dir = root.join("wal");
        let wal_dir_created = ensure_dir(&wal_dir)?;

        let wal_meta_path = wal_dir.join(WAL_META_FILE_NAME);
        let wal_meta_created = create_file_if_missing(&wal_meta_path)?;

        let wal_content_path = wal_dir.join(WAL_CONTENT_FILE_NAME);
        let wal_content_created = create_file_if_missing(&wal_content_path)?;

        let sources_dir = root.join("sources");
        let sources_dir_created = ensure_dir(&sources_dir)?;
        let manifest_path = sources_dir.join(MANIFEST_FILE_NAME);
        let manifest_created = create_file_if_missing(&manifest_path)?;

        if root_created
            || wal_dir_created
            || wal_meta_created
            || wal_content_created
            || sources_dir_created
            || manifest_created
        {
            sync_dir(&wal_dir)?;
            sync_dir(&sources_dir)?;
            sync_dir(&root)?;
        }

        let content_size = File::open(&wal_content_path)?.metadata()?.len();
        let metadata = load_and_recover_metadata_records(&wal_meta_path, content_size)?;
        let manifest = load_manifest(&manifest_path, !metadata.is_empty())?;

        Ok(Self {
            root,
            wal_dir,
            wal_meta_path,
            wal_content_path,
            sources_dir,
            manifest_path,
            metadata: Arc::new(RwLock::new(metadata)),
            manifest: Arc::new(RwLock::new(manifest)),
            #[cfg(feature = "semantic")]
            dense_embedder: Arc::new(RwLock::new(None)),
            #[cfg(feature = "semantic")]
            sparse_encoder: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    pub fn add_text(
        &self,
        source_path: impl Into<String>,
        content: impl Into<String>,
    ) -> Result<u64> {
        let source_path = source_path.into();
        let content_type = ContentType::from_path(&source_path);
        let content = content.into();
        let content_bytes = content.as_bytes();
        let content_hash = hash_content(content_bytes);
        let committed_at = unix_timestamp_ms();
        let line_count = content.lines().count().max(1);

        let mut metadata = self
            .metadata
            .write()
            .map_err(|_| SieveError::LockPoisoned)?;
        let wal_entry_id = metadata
            .last()
            .map(|record| record.wal_entry_id + 1)
            .unwrap_or(0);

        let byte_offset = append_content(&self.wal_content_path, content_bytes)?;
        let byte_length = content_bytes.len() as u64;

        let record = WalMetaRecord {
            wal_entry_id,
            source_path,
            content_hash,
            byte_offset,
            byte_length,
            committed_at,
            line_range_start: 1,
            line_range_end: line_count,
            content_type,
        };

        append_metadata(&self.wal_meta_path, &record)?;
        metadata.push(record.clone());
        drop(metadata);

        self.record_source_entry(
            record.source_path.clone(),
            record.committed_at,
            record.byte_length,
            record.content_hash.clone(),
            wal_entry_id,
        )?;
        self.save_manifest()?;

        Ok(wal_entry_id)
    }

    pub fn search(&self, query: &str, options: SearchOptions) -> Result<Vec<SearchResult>> {
        #[cfg(feature = "semantic")]
        {
            self.search_semantic_internal(query, options, false)
                .map(|outcome| outcome.results)
        }

        #[cfg(not(feature = "semantic"))]
        {
            self.search_legacy(query, options)
        }
    }

    #[cfg(feature = "semantic")]
    pub fn search_with_outcome(
        &self,
        query: &str,
        options: SearchOptions,
    ) -> Result<SearchOutcome> {
        self.search_semantic_internal(query, options, true)
    }

    #[cfg(feature = "semantic")]
    pub fn warm_search_models(&self, include_dense: bool) -> Result<()> {
        let _ = self.load_sparse_encoder()?;
        if include_dense {
            let _ = self.load_embedder()?;
        }
        Ok(())
    }

    #[cfg(feature = "semantic")]
    fn search_semantic_internal(
        &self,
        query: &str,
        options: SearchOptions,
        include_coverage: bool,
    ) -> Result<SearchOutcome> {
        let mut timings = SearchTimingBreakdown::default();
        let empty_outcome = |plan_mode: &str, timings: SearchTimingBreakdown| SearchOutcome {
            results: Vec::new(),
            coverage: SearchCoverage {
                total_chunks: 0,
                embedded_chunks: 0,
                delta_chunks: 0,
                skipped_due_to_budget: false,
            },
            source_sets: Vec::new(),
            debug: Some(SearchDebugInfo {
                plan_mode: plan_mode.to_string(),
                timings,
            }),
        };

        if query.is_empty() {
            return Ok(empty_outcome("empty", timings));
        }

        let top_k = options.top_k.unwrap_or(10);
        let metadata = self.metadata_snapshot()?;
        if metadata.is_empty() {
            return Ok(empty_outcome("empty", timings));
        }

        let snapshot = self.snapshot_search_partition()?;
        let active_ids = self.active_wal_entry_ids()?;
        let manifest_authoritative = self.manifest_path.metadata()?.len() > 0;
        let active_metadata: Vec<WalMetaRecord> = metadata
            .into_iter()
            .filter(|entry| {
                if manifest_authoritative {
                    active_ids.contains(&entry.wal_entry_id)
                } else {
                    true
                }
            })
            .collect();
        if active_metadata.is_empty() {
            return Ok(empty_outcome("empty", timings));
        }
        let fresh_metadata: Vec<WalMetaRecord> = active_metadata
            .iter()
            .filter(|entry| snapshot.fresh_ids.contains(entry.wal_entry_id))
            .cloned()
            .collect();

        let shards_dir = self.root.join("segments");
        let aliases = AliasLexicon::built_in();
        let content_type = dominant_content_type(&active_metadata);
        let plan_started = Instant::now();
        let plan = if (query.starts_with('/') && query.ends_with('/') && query.len() >= 2)
            || (query.starts_with('"') && query.ends_with('"') && query.len() >= 2)
        {
            plan_query(query, None, &aliases, &options, content_type)
        } else {
            let sparse = self.load_sparse_encoder_for_content_type(content_type)?;
            plan_query(query, sparse.as_deref(), &aliases, &options, content_type)
        };
        timings.splade_expand = plan_started.elapsed();
        let coverage = if include_coverage {
            self.search_coverage(&active_metadata)?
        } else {
            SearchCoverage {
                total_chunks: 0,
                embedded_chunks: 0,
                delta_chunks: 0,
                skipped_due_to_budget: false,
            }
        };

        let plan_mode = match &plan {
            QueryPlan::Regex(_) => "regex".to_string(),
            QueryPlan::Exact(_) => "exact".to_string(),
            QueryPlan::Semantic(_) => {
                if options.fresh_only {
                    "semantic:fresh-only".to_string()
                } else {
                    "semantic".to_string()
                }
            }
            QueryPlan::Lexical(_) => "lexical".to_string(),
        };

        let (results, source_sets) = match plan {
            QueryPlan::Regex(pattern) => {
                tracing::debug!(query = %query, pattern = %pattern, "running regex scan layer");
                (
                    scan_regex_results(
                        &self.wal_content_path,
                        &active_metadata,
                        &pattern,
                        ResultSource::RawScan,
                    )?,
                    Vec::new(),
                )
            }
            QueryPlan::Exact(phrase) => {
                let (scan, lexical) = join(
                    || {
                        scan_substring_results(
                            &self.wal_content_path,
                            &active_metadata,
                            phrase.as_bytes(),
                            ResultSource::RawScan,
                        )
                    },
                    || search_lexical_with_fallback(&shards_dir, query, top_k),
                );
                let mut scan = scan?;
                let lexical = lexical?;
                if lexical.skipped_due_to_parse_failure {
                    tag_scan_results_as_fallback(&mut scan);
                }
                let lexical = filter_lexical_matches(lexical.matches, &active_ids);
                let source_sets = vec![
                    SourceResultSet {
                        source: ResultSource::RawScan,
                        weight: 0.90,
                        coverage: CoverageState::Complete,
                        results: scan.clone(),
                    },
                    SourceResultSet {
                        source: ResultSource::LexicalBm25,
                        weight: 1.00,
                        coverage: CoverageState::Complete,
                        results: lexical_to_scored(lexical),
                    },
                ];
                let rrf_started = Instant::now();
                let fused = fuse_source_sets(&source_sets, 20.0);
                timings.rrf_fusion += rrf_started.elapsed();
                (fused, source_sets)
            }
            QueryPlan::Semantic(semantic_query) => {
                let raw_scan_started = Instant::now();
                let scan_metadata = if options.fresh_only {
                    &fresh_metadata
                } else {
                    &active_metadata
                };
                let scan = scan_query_results(&self.wal_content_path, scan_metadata, query)?;
                timings.raw_scan += raw_scan_started.elapsed();
                let semantic_outcome = self.search_semantic_query(
                    semantic_query.as_ref(),
                    SearchOptions {
                        top_k: Some(top_k),
                        fresh_only: options.fresh_only,
                        experimental_rerank: options.experimental_rerank,
                        no_expand: options.no_expand,
                        no_window_scoring: options.no_window_scoring,
                        no_df_filter: options.no_df_filter,
                        random_expansion: options.random_expansion,
                        allow_delta_fallback: options.allow_delta_fallback,
                        query_promoted_dense: options.query_promoted_dense.clone(),
                    },
                )?;
                if let Some(debug) = &semantic_outcome.debug {
                    timings.add(&debug.timings);
                }
                let mut source_sets = Vec::new();
                if !scan.is_empty() {
                    source_sets.push(SourceResultSet {
                        source: ResultSource::RawScan,
                        weight: 0.90,
                        coverage: CoverageState::Complete,
                        results: scan,
                    });
                }
                source_sets.extend(semantic_outcome.source_sets.clone());
                if !options.fresh_only {
                    let dense_started = Instant::now();
                    let dense_sets = self.semantic_dense_result_sets(
                        query,
                        &active_metadata,
                        &active_ids,
                        top_k,
                        options.allow_delta_fallback,
                        &options.query_promoted_dense,
                        &source_sets,
                    )?;
                    timings.dense_knn += dense_started.elapsed();
                    source_sets.extend(dense_sets);
                }
                let fresh_id_set: HashSet<u64> = snapshot.fresh_ids.iter().collect();
                let source_sets = filter_stale_only_source_sets(source_sets, &fresh_id_set, false);
                let rrf_started = Instant::now();
                let fused = fuse_source_sets(&source_sets, 20.0);
                timings.rrf_fusion += rrf_started.elapsed();
                (fused, source_sets)
            }
            QueryPlan::Lexical(query_text) => {
                let (scan, lexical) = join(
                    || {
                        tracing::debug!(query = %query_text, "running default scan layer");
                        scan_query_results(&self.wal_content_path, &active_metadata, &query_text)
                    },
                    || search_lexical_with_fallback(&shards_dir, &query_text, top_k),
                );
                let mut scan = scan?;
                let lexical = lexical?;
                if lexical.skipped_due_to_parse_failure {
                    tag_scan_results_as_fallback(&mut scan);
                }
                let lexical = filter_lexical_matches(lexical.matches, &active_ids);
                let lexical_results = lexical_to_scored(lexical);
                let mut source_sets = vec![
                    SourceResultSet {
                        source: ResultSource::RawScan,
                        weight: 0.90,
                        coverage: CoverageState::Complete,
                        results: scan,
                    },
                    SourceResultSet {
                        source: ResultSource::LexicalBm25,
                        weight: 1.00,
                        coverage: CoverageState::Complete,
                        results: lexical_results,
                    },
                ];
                let dense_started = Instant::now();
                let dense_sets = self.semantic_dense_result_sets(
                    &query_text,
                    &active_metadata,
                    &active_ids,
                    top_k,
                    options.allow_delta_fallback,
                    &options.query_promoted_dense,
                    &source_sets,
                )?;
                timings.dense_knn += dense_started.elapsed();
                source_sets.extend(dense_sets);

                let rrf_started = Instant::now();
                let fused = fuse_source_sets(&source_sets, 20.0);
                timings.rrf_fusion += rrf_started.elapsed();
                (fused, source_sets)
            }
        };

        Ok(SearchOutcome {
            results: self.finalize_search_results(query, results, top_k),
            coverage,
            source_sets,
            debug: Some(SearchDebugInfo { plan_mode, timings }),
        })
    }

    #[cfg(not(feature = "semantic"))]
    fn search_legacy(&self, query: &str, options: SearchOptions) -> Result<Vec<SearchResult>> {
        if query.is_empty() {
            return Ok(Vec::new());
        }
        let top_k = options.top_k.unwrap_or(10);
        let metadata = self.metadata_snapshot()?;
        if metadata.is_empty() {
            return Ok(Vec::new());
        }
        let active_ids = self.active_wal_entry_ids()?;
        let manifest_authoritative = self.manifest_path.metadata()?.len() > 0;
        let active_metadata: Vec<WalMetaRecord> = metadata
            .into_iter()
            .filter(|entry| {
                if manifest_authoritative {
                    active_ids.contains(&entry.wal_entry_id)
                } else {
                    true
                }
            })
            .collect();
        if active_metadata.is_empty() {
            return Ok(Vec::new());
        }
        let shards_dir = self.root.join("segments");
        let query_kind = QueryKind::from_query(query);
        let results = match query_kind {
            QueryKind::Regex(pattern) => scan_regex_results(
                &self.wal_content_path,
                &active_metadata,
                &pattern,
                ResultSource::RawScan,
            )?,
            QueryKind::ExactPhrase(phrase) => {
                let (scan, lexical) = join(
                    || {
                        scan_substring_results(
                            &self.wal_content_path,
                            &active_metadata,
                            phrase.as_bytes(),
                            ResultSource::RawScan,
                        )
                    },
                    || search_lexical_with_fallback(&shards_dir, query, top_k),
                );
                let mut scan = scan?;
                let lexical = lexical?;
                if lexical.skipped_due_to_parse_failure {
                    tag_scan_results_as_fallback(&mut scan);
                }
                let lexical = filter_lexical_matches(lexical.matches, &active_ids);
                weighted_rrf_fuse(
                    vec![
                        (ResultSource::RawScan, 0.90, scan),
                        (ResultSource::LexicalBm25, 1.00, lexical_to_scored(lexical)),
                    ],
                    20.0,
                )
            }
            QueryKind::Default => {
                let (scan, lexical) = join(
                    || scan_query_results(&self.wal_content_path, &active_metadata, query),
                    || search_lexical_with_fallback(&shards_dir, query, top_k),
                );
                let mut scan = scan?;
                let lexical = lexical?;
                if lexical.skipped_due_to_parse_failure {
                    tag_scan_results_as_fallback(&mut scan);
                }
                let lexical = filter_lexical_matches(lexical.matches, &active_ids);
                weighted_rrf_fuse(
                    vec![
                        (ResultSource::RawScan, 0.90, scan),
                        (ResultSource::LexicalBm25, 1.00, lexical_to_scored(lexical)),
                    ],
                    20.0,
                )
            }
        };
        Ok(self.finalize_search_results(query, results, top_k))
    }

    pub fn read_entry_content(&self, wal_entry_id: u64) -> Result<String> {
        let metadata = self.metadata_snapshot()?;
        let entry = metadata
            .into_iter()
            .find(|entry| entry.wal_entry_id == wal_entry_id)
            .ok_or_else(|| {
                SieveError::Io(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("wal entry not found: {wal_entry_id}"),
                ))
            })?;

        let content_file = File::open(&self.wal_content_path)?;
        content_file.lock_shared()?;
        let mmap = unsafe { Mmap::map(&content_file)? };
        let start = entry.byte_offset as usize;
        let end = start + entry.byte_length as usize;
        let content = if end <= mmap.len() {
            String::from_utf8_lossy(&mmap[start..end]).into_owned()
        } else {
            String::new()
        };
        content_file.unlock()?;
        Ok(content)
    }

    pub fn should_skip_source(&self, path: &str, mtime_ms: u64, size: u64) -> Result<bool> {
        let manifest = self.manifest.read().map_err(|_| SieveError::LockPoisoned)?;
        Ok(manifest
            .get(path)
            .is_some_and(|entry| entry.mtime_ms == mtime_ms && entry.size == size))
    }

    pub fn record_source_entry(
        &self,
        path: String,
        mtime_ms: u64,
        size: u64,
        blake3_hash: String,
        wal_entry_id: u64,
    ) -> Result<()> {
        let mut manifest = self
            .manifest
            .write()
            .map_err(|_| SieveError::LockPoisoned)?;
        manifest.insert(
            path.clone(),
            SourceManifestEntry {
                path,
                mtime_ms,
                size,
                blake3_hash,
                wal_entry_id,
            },
        );
        Ok(())
    }

    pub fn prune_manifest_to_paths(&self, paths: &HashSet<String>) -> Result<()> {
        let mut manifest = self
            .manifest
            .write()
            .map_err(|_| SieveError::LockPoisoned)?;
        manifest.retain(|path, _| paths.contains(path));
        Ok(())
    }

    pub fn save_manifest(&self) -> Result<()> {
        let manifest = self.manifest.read().map_err(|_| SieveError::LockPoisoned)?;
        let mut entries: Vec<SourceManifestEntry> = manifest.values().cloned().collect();
        entries.sort_by(|a, b| a.path.cmp(&b.path));
        let serialized = serde_json::to_vec_pretty(&entries)?;
        let mut file = OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(&self.manifest_path)?;
        file.write_all(&serialized)?;
        file.flush()?;
        file.sync_data()?;
        Ok(())
    }

    pub fn wal_dir(&self) -> &Path {
        &self.wal_dir
    }

    pub fn sources_dir(&self) -> &Path {
        &self.sources_dir
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub(crate) fn wal_content_path(&self) -> &Path {
        &self.wal_content_path
    }

    pub(crate) fn metadata_snapshot(&self) -> Result<Vec<WalMetaRecord>> {
        Ok(self
            .metadata
            .read()
            .map_err(|_| SieveError::LockPoisoned)?
            .clone())
    }

    pub fn wal_entries_count(&self) -> Result<usize> {
        Ok(self
            .metadata
            .read()
            .map_err(|_| SieveError::LockPoisoned)?
            .len())
    }

    pub fn chunk_count(&self) -> Result<usize> {
        Ok(self
            .metadata_snapshot()?
            .iter()
            .map(|entry| {
                self.read_entry_content(entry.wal_entry_id)
                    .map(|content| chunk_entry(entry, &content).len())
                    .unwrap_or(0)
            })
            .sum())
    }

    pub fn active_wal_entry_ids(&self) -> Result<HashSet<u64>> {
        let manifest = self.manifest.read().map_err(|_| SieveError::LockPoisoned)?;
        Ok(manifest.values().map(|entry| entry.wal_entry_id).collect())
    }

    #[cfg(feature = "semantic")]
    pub fn snapshot_search_partition(&self) -> Result<SearchSnapshot> {
        let active_hash = self.active_wal_entry_ids()?;
        let mut active_ids = roaring::RoaringTreemap::new();
        for wal_entry_id in active_hash {
            active_ids.insert(wal_entry_id);
        }
        let indexed_all = load_indexed_entries(&self.root.join("segments"))?;
        let indexed_ids = &indexed_all & &active_ids;
        let fresh_ids = &active_ids - &indexed_ids;
        Ok(SearchSnapshot {
            indexed_ids,
            fresh_ids,
            active_ids,
        })
    }

    #[cfg(feature = "semantic")]
    pub fn semantic_status(&self) -> Result<SemanticStatus> {
        let total_chunks = self.total_chunk_count()?;
        let model_manager = self.model_manager();
        let model_dir = model_manager.model_dir(DEFAULT_MODEL_NAME);
        let model_cached = model_manager.is_cached(DEFAULT_MODEL_NAME);
        let dimension = self
            .load_embedder()?
            .as_ref()
            .map(|embedder| embedder.dimension())
            .unwrap_or(384);
        let vectors = if self.root.join("vectors").exists() {
            HotVectorStore::open_or_create(&self.root.join("vectors"), dimension)?.len()
        } else {
            0
        };
        Ok(SemanticStatus {
            model_cached,
            model_dir,
            vectors,
            dimension,
            total_chunks,
        })
    }

    #[cfg(feature = "semantic")]
    pub fn delta_fallback_over_budget(&self, max_chunks: usize) -> Result<bool> {
        let status = self.semantic_status()?;
        Ok(status.total_chunks.saturating_sub(status.vectors) > max_chunks)
    }

    #[cfg(feature = "semantic")]
    pub fn embed_pending(&self, batch_size: usize) -> Result<usize> {
        let Some(embedder) = self.load_embedder()? else {
            tracing::debug!("semantic embedder unavailable; skipping embedding pass");
            return Ok(0);
        };
        let mut store =
            HotVectorStore::open_or_create(&self.root.join("vectors"), embedder.dimension())?;
        let pending: Vec<WalMetaRecord> = self
            .metadata_snapshot()?
            .into_iter()
            .filter(|entry| !store.embedded_set().contains(entry.wal_entry_id))
            .collect();
        tracing::debug!(
            pending = pending.len(),
            batch_size,
            "starting embedding pass"
        );
        if pending.is_empty() {
            return Ok(0);
        }

        let mut embedded = 0usize;
        for chunk in pending.chunks(batch_size.max(1)) {
            let chunk_ids: Vec<u64> = chunk.iter().map(|entry| entry.wal_entry_id).collect();
            tracing::debug!(?chunk_ids, "embedding wal chunk batch");
            let chunk_batches: Result<Vec<Vec<SourceChunk>>> = chunk
                .iter()
                .map(|entry| {
                    self.read_entry_content(entry.wal_entry_id)
                        .map(|content| chunk_entry(entry, &content))
                })
                .collect();
            let chunk_records: Vec<SourceChunk> = chunk_batches?.into_iter().flatten().collect();
            let refs: Vec<&str> = chunk_records
                .iter()
                .map(|entry| entry.chunk.text.as_str())
                .collect();
            let vectors = embedder.embed_batch(&refs)?;
            let metas: Vec<VectorMeta> = chunk_records
                .iter()
                .map(|chunk| VectorMeta {
                    wal_entry_id: chunk.chunk.wal_entry_id,
                    chunk_id: chunk.chunk.chunk_id,
                    source_path: chunk.source_path.clone(),
                    byte_range: (chunk.chunk.byte_start, chunk.chunk.byte_end),
                    line_range: chunk.chunk.line_range,
                })
                .collect();
            store.append(&vectors, &metas)?;
            embedded += metas.len();
        }
        tracing::debug!(embedded, "completed embedding pass");
        Ok(embedded)
    }

    #[cfg(feature = "semantic")]
    fn model_manager(&self) -> ModelManager {
        ModelManager::new(&default_sieve_data_dir())
    }

    #[cfg(feature = "semantic")]
    fn load_embedder(&self) -> Result<Option<Arc<crate::embed::Embedder>>> {
        if let Some(cached) = self
            .dense_embedder
            .read()
            .map_err(|_| SieveError::LockPoisoned)?
            .clone()
        {
            return Ok(Some(cached));
        }

        let manager = self.model_manager();
        if !manager.is_cached(DEFAULT_MODEL_NAME) {
            return Ok(None);
        }
        let handle = manager.ensure_dense_model()?;
        let embedder = Arc::new(crate::embed::Embedder::load(
            &handle.query_model_path,
            &handle.doc_model_path,
        )?);
        *self
            .dense_embedder
            .write()
            .map_err(|_| SieveError::LockPoisoned)? = Some(Arc::clone(&embedder));
        Ok(Some(embedder))
    }

    #[cfg(feature = "semantic")]
    fn total_chunk_count(&self) -> Result<usize> {
        self.chunk_count()
    }

    #[cfg(feature = "semantic")]
    pub fn search_semantic_query(
        &self,
        query: &SemanticQuery,
        options: SearchOptions,
    ) -> Result<SearchOutcome> {
        let mut timings = SearchTimingBreakdown::default();
        let top_k = options.top_k.unwrap_or(10);
        let snapshot = self.snapshot_search_partition()?;
        let metadata = self.metadata_snapshot()?;
        if metadata.is_empty() || snapshot.active_ids.is_empty() {
            return Ok(SearchOutcome {
                results: Vec::new(),
                coverage: SearchCoverage {
                    total_chunks: 0,
                    embedded_chunks: 0,
                    delta_chunks: 0,
                    skipped_due_to_budget: false,
                },
                source_sets: Vec::new(),
                debug: Some(SearchDebugInfo {
                    plan_mode: if options.fresh_only {
                        "semantic:fresh-only".to_string()
                    } else {
                        "semantic".to_string()
                    },
                    timings,
                }),
            });
        }

        let active_metadata: Vec<WalMetaRecord> = metadata
            .into_iter()
            .filter(|entry| snapshot.active_ids.contains(entry.wal_entry_id))
            .collect();
        let fresh_metadata: Vec<WalMetaRecord> = active_metadata
            .iter()
            .filter(|entry| snapshot.fresh_ids.contains(entry.wal_entry_id))
            .cloned()
            .collect();

        let total_active_entries = active_metadata.len();
        let indexed_entry_count = snapshot.indexed_ids.iter().count();
        let mut source_sets = Vec::new();
        if !options.fresh_only && !snapshot.indexed_ids.is_empty() {
            let tantivy_started = Instant::now();
            let lexical = search_semantic_lexical(&self.root.join("segments"), query, top_k)?;
            timings.tantivy_query += tantivy_started.elapsed();
            let lexical = filter_lexical_matches_bitmap(lexical, &snapshot.indexed_ids);
            let results = lexical_to_scored_with_source(lexical, ResultSource::SpladeBm25);
            if !results.is_empty() {
                source_sets.push(SourceResultSet {
                    source: ResultSource::SpladeBm25,
                    weight: 1.10,
                    coverage: coverage_state_from_entry_counts(
                        indexed_entry_count,
                        total_active_entries,
                    ),
                    results,
                });
            }
        }
        if !fresh_metadata.is_empty() {
            let (semantic_results, scan_timing) = semantic_scan_results_with_timing(
                &self.wal_content_path,
                &fresh_metadata,
                query,
                crate::semantic_scan::SemanticScanOptions {
                    no_df_filter: options.no_df_filter,
                    no_window_scoring: options.no_window_scoring,
                },
            )?;
            timings.aho_compile += scan_timing.aho_compile;
            timings.semantic_scan += scan_timing.semantic_scan;
            if !semantic_results.is_empty() {
                source_sets.push(SourceResultSet {
                    source: ResultSource::SemanticScan,
                    weight: 1.10,
                    coverage: CoverageState::Complete,
                    results: semantic_results,
                });
            }
        }

        let mut preview = Vec::new();
        for set in &source_sets {
            preview.extend(set.results.iter().cloned());
        }
        if options.experimental_rerank && self.should_event_rerank(query, preview.len())? {
            if let Ok(reranked) = self.try_event_rerank_results(query, &fresh_metadata, &preview) {
                if !reranked.is_empty() {
                    source_sets.push(SourceResultSet {
                        source: ResultSource::EventReranked,
                        weight: 1.0,
                        coverage: CoverageState::Complete,
                        results: reranked,
                    });
                }
            }
        }

        let fused = fuse_source_sets(&source_sets, 20.0);
        let semantic_results = apply_dense_recency_bonus(
            collapse_fused_results_by_file(fused),
            &active_metadata
                .iter()
                .map(|entry| entry.wal_entry_id)
                .collect::<Vec<_>>(),
        );
        Ok(SearchOutcome {
            results: self.finalize_search_results(&query.normalized_query, semantic_results, top_k),
            coverage: SearchCoverage {
                total_chunks: 0,
                embedded_chunks: 0,
                delta_chunks: 0,
                skipped_due_to_budget: false,
            },
            source_sets,
            debug: Some(SearchDebugInfo {
                plan_mode: if options.fresh_only {
                    "semantic:fresh-only".to_string()
                } else {
                    "semantic".to_string()
                },
                timings,
            }),
        })
    }

    fn finalize_search_results(
        &self,
        query: &str,
        results: Vec<ScoredResult>,
        top_k: usize,
    ) -> Vec<SearchResult> {
        results
            .into_iter()
            .take(top_k)
            .map(|result| {
                let (line_range, snippet) = if result.snippet.is_empty() {
                    self.read_entry_content(result.wal_entry_id)
                        .ok()
                        .map(|content| {
                            #[cfg(feature = "semantic")]
                            if matches!(
                                result.source_layer,
                                ResultSource::HotVector
                                    | ResultSource::DeltaFallback
                                    | ResultSource::QueryPromoted
                                    | ResultSource::SemanticScan
                            ) {
                                snippet_from_byte_range(
                                    &content,
                                    (result.result_id.byte_start, result.result_id.byte_end),
                                )
                                .map(|snippet| (result.line_range, snippet))
                                .unwrap_or_else(|_| {
                                    snippet_for_query(
                                        &content,
                                        result.line_range,
                                        query,
                                        result.source_layer,
                                    )
                                })
                            } else {
                                snippet_for_query(
                                    &content,
                                    result.line_range,
                                    query,
                                    result.source_layer,
                                )
                            }
                            #[cfg(not(feature = "semantic"))]
                            {
                                snippet_for_query(
                                    &content,
                                    result.line_range,
                                    query,
                                    result.source_layer,
                                )
                            }
                        })
                        .unwrap_or((result.line_range, String::new()))
                } else {
                    (result.line_range, result.snippet.clone())
                };
                SearchResult {
                    line_number: line_range.0,
                    line_range,
                    chunk_id: result.chunk_id,
                    byte_range: (result.result_id.byte_start, result.result_id.byte_end),
                    source_path: result.source_path,
                    snippet,
                    score: result.score,
                    source_layer: result.source_layer,
                    wal_entry_id: result.wal_entry_id,
                }
            })
            .collect()
    }

    #[cfg(feature = "semantic")]
    fn should_event_rerank(&self, query: &SemanticQuery, candidate_count: usize) -> Result<bool> {
        if !crate::event_rerank::should_rerank_event_windows(query, candidate_count, true) {
            return Ok(false);
        }
        Ok(self.model_manager().registry()?.event_reranker.is_some())
    }

    #[cfg(feature = "semantic")]
    fn try_event_rerank_results(
        &self,
        query: &SemanticQuery,
        fresh_metadata: &[WalMetaRecord],
        existing: &[ScoredResult],
    ) -> Result<Vec<ScoredResult>> {
        if fresh_metadata.is_empty() {
            return Ok(Vec::new());
        }
        let Some(handle) = self.model_manager().registry()?.event_reranker else {
            return Ok(Vec::new());
        };
        let reranker = match crate::event_rerank::EventReranker::load(&handle.model_path) {
            Ok(reranker) => reranker,
            Err(_) => return Ok(Vec::new()),
        };

        let (windows, _scan_timing) = semantic_scan_scored_windows(
            &self.wal_content_path,
            fresh_metadata,
            query,
            SemanticScanOptions::default(),
        )?;
        if windows.is_empty() {
            return Ok(Vec::new());
        }

        let mut accumulators: Vec<crate::semantic_scan::WindowAccumulator> = windows
            .iter()
            .take(crate::event_rerank::MAX_RERANK_WINDOWS)
            .map(|(window, _)| crate::semantic_scan::WindowAccumulator {
                wal_entry_id: window.wal_entry_id,
                window_start: window.window_start,
                window_end: window.window_end,
                events: window.events.clone(),
                has_anchor: window.has_anchor,
            })
            .collect();
        let mut reranked: Vec<ScoredResult> = windows
            .into_iter()
            .take(crate::event_rerank::MAX_RERANK_WINDOWS)
            .map(|(_, scored)| scored)
            .collect();
        let formula_scores: Vec<f32> = reranked.iter().map(|scored| scored.score as f32).collect();
        let idf = vec![1.0; query.terms.len().max(1)];
        let updated_scores = match reranker.rerank(&mut accumulators, query, &idf, &formula_scores)
        {
            Ok(scores) => scores,
            Err(_) => return Ok(Vec::new()),
        };

        let existing_by_id: HashMap<ResultId, &ScoredResult> = existing
            .iter()
            .map(|result| (result.result_id, result))
            .collect();
        for (idx, candidate) in reranked.iter_mut().enumerate() {
            if let Some(score) = updated_scores.get(idx) {
                candidate.score = *score as f64;
            }
            if let Some(existing) = existing_by_id.get(&candidate.result_id) {
                candidate.score = candidate.score.max(existing.score);
            }
            candidate.source_layer = ResultSource::EventReranked;
        }
        reranked.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(reranked)
    }

    #[cfg(feature = "semantic")]
    fn search_coverage(&self, active_metadata: &[WalMetaRecord]) -> Result<SearchCoverage> {
        let total_chunks: usize = active_metadata
            .iter()
            .map(|entry| {
                self.read_entry_content(entry.wal_entry_id)
                    .map(|content| chunk_entry(entry, &content).len())
                    .unwrap_or(0)
            })
            .sum();
        let embedded_chunks = if self.root.join("vectors").exists() {
            let dimension = self
                .load_embedder()?
                .as_ref()
                .map(|embedder| embedder.dimension())
                .unwrap_or(384);
            HotVectorStore::open_or_create(&self.root.join("vectors"), dimension)?.len()
        } else {
            0
        };
        let delta_chunks = total_chunks.saturating_sub(embedded_chunks);
        Ok(SearchCoverage {
            total_chunks,
            embedded_chunks,
            delta_chunks,
            skipped_due_to_budget: delta_chunks > 50,
        })
    }

    #[cfg(feature = "semantic")]
    #[allow(clippy::too_many_arguments)]
    fn semantic_dense_result_sets(
        &self,
        query: &str,
        active_metadata: &[WalMetaRecord],
        active_ids: &HashSet<u64>,
        top_k: usize,
        allow_delta_fallback: bool,
        query_promoted: &QueryPromotedDense,
        existing_source_sets: &[SourceResultSet],
    ) -> Result<Vec<SourceResultSet>> {
        if !should_use_semantic(query) {
            return Ok(Vec::new());
        }
        let Some(embedder) = self.load_embedder()? else {
            return Ok(Vec::new());
        };

        let query_vec = embedder.embed_query(query)?;
        let total_active_entries = active_metadata.len();
        let mut embedded_set = roaring::RoaringTreemap::new();
        let mut semantic_sets = Vec::new();
        if self.root.join("vectors").exists() {
            let store =
                HotVectorStore::open_or_create(&self.root.join("vectors"), embedder.dimension())?;
            embedded_set = store.embedded_set().clone();
            let vector_matches = filter_vector_matches(
                store.search_knn(&query_vec, top_k.saturating_mul(8).max(top_k))?,
                active_ids,
            );
            let vector_matches = collapse_vector_matches_by_file(vector_matches, top_k);
            if !vector_matches.is_empty() {
                let covered_entries = embedded_set.iter().count();
                let wal_order: Vec<u64> = active_metadata
                    .iter()
                    .map(|entry| entry.wal_entry_id)
                    .collect();
                let dense_results =
                    apply_dense_recency_bonus(vector_matches_to_scored(vector_matches), &wal_order);
                semantic_sets.push(SourceResultSet {
                    source: ResultSource::HotVector,
                    weight: 1.00,
                    coverage: coverage_state_from_entry_counts(
                        covered_entries,
                        total_active_entries,
                    ),
                    results: dense_results,
                });
            }
        }
        let delta_entries: Vec<WalMetaRecord> = active_metadata
            .iter()
            .filter(|entry| !embedded_set.contains(entry.wal_entry_id))
            .cloned()
            .collect();

        if !delta_entries.is_empty() && query_promoted.max_promoted_chunks > 0 {
            let delta_chunks: Vec<SourceChunk> = delta_entries
                .iter()
                .map(|entry| {
                    self.read_entry_content(entry.wal_entry_id)
                        .map(|content| chunk_entry(entry, &content))
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flatten()
                .collect();
            let scan_seed_results: Vec<ScoredResult> = existing_source_sets
                .iter()
                .filter(|set| {
                    matches!(
                        set.source,
                        ResultSource::RawScan | ResultSource::ScanFallback
                    )
                })
                .flat_map(|set| set.results.clone())
                .collect();
            let promoted_chunks = select_query_promoted_chunks(
                &scan_seed_results,
                &delta_chunks,
                &embedded_set,
                query_promoted.max_promoted_chunks.max(1),
                query_promoted,
            );
            let promoted_results = collapse_scored_results_by_file(
                query_promoted_dense_results_with(
                    &query_vec,
                    &promoted_chunks,
                    top_k.saturating_mul(8).max(top_k),
                    query_promoted,
                    |text| embedder.embed_one(text),
                )?,
                top_k,
            );
            if !promoted_results.is_empty() {
                let promoted_entries: HashSet<u64> = promoted_chunks
                    .iter()
                    .map(|chunk| chunk.chunk.wal_entry_id)
                    .collect();
                semantic_sets.push(SourceResultSet {
                    source: ResultSource::QueryPromoted,
                    weight: 1.00,
                    coverage: coverage_state_from_entry_counts(
                        promoted_entries.len(),
                        delta_entries.len(),
                    ),
                    results: promoted_results,
                });
            }
        }

        if allow_delta_fallback && !delta_entries.is_empty() && delta_entries.len() <= 50 {
            let chunk_batches: Result<Vec<Vec<SourceChunk>>> = delta_entries
                .iter()
                .map(|entry| {
                    self.read_entry_content(entry.wal_entry_id)
                        .map(|content| chunk_entry(entry, &content))
                })
                .collect();
            let delta_chunks: Vec<SourceChunk> = chunk_batches?.into_iter().flatten().collect();
            let refs: Vec<&str> = delta_chunks
                .iter()
                .map(|chunk| chunk.chunk.text.as_str())
                .collect();
            let vectors = embedder.embed_batch(&refs)?;
            let delta = collapse_scored_results_by_file(
                score_delta_vectors(
                    &query_vec,
                    &delta_chunks,
                    &vectors,
                    top_k.saturating_mul(8).max(top_k),
                ),
                top_k,
            );
            if !delta.is_empty() {
                semantic_sets.push(SourceResultSet {
                    source: ResultSource::DeltaFallback,
                    weight: 0.85,
                    coverage: coverage_state_from_entry_counts(
                        delta_entries.len(),
                        total_active_entries,
                    ),
                    results: delta,
                });
            }
        }
        Ok(semantic_sets)
    }

    #[cfg(feature = "semantic")]
    fn load_sparse_encoder(&self) -> Result<Option<Arc<crate::sparse::SpladeEncoder>>> {
        self.load_sparse_encoder_for_content_type(ContentType::Prose)
    }

    #[cfg(feature = "semantic")]
    fn load_sparse_encoder_for_content_type(
        &self,
        content_type: ContentType,
    ) -> Result<Option<Arc<crate::sparse::SpladeEncoder>>> {
        let manager = self.model_manager();
        let registry = manager.registry()?;
        let route = crate::model::select_sparse_route(
            content_type,
            registry.sparse_code.is_some(),
            registry.sparse.is_some(),
        );
        if let Some(cached) = self
            .sparse_encoder
            .read()
            .map_err(|_| SieveError::LockPoisoned)?
            .get(&route.route)
            .cloned()
        {
            return Ok(Some(cached));
        }
        if route.warned_fallback {
            tracing::warn!(
                content_type = ?content_type,
                "SPLADE-Code unavailable; falling back to generic SPLADE"
            );
        }

        let handle = match route.route {
            crate::model::SparseRoute::CodeSplade => match manager.ensure_code_sparse_model() {
                Ok(handle) => handle,
                Err(err) => {
                    tracing::debug!(error = %err, "code sparse model unavailable; falling back to lexical planning");
                    return Ok(None);
                }
            },
            crate::model::SparseRoute::GenericSplade => {
                if !manager.is_cached(DEFAULT_SPARSE_MODEL_NAME) {
                    return Ok(None);
                }
                match manager.ensure_sparse_model() {
                    Ok(handle) => handle,
                    Err(err) => {
                        tracing::debug!(error = %err, "sparse model unavailable; falling back to lexical planning");
                        return Ok(None);
                    }
                }
            }
        };

        match crate::sparse::SpladeEncoder::load(&handle.model_path, &handle.tokenizer_path) {
            Ok(encoder) => {
                let encoder = Arc::new(encoder);
                self.sparse_encoder
                    .write()
                    .map_err(|_| SieveError::LockPoisoned)?
                    .insert(route.route, Arc::clone(&encoder));
                Ok(Some(encoder))
            }
            Err(err) => {
                tracing::debug!(error = %err, model = %handle.name, "failed to load SPLADE encoder; falling back to lexical planning");
                Ok(None)
            }
        }
    }
}

#[cfg(not(feature = "semantic"))]
enum QueryKind<'a> {
    Regex(String),
    ExactPhrase(&'a str),
    Default,
}

#[cfg(not(feature = "semantic"))]
impl<'a> QueryKind<'a> {
    fn from_query(query: &'a str) -> Self {
        if query.starts_with('/') && query.ends_with('/') && query.len() >= 2 {
            return Self::Regex(query[1..query.len() - 1].to_string());
        }
        if query.starts_with('"') && query.ends_with('"') && query.len() >= 2 {
            return Self::ExactPhrase(&query[1..query.len() - 1]);
        }
        Self::Default
    }
}

fn append_content(path: &Path, content: &[u8]) -> Result<u64> {
    let mut file = OpenOptions::new().append(true).read(true).open(path)?;
    file.lock_exclusive()?;
    let offset = file.metadata()?.len();
    file.write_all(content)?;
    file.flush()?;
    file.sync_data()?;
    file.unlock()?;
    Ok(offset)
}

fn append_metadata(path: &Path, record: &WalMetaRecord) -> Result<()> {
    let mut serialized = serde_json::to_vec(record)?;
    serialized.push(b'\n');

    let mut file = OpenOptions::new().append(true).open(path)?;
    file.lock_exclusive()?;
    file.write_all(&serialized)?;
    file.flush()?;
    file.sync_data()?;
    file.unlock()?;
    Ok(())
}

fn load_and_recover_metadata_records(path: &Path, content_size: u64) -> Result<Vec<WalMetaRecord>> {
    let mut file = OpenOptions::new().read(true).write(true).open(path)?;
    file.lock_exclusive()?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;

    let mut records = Vec::new();
    let mut recovered = Vec::new();
    let mut start = 0;
    let mut changed = false;
    for end in memchr_iter(b'\n', &bytes) {
        let line = &bytes[start..end];
        if let Some(record) = parse_metadata_line(line, content_size) {
            recovered.extend_from_slice(line);
            recovered.push(b'\n');
            records.push(record);
        } else if !line.is_empty() {
            changed = true;
        }
        start = end + 1;
    }

    if start < bytes.len() {
        changed = true;
    }

    if changed {
        file.set_len(0)?;
        file.seek(SeekFrom::Start(0))?;
        file.write_all(&recovered)?;
        file.flush()?;
        file.sync_data()?;
    }

    file.unlock()?;
    Ok(records)
}

fn load_manifest(
    path: &Path,
    _expect_non_empty: bool,
) -> Result<HashMap<String, SourceManifestEntry>> {
    let bytes = fs::read(path)?;
    if bytes.is_empty() {
        return Ok(HashMap::new());
    }

    let entries: Vec<SourceManifestEntry> = serde_json::from_slice(&bytes)?;
    Ok(entries
        .into_iter()
        .map(|entry| (entry.path.clone(), entry))
        .collect())
}

fn parse_metadata_line(line: &[u8], content_size: u64) -> Option<WalMetaRecord> {
    if line.is_empty() {
        return None;
    }

    let line = std::str::from_utf8(line).ok()?.trim();
    if line.is_empty() {
        return None;
    }

    let mut record: WalMetaRecord = serde_json::from_str(line).ok()?;
    record.content_type = ContentType::from_path(&record.source_path);
    let end = record.byte_offset.checked_add(record.byte_length)?;
    if end > content_size {
        return None;
    }

    Some(record)
}

fn scan_substring_results(
    wal_content_path: &Path,
    metadata: &[WalMetaRecord],
    needle: &[u8],
    source_layer: ResultSource,
) -> Result<Vec<ScoredResult>> {
    scan_with_content_slices(wal_content_path, metadata, |slice, entry, results| {
        scan_substring(slice, needle, entry, results, source_layer)
    })
}

fn scan_query_results(
    wal_content_path: &Path,
    metadata: &[WalMetaRecord],
    query: &str,
) -> Result<Vec<ScoredResult>> {
    let mut exact = scan_substring_results(
        wal_content_path,
        metadata,
        query.as_bytes(),
        ResultSource::RawScan,
    )?;
    if !exact.is_empty() {
        tracing::debug!(query = %query, matches = exact.len(), "default scan found exact substring matches");
        return Ok(exact);
    }

    let tokens: Vec<&str> = query
        .split_whitespace()
        .filter(|token| !token.is_empty())
        .collect();
    if tokens.len() <= 1 {
        tracing::debug!(query = %query, matches = 0, "default scan found no exact substring matches");
        return Ok(exact);
    }

    let mut by_line: HashMap<(u64, usize), ScoredResult> = HashMap::new();
    for token in tokens {
        for result in scan_substring_results(
            wal_content_path,
            metadata,
            token.as_bytes(),
            ResultSource::RawScan,
        )? {
            let key = (result.wal_entry_id, result.line_range.0);
            by_line
                .entry(key)
                .and_modify(|existing| existing.score += 1.0)
                .or_insert(result);
        }
    }

    exact = by_line.into_values().collect();
    exact.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.source_path.cmp(&right.source_path))
            .then_with(|| left.line_range.0.cmp(&right.line_range.0))
    });
    tracing::debug!(query = %query, matches = exact.len(), "default scan fell back to token-wise substring matching");
    Ok(exact)
}

fn scan_regex_results(
    wal_content_path: &Path,
    metadata: &[WalMetaRecord],
    pattern: &str,
    source_layer: ResultSource,
) -> Result<Vec<ScoredResult>> {
    let regex = Regex::new(pattern)?;
    scan_with_content_slices(wal_content_path, metadata, |slice, entry, results| {
        scan_regex(slice, &regex, entry, results, source_layer)
    })
}

fn scan_with_content_slices<F>(
    wal_content_path: &Path,
    metadata: &[WalMetaRecord],
    mut scanner: F,
) -> Result<Vec<ScoredResult>>
where
    F: FnMut(&[u8], &WalMetaRecord, &mut Vec<ScoredResult>),
{
    let content_file = File::open(wal_content_path)?;
    content_file.lock_shared()?;
    let mmap = unsafe { Mmap::map(&content_file)? };
    let mut results = Vec::new();

    for entry in metadata {
        let start = entry.byte_offset as usize;
        let end = start + entry.byte_length as usize;
        if end > mmap.len() {
            continue;
        }
        scanner(&mmap[start..end], entry, &mut results);
    }

    content_file.unlock()?;
    Ok(results)
}

fn scan_substring(
    slice: &[u8],
    needle: &[u8],
    entry: &WalMetaRecord,
    results: &mut Vec<ScoredResult>,
    source_layer: ResultSource,
) {
    if needle.is_empty() {
        return;
    }

    let finder = memmem::Finder::new(needle);
    let mut last_line_start: Option<usize> = None;

    for match_offset in finder.find_iter(slice) {
        push_line_result(
            slice,
            match_offset,
            entry,
            results,
            &mut last_line_start,
            source_layer,
        );
    }
}

fn scan_regex(
    slice: &[u8],
    regex: &Regex,
    entry: &WalMetaRecord,
    results: &mut Vec<ScoredResult>,
    source_layer: ResultSource,
) {
    let mut last_line_start: Option<usize> = None;
    for match_offset in regex.find_iter(slice).map(|m| m.start()) {
        push_line_result(
            slice,
            match_offset,
            entry,
            results,
            &mut last_line_start,
            source_layer,
        );
    }
}

fn tag_scan_results_as_fallback(results: &mut [ScoredResult]) {
    for result in results {
        if result.source_layer == ResultSource::RawScan {
            result.source_layer = ResultSource::ScanFallback;
        }
    }
}

fn collapse_fused_results_by_file(results: Vec<ScoredResult>) -> Vec<ScoredResult> {
    let mut best: BTreeMap<String, ScoredResult> = BTreeMap::new();
    for result in results {
        match best.get_mut(&result.source_path) {
            Some(existing) => {
                existing.score += result.score;
                if result.score > existing.score {
                    let cumulative = existing.score;
                    *existing = result;
                    existing.score = cumulative;
                }
            }
            None => {
                best.insert(result.source_path.clone(), result);
            }
        }
    }
    let mut collapsed: Vec<ScoredResult> = best.into_values().collect();
    collapsed.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.source_path.cmp(&right.source_path))
            .then_with(|| left.line_range.0.cmp(&right.line_range.0))
    });
    collapsed
}

fn push_line_result(
    slice: &[u8],
    match_offset: usize,
    entry: &WalMetaRecord,
    results: &mut Vec<ScoredResult>,
    last_line_start: &mut Option<usize>,
    source_layer: ResultSource,
) {
    let line_start = memrchr(b'\n', &slice[..match_offset])
        .map(|idx| idx + 1)
        .unwrap_or(0);
    if *last_line_start == Some(line_start) {
        return;
    }

    let line_end = slice[match_offset..]
        .iter()
        .position(|b| *b == b'\n')
        .map(|offset| match_offset + offset)
        .unwrap_or(slice.len());
    let preceding_newlines = bytecount_newlines(&slice[..line_start]);
    let line_number = entry.line_range_start + preceding_newlines;
    let snippet = String::from_utf8_lossy(&slice[line_start..line_end])
        .trim_end_matches('\r')
        .to_string();
    let (result_id, chunk_id) =
        scan_result_identity(slice, entry, match_offset, line_start, line_end);

    results.push(ScoredResult {
        result_id,
        source_path: entry.source_path.clone(),
        line_range: (line_number, line_number),
        chunk_id,
        snippet,
        score: 1.0,
        source_layer,
        wal_entry_id: entry.wal_entry_id,
    });
    *last_line_start = Some(line_start);
}

fn scan_result_identity(
    slice: &[u8],
    entry: &WalMetaRecord,
    match_offset: usize,
    line_start: usize,
    line_end: usize,
) -> (ResultId, u32) {
    let content = String::from_utf8_lossy(slice);
    if let Some(chunk) = chunk_entry(entry, &content).into_iter().find(|chunk| {
        let start = chunk.chunk.byte_start as usize;
        let end = chunk.chunk.byte_end as usize;
        match_offset >= start && match_offset < end
    }) {
        return (
            ResultId {
                wal_entry_id: entry.wal_entry_id,
                byte_start: chunk.chunk.byte_start,
                byte_end: chunk.chunk.byte_end,
            },
            chunk.chunk.chunk_id,
        );
    }

    (
        ResultId {
            wal_entry_id: entry.wal_entry_id,
            byte_start: line_start as u32,
            byte_end: line_end as u32,
        },
        0,
    )
}

#[cfg(feature = "semantic")]
fn semantic_scan_results_with_timing(
    wal_content_path: &Path,
    metadata: &[WalMetaRecord],
    semantic_query: &SemanticQuery,
    options: crate::semantic_scan::SemanticScanOptions,
) -> Result<(Vec<ScoredResult>, SemanticScanTiming)> {
    let (windows, timing) =
        semantic_scan_scored_windows(wal_content_path, metadata, semantic_query, options)?;
    Ok((
        windows.into_iter().map(|(_, scored)| scored).collect(),
        timing,
    ))
}

#[cfg(feature = "semantic")]
pub(crate) fn semantic_scan_scored_windows(
    wal_content_path: &Path,
    metadata: &[WalMetaRecord],
    semantic_query: &SemanticQuery,
    options: crate::semantic_scan::SemanticScanOptions,
) -> Result<(
    Vec<(crate::semantic_scan::ScoredWindow, ScoredResult)>,
    SemanticScanTiming,
)> {
    if metadata.is_empty() {
        return Ok((Vec::new(), SemanticScanTiming::default()));
    }

    let mut realized_query = semantic_query.clone();
    let patterns = realize_surfaces(&mut realized_query, &|term| static_df_frac(term));
    if patterns.is_empty() {
        return Ok((Vec::new(), SemanticScanTiming::default()));
    }
    let compile_started = Instant::now();
    let compiled = compile_scan_query_with_options(&patterns, options)?;
    let aho_compile = compile_started.elapsed();

    let content_file = File::open(wal_content_path)?;
    content_file.lock_shared()?;
    let mmap = unsafe { Mmap::map(&content_file)? };

    let mut entry_bytes = Vec::new();
    for entry in metadata {
        let start = entry.byte_offset as usize;
        let end = start + entry.byte_length as usize;
        if end > mmap.len() {
            continue;
        }
        entry_bytes.push((entry.wal_entry_id, &mmap[start..end], entry.content_type));
    }

    let scan_started = Instant::now();
    let (windows, _df_counts) = crate::semantic_scan::semantic_scan_with_options(
        &compiled,
        &entry_bytes,
        &realized_query,
        64,
        options,
    );
    let semantic_scan = scan_started.elapsed();

    let entry_map: HashMap<u64, &WalMetaRecord> = metadata
        .iter()
        .map(|entry| (entry.wal_entry_id, entry))
        .collect();
    let mut scored = Vec::new();
    for window in windows {
        let Some(entry) = entry_map.get(&window.wal_entry_id).copied() else {
            continue;
        };
        let start = entry.byte_offset as usize;
        let end = start + entry.byte_length as usize;
        if end > mmap.len() {
            continue;
        }
        let slice = &mmap[start..end];
        scored.push((
            window.clone(),
            semantic_window_to_scored(slice, entry, &window, window.score as f64),
        ));
    }

    content_file.unlock()?;

    scored.sort_by(|left, right| {
        right
            .1
            .score
            .partial_cmp(&left.1.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.1.source_path.cmp(&right.1.source_path))
            .then_with(|| left.1.line_range.0.cmp(&right.1.line_range.0))
    });
    Ok((
        scored,
        SemanticScanTiming {
            aho_compile,
            semantic_scan,
        },
    ))
}

#[cfg(feature = "semantic")]
fn semantic_window_to_scored(
    slice: &[u8],
    entry: &WalMetaRecord,
    window: &crate::semantic_scan::ScoredWindow,
    score: f64,
) -> ScoredResult {
    let window_start = (window.window_start as usize).min(slice.len());
    let window_end = (window.window_end as usize)
        .min(slice.len())
        .max(window_start);
    let line_start = memrchr(b'\n', &slice[..window_start])
        .map(|idx| idx + 1)
        .unwrap_or(0);
    let line_end = slice[window_end..]
        .iter()
        .position(|b| *b == b'\n')
        .map(|offset| window_end + offset)
        .unwrap_or(slice.len());
    let line_number = entry.line_range_start + bytecount_newlines(&slice[..line_start]);
    let line_end_number = line_number + bytecount_newlines(&slice[line_start..line_end]);
    let content = String::from_utf8_lossy(slice);
    let chunk = chunk_entry(entry, &content).into_iter().find(|chunk| {
        let start = chunk.chunk.byte_start as usize;
        let end = chunk.chunk.byte_end as usize;
        window_start < end && window_end > start
    });
    let (result_id, chunk_id, line_range) = match chunk {
        Some(chunk) => (
            ResultId {
                wal_entry_id: entry.wal_entry_id,
                byte_start: chunk.chunk.byte_start,
                byte_end: chunk.chunk.byte_end,
            },
            chunk.chunk.chunk_id,
            chunk.chunk.line_range,
        ),
        None => (
            ResultId {
                wal_entry_id: entry.wal_entry_id,
                byte_start: window.window_start,
                byte_end: window.window_end,
            },
            0,
            (line_number, line_end_number.max(line_number)),
        ),
    };

    ScoredResult {
        result_id,
        source_path: entry.source_path.clone(),
        line_range,
        chunk_id,
        snippet: String::from_utf8_lossy(&slice[line_start..line_end])
            .trim_end_matches('\r')
            .to_string(),
        score,
        source_layer: ResultSource::SemanticScan,
        wal_entry_id: entry.wal_entry_id,
    }
}

fn snippet_for_query(
    content: &str,
    line_range: (usize, usize),
    query: &str,
    source_layer: ResultSource,
) -> ((usize, usize), String) {
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return (line_range, String::new());
    }
    if matches!(
        source_layer,
        ResultSource::HotVector | ResultSource::DeltaFallback | ResultSource::QueryPromoted
    ) {
        if let Some((line_no, snippet)) = semantic_snippet_for_query(&lines, query) {
            return ((line_no, line_no), snippet);
        }
    }

    let start = line_range
        .0
        .saturating_sub(1)
        .min(lines.len().saturating_sub(1));
    let end = line_range.1.max(line_range.0).min(lines.len());
    let selected = &lines[start..end];
    if selected.len() > 3 {
        let snippet = selected
            .iter()
            .find(|line| !line.trim().is_empty())
            .map(|line| (*line).to_string())
            .or_else(|| lines.get(start).map(|line| (*line).to_string()))
            .unwrap_or_default();
        return ((start + 1, start + 1), snippet);
    }
    let snippet = selected.join("\n");
    if snippet.is_empty() {
        let fallback = lines
            .get(start)
            .map(|line| (*line).to_string())
            .unwrap_or_default();
        ((start + 1, start + 1), fallback)
    } else {
        ((start + 1, end.max(start + 1)), snippet)
    }
}

fn semantic_snippet_for_query(lines: &[&str], query: &str) -> Option<(usize, String)> {
    let mut terms = normalized_query_terms(query);
    let hint_terms = semantic_hint_terms(query);
    if terms.is_empty() {
        return lines
            .iter()
            .enumerate()
            .find(|(_, line)| !line.trim().is_empty())
            .map(|(index, line)| (index + 1, (*line).to_string()));
    }
    terms.extend(hint_terms.clone());

    let query_lower = query.to_lowercase();
    let error_query = query_lower.contains("error")
        || query_lower.contains("fail")
        || query_lower.contains("exception");
    let vector_query = query_lower.contains("embed") || query_lower.contains("vector");

    let mut best: Option<(usize, usize, String)> = None;
    for (index, line) in lines.iter().enumerate() {
        let lower = line.to_lowercase();
        let mut score = terms
            .iter()
            .filter(|term| lower.contains(term.as_str()))
            .count();
        if error_query
            && ["result", "error", "err", "map_err", "fallback", "panic"]
                .iter()
                .any(|term| lower.contains(term))
        {
            score += 2;
        }
        if vector_query
            && [
                "embed",
                "embedding",
                "vector",
                "vectors",
                "dimension",
                "tokenizer",
            ]
            .iter()
            .any(|term| lower.contains(term))
        {
            score += 2;
        }
        if lower.contains('"') && !lower.contains("error") && !lower.contains("vector") {
            score = score.saturating_sub(2);
        }
        if lower.trim_start().starts_with("[")
            || lower.contains("[\"")
            || lower.contains("contains(\"")
            || lower.contains("contains('\"")
        {
            score = 0;
        }
        if score == 0 {
            continue;
        }
        let candidate = (score, index + 1, (*line).to_string());
        if best.as_ref().is_none_or(|current| candidate.0 > current.0) {
            best = Some(candidate);
        }
    }
    best.map(|(_, line_no, snippet)| (line_no, snippet))
        .or_else(|| {
            lines
                .iter()
                .enumerate()
                .find(|(_, line)| !line.trim().is_empty())
                .map(|(index, line)| (index + 1, (*line).to_string()))
        })
}

fn normalized_query_terms(query: &str) -> Vec<String> {
    query
        .split_whitespace()
        .map(|term| {
            term.trim_matches(|ch: char| !ch.is_alphanumeric())
                .to_lowercase()
        })
        .filter(|term| !term.is_empty())
        .flat_map(|term| {
            if term.ends_with('s') && term.len() > 3 {
                vec![term.clone(), term.trim_end_matches('s').to_string()]
            } else {
                vec![term]
            }
        })
        .collect()
}

fn semantic_hint_terms(query: &str) -> Vec<String> {
    let lower = query.to_lowercase();
    let mut hints = Vec::new();
    if lower.contains("error") || lower.contains("fail") || lower.contains("exception") {
        hints.extend(
            [
                "error",
                "errors",
                "err",
                "result",
                "sieveerror",
                "map_err",
                "fallback",
                "failed",
                "failure",
                "panic",
            ]
            .into_iter()
            .map(str::to_string),
        );
    }
    if lower.contains("handle") || lower.contains("handling") {
        hints.extend(
            ["handle", "handling", "result", "map_err", "match"]
                .into_iter()
                .map(str::to_string),
        );
    }
    if lower.contains("embed") || lower.contains("vector") {
        hints.extend(
            [
                "embed",
                "embedding",
                "vector",
                "vectors",
                "tokenizer",
                "session",
                "dimension",
            ]
            .into_iter()
            .map(str::to_string),
        );
    }
    hints
}

fn bytecount_newlines(bytes: &[u8]) -> usize {
    memchr_iter(b'\n', bytes).count()
}

fn filter_lexical_matches(
    matches: Vec<LexicalMatch>,
    active_ids: &HashSet<u64>,
) -> Vec<LexicalMatch> {
    matches
        .into_iter()
        .filter(|entry| active_ids.contains(&entry.wal_entry_id))
        .collect()
}

#[cfg(feature = "semantic")]
fn filter_lexical_matches_bitmap(
    matches: Vec<LexicalMatch>,
    active_ids: &roaring::RoaringTreemap,
) -> Vec<LexicalMatch> {
    matches
        .into_iter()
        .filter(|entry| active_ids.contains(entry.wal_entry_id))
        .collect()
}

fn lexical_to_scored(matches: Vec<LexicalMatch>) -> Vec<ScoredResult> {
    lexical_to_scored_with_source(matches, ResultSource::LexicalBm25)
}

fn lexical_to_scored_with_source(
    matches: Vec<LexicalMatch>,
    source_layer: ResultSource,
) -> Vec<ScoredResult> {
    matches
        .into_iter()
        .map(|entry| ScoredResult {
            result_id: ResultId {
                wal_entry_id: entry.wal_entry_id,
                byte_start: entry.byte_range.0,
                byte_end: entry.byte_range.1,
            },
            source_path: entry.source_path,
            line_range: entry.line_range,
            chunk_id: entry.chunk_id,
            snippet: entry.snippet,
            score: entry.bm25_score,
            source_layer,
            wal_entry_id: entry.wal_entry_id,
        })
        .collect()
}

#[cfg(feature = "semantic")]
fn vector_matches_to_scored(matches: Vec<VectorMatch>) -> Vec<ScoredResult> {
    vector_matches_to_scored_with_source(matches, ResultSource::HotVector)
}

#[cfg(feature = "semantic")]
fn vector_matches_to_scored_with_source(
    matches: Vec<VectorMatch>,
    source: ResultSource,
) -> Vec<ScoredResult> {
    matches
        .into_iter()
        .map(|entry| ScoredResult {
            result_id: ResultId {
                wal_entry_id: entry.wal_entry_id,
                byte_start: entry.byte_range.0,
                byte_end: entry.byte_range.1,
            },
            source_path: entry.source_path,
            line_range: entry.line_range,
            chunk_id: entry.chunk_id,
            snippet: String::new(),
            score: entry.score,
            source_layer: source,
            wal_entry_id: entry.wal_entry_id,
        })
        .collect()
}

#[cfg(feature = "semantic")]
fn filter_vector_matches(matches: Vec<VectorMatch>, active_ids: &HashSet<u64>) -> Vec<VectorMatch> {
    matches
        .into_iter()
        .filter(|entry| active_ids.contains(&entry.wal_entry_id))
        .collect()
}

#[cfg(feature = "semantic")]
pub fn collapse_vector_matches_by_file(
    matches: Vec<VectorMatch>,
    top_k_files: usize,
) -> Vec<VectorMatch> {
    let mut best: BTreeMap<String, (usize, VectorMatch)> = BTreeMap::new();
    for (rank, entry) in matches.into_iter().enumerate() {
        match best.get_mut(&entry.source_path) {
            Some((existing_rank, existing)) => {
                if entry.score > existing.score
                    || (entry.score == existing.score && rank < *existing_rank)
                {
                    *existing_rank = rank;
                    *existing = entry;
                }
            }
            None => {
                best.insert(entry.source_path.clone(), (rank, entry));
            }
        }
    }
    let mut collapsed: Vec<(usize, VectorMatch)> = best.into_values().collect();
    collapsed.sort_by(|(left_rank, left), (right_rank, right)| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left_rank.cmp(right_rank))
            .then_with(|| left.source_path.cmp(&right.source_path))
            .then_with(|| left.line_range.0.cmp(&right.line_range.0))
    });
    let mut collapsed: Vec<VectorMatch> = collapsed.into_iter().map(|(_, entry)| entry).collect();
    if collapsed.len() > top_k_files {
        collapsed.truncate(top_k_files);
    }
    collapsed
}

#[cfg(feature = "semantic")]
fn collapse_scored_results_by_file(
    results: Vec<ScoredResult>,
    top_k_files: usize,
) -> Vec<ScoredResult> {
    let mut best: BTreeMap<String, (usize, ScoredResult)> = BTreeMap::new();
    for (rank, entry) in results.into_iter().enumerate() {
        match best.get_mut(&entry.source_path) {
            Some((existing_rank, existing)) => {
                if entry.score > existing.score
                    || (entry.score == existing.score && rank < *existing_rank)
                {
                    *existing_rank = rank;
                    *existing = entry;
                }
            }
            None => {
                best.insert(entry.source_path.clone(), (rank, entry));
            }
        }
    }
    let mut collapsed: Vec<(usize, ScoredResult)> = best.into_values().collect();
    collapsed.sort_by(|(left_rank, left), (right_rank, right)| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left_rank.cmp(right_rank))
            .then_with(|| left.source_path.cmp(&right.source_path))
            .then_with(|| left.line_range.0.cmp(&right.line_range.0))
    });
    let mut collapsed: Vec<ScoredResult> = collapsed.into_iter().map(|(_, entry)| entry).collect();
    if collapsed.len() > top_k_files {
        collapsed.truncate(top_k_files);
    }
    collapsed
}

#[cfg(feature = "semantic")]
pub fn apply_dense_recency_bonus(
    mut results: Vec<ScoredResult>,
    wal_order: &[u64],
) -> Vec<ScoredResult> {
    let recent_count = wal_order.len().clamp(1, 8);
    let recent: Vec<u64> = wal_order[wal_order.len().saturating_sub(recent_count)..]
        .iter()
        .rev()
        .copied()
        .collect();
    for result in &mut results {
        if let Some(position) = recent.iter().position(|wal| *wal == result.wal_entry_id) {
            let recency_rank = recent_count.saturating_sub(position) as f64 / recent_count as f64;
            result.score += 0.02 * recency_rank;
        }
    }
    results.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.source_path.cmp(&right.source_path))
            .then_with(|| left.line_range.0.cmp(&right.line_range.0))
    });
    results
}

#[cfg(feature = "semantic")]
fn score_delta_vectors(
    query_vec: &[f32],
    entries: &[SourceChunk],
    vectors: &[Vec<f32>],
    top_k: usize,
) -> Vec<ScoredResult> {
    let mut scored: Vec<ScoredResult> = entries
        .iter()
        .zip(vectors.iter())
        .map(|(entry, vector)| ScoredResult {
            result_id: ResultId {
                wal_entry_id: entry.chunk.wal_entry_id,
                byte_start: entry.chunk.byte_start,
                byte_end: entry.chunk.byte_end,
            },
            source_path: entry.source_path.clone(),
            line_range: entry.chunk.line_range,
            chunk_id: entry.chunk.chunk_id,
            snippet: String::new(),
            score: dot_product(query_vec, vector),
            source_layer: ResultSource::DeltaFallback,
            wal_entry_id: entry.chunk.wal_entry_id,
        })
        .collect();
    scored.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if scored.len() > top_k {
        scored.truncate(top_k);
    }
    scored
}

#[cfg(feature = "semantic")]
fn select_query_promoted_chunks(
    scan_results: &[ScoredResult],
    delta_chunks: &[SourceChunk],
    embedded_set: &roaring::RoaringTreemap,
    recent_wal_entries: usize,
    config: &QueryPromotedDense,
) -> Vec<SourceChunk> {
    if config.max_promoted_chunks == 0 {
        return Vec::new();
    }
    let mut scan_ranks: HashMap<u64, usize> = HashMap::new();
    for (rank, result) in scan_results.iter().enumerate() {
        if embedded_set.contains(result.wal_entry_id) {
            continue;
        }
        scan_ranks
            .entry(result.wal_entry_id)
            .and_modify(|existing| *existing = (*existing).min(rank))
            .or_insert(rank);
    }
    let mut recent_ids: Vec<u64> = delta_chunks
        .iter()
        .map(|chunk| chunk.chunk.wal_entry_id)
        .filter(|wal_entry_id| !embedded_set.contains(*wal_entry_id))
        .collect();
    recent_ids.sort_unstable();
    recent_ids.dedup();
    recent_ids.reverse();
    let recent_rank: HashMap<u64, usize> = recent_ids
        .into_iter()
        .take(recent_wal_entries.max(1))
        .enumerate()
        .map(|(index, wal_entry_id)| (wal_entry_id, index))
        .collect();

    let mut selected: Vec<SourceChunk> = delta_chunks
        .iter()
        .filter(|chunk| !embedded_set.contains(chunk.chunk.wal_entry_id))
        .filter(|chunk| {
            scan_ranks.contains_key(&chunk.chunk.wal_entry_id)
                || recent_rank.contains_key(&chunk.chunk.wal_entry_id)
        })
        .cloned()
        .collect();
    selected.sort_by(|left, right| {
        let left_scan = scan_ranks
            .get(&left.chunk.wal_entry_id)
            .copied()
            .unwrap_or(usize::MAX);
        let right_scan = scan_ranks
            .get(&right.chunk.wal_entry_id)
            .copied()
            .unwrap_or(usize::MAX);
        let left_recent = recent_rank
            .get(&left.chunk.wal_entry_id)
            .copied()
            .unwrap_or(usize::MAX);
        let right_recent = recent_rank
            .get(&right.chunk.wal_entry_id)
            .copied()
            .unwrap_or(usize::MAX);
        left_scan
            .cmp(&right_scan)
            .then_with(|| left_recent.cmp(&right_recent))
            .then_with(|| right.chunk.wal_entry_id.cmp(&left.chunk.wal_entry_id))
            .then_with(|| left.chunk.chunk_id.cmp(&right.chunk.chunk_id))
    });
    selected.truncate(config.max_promoted_chunks);
    selected
}

#[cfg(feature = "semantic")]
fn query_promoted_dense_results_with<E>(
    query_vec: &[f32],
    promoted_chunks: &[SourceChunk],
    top_k: usize,
    config: &QueryPromotedDense,
    mut embed_one: E,
) -> Result<Vec<ScoredResult>>
where
    E: FnMut(&str) -> Result<Vec<f32>>,
{
    if promoted_chunks.is_empty() || config.max_promoted_chunks == 0 || top_k == 0 {
        return Ok(Vec::new());
    }
    let started = Instant::now();
    let mut vectors = Vec::new();
    let mut embedded_chunks = Vec::new();
    for chunk in promoted_chunks.iter().take(config.max_promoted_chunks) {
        if started.elapsed().as_millis() as u64 >= config.max_promoted_ms {
            break;
        }
        let vector = embed_one(chunk.chunk.text.as_str())?;
        if started.elapsed().as_millis() as u64 > config.max_promoted_ms {
            break;
        }
        embedded_chunks.push(chunk.clone());
        vectors.push(vector);
    }
    if embedded_chunks.is_empty() {
        return Ok(Vec::new());
    }
    let mut scored = score_delta_vectors(query_vec, &embedded_chunks, &vectors, top_k);
    for result in &mut scored {
        result.source_layer = ResultSource::QueryPromoted;
    }
    Ok(scored)
}

#[cfg(feature = "semantic")]
fn dot_product(left: &[f32], right: &[f32]) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(left, right)| (*left as f64) * (*right as f64))
        .sum()
}

fn ensure_dir(path: &Path) -> Result<bool> {
    if path.exists() {
        return Ok(false);
    }
    fs::create_dir_all(path)?;
    Ok(true)
}

#[cfg(feature = "semantic")]
fn should_use_semantic(query: &str) -> bool {
    let had_code_punctuation = query
        .chars()
        .any(|ch| matches!(ch, ':' | '.' | '_' | '/' | '-'));
    let normalized = query
        .chars()
        .map(|ch| {
            if ch.is_alphanumeric() || ch.is_whitespace() {
                ch
            } else {
                ' '
            }
        })
        .collect::<String>();
    let tokens: Vec<&str> = normalized.split_whitespace().collect();
    (tokens.len() >= 2 && tokens.iter().any(|token| token.len() > 4))
        || (had_code_punctuation
            && tokens.len() >= 2
            && tokens.iter().any(|token| token.len() >= 3))
}

fn create_file_if_missing(path: &Path) -> Result<bool> {
    if path.exists() {
        return Ok(false);
    }
    File::create(path)?;
    Ok(true)
}

fn sync_dir(path: &Path) -> Result<()> {
    File::open(path)?.sync_all()?;
    Ok(())
}

fn hash_content(content: &[u8]) -> String {
    let mut hasher = Hasher::new();
    hasher.update(content);
    hasher.finalize().to_hex().to_string()
}

#[cfg_attr(not(feature = "semantic"), allow(dead_code))]
#[derive(Debug, Clone)]
struct SourceChunk {
    source_path: String,
    chunk: Chunk,
}

fn chunk_entry(entry: &WalMetaRecord, content: &str) -> Vec<SourceChunk> {
    SlidingChunker::default()
        .chunk_entry(entry.wal_entry_id, content)
        .into_iter()
        .map(|chunk| SourceChunk {
            source_path: entry.source_path.clone(),
            chunk,
        })
        .collect()
}

fn unix_timestamp_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or_default()
}

fn default_line_start() -> usize {
    1
}

fn default_line_end() -> usize {
    1
}

pub fn blake3_hex(bytes: &[u8]) -> String {
    hash_content(bytes)
}

#[cfg(feature = "semantic")]
pub fn default_sieve_data_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".sieve")
}

#[cfg(all(test, feature = "semantic"))]
mod semantic_unit_tests {
    use std::collections::HashSet;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    use roaring::RoaringTreemap;

    use super::{
        filter_vector_matches, query_promoted_dense_results_with, select_query_promoted_chunks,
        QueryPromotedDense, SourceChunk,
    };
    use crate::chunk::Chunk;
    use crate::fusion::{ResultId, ResultSource, ScoredResult};
    use crate::vectors::{HotVectorStore, VectorMatch, VectorMeta};
    use tempfile::tempdir;

    #[test]
    fn test_filter_vector_matches_excludes_inactive_wal_entries() {
        let matches = vec![
            VectorMatch {
                wal_entry_id: 1,
                source_path: "active.rs".into(),
                byte_range: (0, 8),
                line_range: (1, 1),
                chunk_id: 0,
                score: 0.9,
            },
            VectorMatch {
                wal_entry_id: 2,
                source_path: "stale.rs".into(),
                byte_range: (8, 16),
                line_range: (2, 2),
                chunk_id: 0,
                score: 0.8,
            },
        ];
        let active_ids = HashSet::from([1_u64]);
        let filtered = filter_vector_matches(matches, &active_ids);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].wal_entry_id, 1);
        assert_eq!(filtered[0].source_path, "active.rs");
    }

    #[test]
    fn test_should_use_semantic_accepts_code_queries_with_punctuation() {
        assert!(super::should_use_semantic("std::error::Error"));
        assert!(super::should_use_semantic("foo/bar_baz.rs"));
    }

    fn sample_source_chunk(
        wal_entry_id: u64,
        source_path: &str,
        chunk_id: u32,
        text: &str,
    ) -> SourceChunk {
        SourceChunk {
            source_path: source_path.to_string(),
            chunk: Chunk {
                wal_entry_id,
                chunk_id,
                byte_start: chunk_id * 32,
                byte_end: chunk_id * 32 + text.len() as u32,
                line_range: (1, 1),
                text: text.to_string(),
            },
        }
    }

    fn sample_scan_result(wal_entry_id: u64, source_path: &str, rank_score: f64) -> ScoredResult {
        ScoredResult {
            result_id: ResultId {
                wal_entry_id,
                byte_start: 0,
                byte_end: 32,
            },
            source_path: source_path.to_string(),
            line_range: (1, 1),
            chunk_id: 0,
            snippet: source_path.to_string(),
            score: rank_score,
            source_layer: ResultSource::RawScan,
            wal_entry_id,
        }
    }

    #[test]
    fn test_query_promoted_embeds_scan_hits() {
        let chunks = vec![
            sample_source_chunk(1, "scan-hit.rs", 0, "scan-hit body"),
            sample_source_chunk(2, "recent.rs", 0, "recent body"),
        ];
        let scan = vec![sample_scan_result(1, "scan-hit.rs", 2.0)];
        let embedded = RoaringTreemap::new();
        let selected = select_query_promoted_chunks(
            &scan,
            &chunks,
            &embedded,
            1,
            &QueryPromotedDense {
                max_promoted_chunks: 1,
                max_promoted_ms: 300,
            },
        );
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].source_path, "scan-hit.rs");

        let query_vec = vec![1.0, 0.0];
        let results = query_promoted_dense_results_with(
            &query_vec,
            &selected,
            5,
            &QueryPromotedDense {
                max_promoted_chunks: 1,
                max_promoted_ms: 300,
            },
            |_| Ok(vec![1.0, 0.0]),
        )
        .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].source_path, "scan-hit.rs");
        assert_eq!(results[0].source_layer, ResultSource::QueryPromoted);
    }

    #[test]
    fn test_query_promoted_includes_recent_files() {
        let chunks = vec![
            sample_source_chunk(10, "older.rs", 0, "older body"),
            sample_source_chunk(11, "fresh.rs", 0, "fresh body"),
        ];
        let embedded = RoaringTreemap::new();
        let selected = select_query_promoted_chunks(
            &[],
            &chunks,
            &embedded,
            2,
            &QueryPromotedDense {
                max_promoted_chunks: 1,
                max_promoted_ms: 300,
            },
        );
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].chunk.wal_entry_id, 11);
    }

    #[test]
    fn test_query_promoted_respects_budget() {
        let chunks = vec![
            sample_source_chunk(1, "one.rs", 0, "one"),
            sample_source_chunk(2, "two.rs", 0, "two"),
            sample_source_chunk(3, "three.rs", 0, "three"),
        ];
        let selected = select_query_promoted_chunks(
            &[],
            &chunks,
            &RoaringTreemap::new(),
            3,
            &QueryPromotedDense {
                max_promoted_chunks: 2,
                max_promoted_ms: 300,
            },
        );
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn test_query_promoted_respects_time_budget() {
        let chunks = vec![
            sample_source_chunk(1, "one.rs", 0, "one"),
            sample_source_chunk(2, "two.rs", 0, "two"),
            sample_source_chunk(3, "three.rs", 0, "three"),
        ];
        let counter = AtomicUsize::new(0);
        let started = Instant::now();
        let results = query_promoted_dense_results_with(
            &[1.0, 0.0],
            &chunks,
            5,
            &QueryPromotedDense {
                max_promoted_chunks: 3,
                max_promoted_ms: 100,
            },
            |_| {
                std::thread::sleep(Duration::from_millis(60));
                counter.fetch_add(1, Ordering::SeqCst);
                Ok(vec![1.0, 0.0])
            },
        )
        .unwrap();
        assert!(started.elapsed() < Duration::from_millis(180));
        assert!(counter.load(Ordering::SeqCst) <= 2);
        assert!(results.len() <= 2);
    }

    #[test]
    fn test_query_promoted_skips_when_all_embedded() {
        let chunks = vec![sample_source_chunk(1, "done.rs", 0, "done")];
        let embedded = RoaringTreemap::from([1_u64]);
        let selected = select_query_promoted_chunks(
            &[],
            &chunks,
            &embedded,
            1,
            &QueryPromotedDense {
                max_promoted_chunks: 10,
                max_promoted_ms: 300,
            },
        );
        assert!(selected.is_empty());
    }

    #[test]
    fn test_query_promoted_results_not_persisted() {
        let dir = tempdir().unwrap();
        let mut store = HotVectorStore::open_or_create(dir.path(), 2).unwrap();
        store
            .append(
                &[vec![1.0, 0.0]],
                &[VectorMeta {
                    wal_entry_id: 1,
                    chunk_id: 0,
                    source_path: "persisted.rs".into(),
                    byte_range: (0, 8),
                    line_range: (1, 1),
                }],
            )
            .unwrap();
        let before = store.len();
        let chunks = vec![sample_source_chunk(2, "ephemeral.rs", 0, "ephemeral")];
        let _ = query_promoted_dense_results_with(
            &[1.0, 0.0],
            &chunks,
            5,
            &QueryPromotedDense {
                max_promoted_chunks: 1,
                max_promoted_ms: 300,
            },
            |_| Ok(vec![1.0, 0.0]),
        )
        .unwrap();
        let reopened = HotVectorStore::open_or_create(dir.path(), 2).unwrap();
        assert_eq!(before, reopened.len());
    }

    #[test]
    fn test_query_promoted_tagged_correctly() {
        let chunks = vec![sample_source_chunk(1, "promoted.rs", 0, "body")];
        let results = query_promoted_dense_results_with(
            &[1.0, 0.0],
            &chunks,
            5,
            &QueryPromotedDense {
                max_promoted_chunks: 1,
                max_promoted_ms: 300,
            },
            |_| Ok(vec![1.0, 0.0]),
        )
        .unwrap();
        assert_eq!(results[0].source_layer, ResultSource::QueryPromoted);
    }
}
