pub mod fusion;
pub mod lexical;

use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use blake3::Hasher;
use fs2::FileExt;
use memchr::{memchr_iter, memmem, memrchr};
use memmap2::Mmap;
use rayon::join;
use regex::bytes::Regex;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::fusion::{rrf_fuse, ResultSource, ScoredResult};
use crate::lexical::{search_lexical_with_fallback, LexicalMatch};

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
}

#[derive(Debug, Clone, Default)]
pub struct SearchOptions {
    pub top_k: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    pub source_path: String,
    pub line_number: usize,
    pub line_range: (usize, usize),
    pub snippet: String,
    pub score: f64,
    pub source_layer: ResultSource,
    pub wal_entry_id: u64,
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceManifestEntry {
    pub path: String,
    pub mtime_ms: u64,
    pub size: u64,
    pub blake3_hash: String,
    pub wal_entry_id: u64,
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
        })
    }

    pub fn add_text(
        &self,
        source_path: impl Into<String>,
        content: impl Into<String>,
    ) -> Result<u64> {
        let source_path = source_path.into();
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
                rrf_fuse(vec![scan, lexical_to_scored(lexical)], 5.0)
            }
            QueryKind::Default => {
                let (scan, lexical) = join(
                    || {
                        scan_substring_results(
                            &self.wal_content_path,
                            &active_metadata,
                            query.as_bytes(),
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
                rrf_fuse(vec![scan, lexical_to_scored(lexical)], 5.0)
            }
        };

        Ok(results
            .into_iter()
            .take(top_k)
            .map(|result| SearchResult {
                line_number: result.line_range.0,
                line_range: result.line_range,
                source_path: result.source_path,
                snippet: result.snippet,
                score: result.score,
                source_layer: result.source_layer,
                wal_entry_id: result.wal_entry_id,
            })
            .collect())
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

    pub fn active_wal_entry_ids(&self) -> Result<HashSet<u64>> {
        let manifest = self.manifest.read().map_err(|_| SieveError::LockPoisoned)?;
        Ok(manifest.values().map(|entry| entry.wal_entry_id).collect())
    }
}

enum QueryKind<'a> {
    Regex(String),
    ExactPhrase(&'a str),
    Default,
}

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

    let record: WalMetaRecord = serde_json::from_str(line).ok()?;
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

    results.push(ScoredResult {
        source_path: entry.source_path.clone(),
        line_range: (line_number, line_number),
        snippet,
        score: 1.0,
        source_layer,
        wal_entry_id: entry.wal_entry_id,
    });
    *last_line_start = Some(line_start);
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

fn lexical_to_scored(matches: Vec<LexicalMatch>) -> Vec<ScoredResult> {
    matches
        .into_iter()
        .map(|entry| ScoredResult {
            source_path: entry.source_path,
            line_range: entry.line_range,
            snippet: entry.snippet,
            score: entry.bm25_score,
            source_layer: ResultSource::LexicalBm25,
            wal_entry_id: entry.wal_entry_id,
        })
        .collect()
}

fn ensure_dir(path: &Path) -> Result<bool> {
    if path.exists() {
        return Ok(false);
    }
    fs::create_dir_all(path)?;
    Ok(true)
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
