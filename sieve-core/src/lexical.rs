use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use roaring::RoaringTreemap;
use tantivy::collector::TopDocs;
use tantivy::directory::MmapDirectory;
use tantivy::query::QueryParser;
use tantivy::schema::{Field, Schema, SchemaBuilder, Value, STORED, STRING, TEXT};
use tantivy::{doc, Index as TantivyIndex, TantivyDocument};
use tracing::debug;

use crate::{Index, Result, SieveError, WalMetaRecord};

const INDEXED_ENTRIES_FILE: &str = "indexed_entries.bin";
const SHARD_PREFIX: &str = "seg_";

#[derive(Debug, Clone, PartialEq)]
pub struct LexicalMatch {
    pub source_path: String,
    pub line_range: (usize, usize),
    pub snippet: String,
    pub bm25_score: f64,
    pub wal_entry_id: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LexicalSearchOutcome {
    pub matches: Vec<LexicalMatch>,
    pub skipped_due_to_parse_failure: bool,
}

#[derive(Clone, Copy)]
struct LexicalFields {
    source_path: Field,
    content: Field,
    wal_entry_id: Field,
    byte_offset: Field,
    line_range_start: Field,
    line_range_end: Field,
}

pub fn build_pending_shards(index: &Index) -> Result<usize> {
    let segments_dir = index.root().join("segments");
    fs::create_dir_all(&segments_dir)?;

    let indexed_entries = load_indexed_entries(&segments_dir)?;
    let metadata = index.metadata_snapshot()?;
    let pending: Vec<WalMetaRecord> = metadata
        .into_iter()
        .filter(|entry| !indexed_entries.contains(entry.wal_entry_id))
        .collect();

    if pending.is_empty() {
        return Ok(0);
    }

    let content_file = File::open(index.wal_content_path())?;
    let mmap = unsafe { Mmap::map(&content_file)? };

    let shard_id = next_shard_id(&segments_dir)?;
    let temp_dir = segments_dir.join(format!(".tmp_seg_{shard_id}"));
    if temp_dir.exists() {
        fs::remove_dir_all(&temp_dir)?;
    }
    fs::create_dir_all(&temp_dir)?;

    let schema = lexical_schema();
    let fields = schema_fields(&schema).ok_or_else(|| {
        SieveError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "missing lexical schema fields",
        ))
    })?;
    let tantivy_index =
        TantivyIndex::open_or_create(MmapDirectory::open(&temp_dir)?, schema.clone())?;
    let mut writer = tantivy_index.writer(50_000_000)?;

    for entry in &pending {
        let start = entry.byte_offset as usize;
        let end = start + entry.byte_length as usize;
        if end > mmap.len() {
            continue;
        }
        let content = String::from_utf8_lossy(&mmap[start..end]).into_owned();
        writer.add_document(doc!(
            fields.source_path => entry.source_path.clone(),
            fields.content => content,
            fields.wal_entry_id => entry.wal_entry_id,
            fields.byte_offset => entry.byte_offset,
            fields.line_range_start => entry.line_range_start as u64,
            fields.line_range_end => entry.line_range_end as u64,
        ))?;
    }

    writer.commit()?;

    let final_dir = segments_dir.join(format!("{SHARD_PREFIX}{shard_id:04}"));
    fs::rename(&temp_dir, &final_dir)?;

    let mut new_bitmap = indexed_entries;
    for entry in pending {
        new_bitmap.insert(entry.wal_entry_id);
    }
    persist_indexed_entries(&segments_dir, &new_bitmap)?;

    let _ = merge_small_shards(&segments_dir);

    Ok(1)
}

pub fn search_lexical(shards_dir: &Path, query: &str, top_k: usize) -> Result<Vec<LexicalMatch>> {
    Ok(search_lexical_with_fallback(shards_dir, query, top_k)?.matches)
}

pub(crate) fn search_lexical_with_fallback(
    shards_dir: &Path,
    query: &str,
    top_k: usize,
) -> Result<LexicalSearchOutcome> {
    if !shards_dir.exists() {
        return Ok(LexicalSearchOutcome {
            matches: Vec::new(),
            skipped_due_to_parse_failure: false,
        });
    }

    let shard_dirs = shard_directories(shards_dir)?;
    if shard_dirs.is_empty() {
        return Ok(LexicalSearchOutcome {
            matches: Vec::new(),
            skipped_due_to_parse_failure: false,
        });
    }

    let mut results = Vec::new();

    for shard_dir in shard_dirs {
        let index = TantivyIndex::open_in_dir(&shard_dir)?;
        let schema = index.schema();
        let fields = schema_fields(&schema).ok_or_else(|| {
            SieveError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "missing lexical schema fields",
            ))
        })?;
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let parser = QueryParser::for_index(&index, vec![fields.content]);
        let Some(query_obj) = parse_query_with_phrase_retry(&index, &parser, query) else {
            return Ok(LexicalSearchOutcome {
                matches: Vec::new(),
                skipped_due_to_parse_failure: true,
            });
        };
        let top_docs = searcher.search(&query_obj, &TopDocs::with_limit(top_k))?;

        for (score, doc_address) in top_docs {
            let retrieved: TantivyDocument = searcher.doc(doc_address)?;
            let source_path = retrieved
                .get_first(fields.source_path)
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            let content = retrieved
                .get_first(fields.content)
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            let wal_entry_id = retrieved
                .get_first(fields.wal_entry_id)
                .and_then(|v| v.as_u64())
                .unwrap_or_default();
            let default_line_start = retrieved
                .get_first(fields.line_range_start)
                .and_then(|v| v.as_u64())
                .unwrap_or(1) as usize;
            let default_line_end = retrieved
                .get_first(fields.line_range_end)
                .and_then(|v| v.as_u64())
                .unwrap_or(default_line_start as u64) as usize;
            let (line_range, snippet) = extract_snippet_and_line_range(
                &content,
                query,
                (default_line_start, default_line_end),
            );
            results.push(LexicalMatch {
                source_path,
                line_range,
                snippet,
                bm25_score: score as f64,
                wal_entry_id,
            });
        }
    }

    results.sort_by(|a, b| {
        b.bm25_score
            .partial_cmp(&a.bm25_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.source_path.cmp(&b.source_path))
    });
    if results.len() > top_k {
        results.truncate(top_k);
    }
    Ok(LexicalSearchOutcome {
        matches: results,
        skipped_due_to_parse_failure: false,
    })
}

fn parse_query_with_phrase_retry(
    index: &TantivyIndex,
    parser: &QueryParser,
    query: &str,
) -> Option<Box<dyn tantivy::query::Query>> {
    match parser.parse_query(query) {
        Ok(parsed) => Some(parsed),
        Err(default_err) => {
            debug!(query = %query, error = %default_err, "tantivy default query parse failed; retrying as exact phrase");
            if !phrase_retry_preserves_query_shape(index, query) {
                debug!(query = %query, "tantivy phrase retry would lossy-normalize the query; skipping lexical layer");
                return None;
            }
            let phrase_query = exact_phrase_query(query);
            match parser.parse_query(&phrase_query) {
                Ok(parsed) => Some(parsed),
                Err(phrase_err) => {
                    debug!(query = %query, phrase_query = %phrase_query, error = %phrase_err, "tantivy exact phrase retry failed; skipping lexical layer");
                    None
                }
            }
        }
    }
}

fn phrase_retry_preserves_query_shape(index: &TantivyIndex, query: &str) -> bool {
    let Some(mut tokenizer) = index.tokenizers().get("default") else {
        return false;
    };
    let mut stream = tokenizer.token_stream(query);
    let mut tokens = Vec::new();
    while stream.advance() {
        tokens.push(stream.token().text.clone());
    }
    if tokens.is_empty() {
        return false;
    }
    let normalized_query = query.trim().to_lowercase();
    let tokenized_query = tokens.join(" ");
    normalized_query == tokenized_query
}

fn exact_phrase_query(query: &str) -> String {
    let escaped = query.replace('\\', r"\\").replace('"', r#"\""#);
    format!("\"{escaped}\"")
}

pub fn load_indexed_entries(shards_dir: &Path) -> Result<RoaringTreemap> {
    let path = shards_dir.join(INDEXED_ENTRIES_FILE);
    if !path.exists() {
        return Ok(RoaringTreemap::new());
    }
    let mut file = File::open(path)?;
    let bitmap = RoaringTreemap::deserialize_from(&mut file)
        .map_err(|err| SieveError::Io(std::io::Error::other(err.to_string())))?;
    Ok(bitmap)
}

pub fn merge_small_shards(shards_dir: &Path) -> Result<()> {
    let shard_dirs = shard_directories(shards_dir)?;
    if shard_dirs.len() <= 10 {
        return Ok(());
    }

    let schema = lexical_schema();
    let fields = schema_fields(&schema).ok_or_else(|| {
        SieveError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "missing lexical schema fields",
        ))
    })?;
    let merged_id = next_shard_id(shards_dir)?;
    let temp_dir = shards_dir.join(format!(".tmp_seg_{merged_id}"));
    if temp_dir.exists() {
        fs::remove_dir_all(&temp_dir)?;
    }
    fs::create_dir_all(&temp_dir)?;
    let merged_index = TantivyIndex::open_or_create(MmapDirectory::open(&temp_dir)?, schema)?;
    let mut writer = merged_index.writer(50_000_000)?;

    for shard_dir in &shard_dirs {
        let shard_index = TantivyIndex::open_in_dir(shard_dir)?;
        let reader = shard_index.reader()?;
        let searcher = reader.searcher();
        for segment_reader in searcher.segment_readers() {
            let store_reader = segment_reader.get_store_reader(10)?;
            for doc_id in 0..segment_reader.max_doc() {
                let doc: TantivyDocument = store_reader.get(doc_id)?;
                let source_path = doc
                    .get_first(fields.source_path)
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                let content = doc
                    .get_first(fields.content)
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                let wal_entry_id = doc
                    .get_first(fields.wal_entry_id)
                    .and_then(|v| v.as_u64())
                    .unwrap_or_default();
                let byte_offset = doc
                    .get_first(fields.byte_offset)
                    .and_then(|v| v.as_u64())
                    .unwrap_or_default();
                let line_start = doc
                    .get_first(fields.line_range_start)
                    .and_then(|v| v.as_u64())
                    .unwrap_or(1);
                let line_end = doc
                    .get_first(fields.line_range_end)
                    .and_then(|v| v.as_u64())
                    .unwrap_or(line_start);
                writer.add_document(doc!(
                    fields.source_path => source_path,
                    fields.content => content,
                    fields.wal_entry_id => wal_entry_id,
                    fields.byte_offset => byte_offset,
                    fields.line_range_start => line_start,
                    fields.line_range_end => line_end,
                ))?;
            }
        }
    }

    writer.commit()?;
    let final_dir = shards_dir.join(format!("{SHARD_PREFIX}{merged_id:04}"));
    fs::rename(&temp_dir, &final_dir)?;
    for shard_dir in shard_dirs {
        fs::remove_dir_all(shard_dir)?;
    }
    Ok(())
}

fn persist_indexed_entries(shards_dir: &Path, bitmap: &RoaringTreemap) -> Result<()> {
    let path = shards_dir.join(INDEXED_ENTRIES_FILE);
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(path)?;
    bitmap
        .serialize_into(&mut file)
        .map_err(|err| SieveError::Io(std::io::Error::other(err.to_string())))?;
    file.flush()?;
    file.sync_data()?;
    Ok(())
}

fn lexical_schema() -> Schema {
    let mut builder = SchemaBuilder::default();
    builder.add_text_field("source_path", STRING | STORED);
    builder.add_text_field("content", TEXT | STORED);
    builder.add_u64_field("wal_entry_id", STORED);
    builder.add_u64_field("byte_offset", STORED);
    builder.add_u64_field("line_range_start", STORED);
    builder.add_u64_field("line_range_end", STORED);
    builder.build()
}

fn schema_fields(schema: &Schema) -> Option<LexicalFields> {
    Some(LexicalFields {
        source_path: schema.get_field("source_path").ok()?,
        content: schema.get_field("content").ok()?,
        wal_entry_id: schema.get_field("wal_entry_id").ok()?,
        byte_offset: schema.get_field("byte_offset").ok()?,
        line_range_start: schema.get_field("line_range_start").ok()?,
        line_range_end: schema.get_field("line_range_end").ok()?,
    })
}

fn next_shard_id(shards_dir: &Path) -> Result<usize> {
    let mut max_id = 0usize;
    for dir in shard_directories(shards_dir)? {
        if let Some(name) = dir.file_name().and_then(|n| n.to_str()) {
            if let Some(id_str) = name.strip_prefix(SHARD_PREFIX) {
                if let Ok(id) = id_str.parse::<usize>() {
                    max_id = max_id.max(id);
                }
            }
        }
    }
    Ok(max_id + 1)
}

fn shard_directories(shards_dir: &Path) -> Result<Vec<PathBuf>> {
    if !shards_dir.exists() {
        return Ok(Vec::new());
    }
    let mut dirs: Vec<PathBuf> = fs::read_dir(shards_dir)?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| {
            path.is_dir()
                && path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with(SHARD_PREFIX))
        })
        .collect();
    dirs.sort();
    Ok(dirs)
}

fn extract_snippet_and_line_range(
    content: &str,
    query: &str,
    default_range: (usize, usize),
) -> ((usize, usize), String) {
    let normalized = query.trim_matches('"');
    if let Some((idx, line)) = content
        .lines()
        .enumerate()
        .find(|(_, line)| line.contains(normalized))
    {
        let line_no = idx + 1;
        return ((line_no, line_no), line.to_string());
    }

    let terms: Vec<&str> = normalized
        .split_whitespace()
        .filter(|term| !term.is_empty())
        .collect();
    if let Some((idx, line)) = content
        .lines()
        .enumerate()
        .find(|(_, line)| terms.iter().any(|term| line.contains(term)))
    {
        let line_no = idx + 1;
        return ((line_no, line_no), line.to_string());
    }

    (
        default_range,
        content.lines().next().unwrap_or_default().to_string(),
    )
}
