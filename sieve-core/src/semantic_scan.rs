use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap, VecDeque};

use crate::df_prior::static_df_frac;
use crate::semantic_query::{GroupId, PhraseId, TermId};
use crate::surface::{BoundaryMode, RealizedPattern};
use crate::window_score::{compute_idf, score_window};

pub const MAX_PATTERN_DF: f32 = 0.08;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SemanticScanOptions {
    pub no_df_filter: bool,
    pub no_window_scoring: bool,
}

pub struct CompiledScanQuery {
    pub ac: aho_corasick::AhoCorasick,
    pub patterns: Vec<PatternMeta>,
}

pub struct PatternMeta {
    pub term_id: Option<TermId>,
    pub phrase_id: Option<PhraseId>,
    pub primary_group_id: GroupId,
    pub weight: f32,
    pub is_anchor: bool,
    pub boundary: BoundaryMode,
}

pub fn compile_scan_query(patterns: &[RealizedPattern]) -> crate::Result<CompiledScanQuery> {
    compile_scan_query_with_options(patterns, SemanticScanOptions::default())
}

pub fn compile_scan_query_with_options(
    patterns: &[RealizedPattern],
    options: SemanticScanOptions,
) -> crate::Result<CompiledScanQuery> {
    let mut filtered_patterns = patterns.to_vec();
    if !options.no_df_filter {
        filter_high_df_patterns(&mut filtered_patterns);
    }
    if filtered_patterns.is_empty() {
        return Err(crate::SieveError::Message(
            "semantic scan requires at least one pattern".to_string(),
        ));
    }
    let ac = aho_corasick::AhoCorasickBuilder::new()
        .match_kind(aho_corasick::MatchKind::Standard)
        .ascii_case_insensitive(false)
        .build(
            filtered_patterns
                .iter()
                .map(|pattern| pattern.bytes.as_slice()),
        )
        .map_err(|err| crate::SieveError::Message(err.to_string()))?;
    Ok(CompiledScanQuery {
        ac,
        patterns: filtered_patterns
            .iter()
            .map(|pattern| PatternMeta {
                term_id: pattern.term_id,
                phrase_id: pattern.phrase_id,
                primary_group_id: pattern.primary_group_id,
                weight: pattern.weight,
                is_anchor: pattern.is_anchor,
                boundary: pattern.boundary,
            })
            .collect(),
    })
}

pub fn filter_high_df_patterns(patterns: &mut Vec<RealizedPattern>) {
    let total = patterns.len();
    patterns.retain(|pattern| {
        let text = String::from_utf8_lossy(&pattern.bytes);
        let df = static_df_frac(&text);
        pattern.is_anchor || df <= MAX_PATTERN_DF
    });
    let dropped = total.saturating_sub(patterns.len());
    tracing::debug!("DF filter: dropped {} of {} patterns", dropped, total);
}

#[derive(Debug, Clone, PartialEq)]
pub struct MatchEvent {
    pub term_id: Option<TermId>,
    pub phrase_id: Option<PhraseId>,
    pub primary_group_id: GroupId,
    pub weight: f32,
    pub byte_start: u32,
    pub byte_end: u32,
    pub is_anchor: bool,
}

pub const WINDOW_BYTES: usize = 384;
pub const WINDOW_STRIDE: usize = 192;

#[derive(Debug, Clone, PartialEq)]
pub struct WindowAccumulator {
    pub wal_entry_id: u64,
    pub window_start: u32,
    pub window_end: u32,
    pub events: Vec<MatchEvent>,
    pub has_anchor: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScoredWindow {
    pub score: f32,
    pub wal_entry_id: u64,
    pub window_start: u32,
    pub window_end: u32,
    pub events: Vec<MatchEvent>,
    pub has_anchor: bool,
}

impl Eq for ScoredWindow {}

impl Ord for ScoredWindow {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| self.wal_entry_id.cmp(&other.wal_entry_id))
            .then_with(|| self.window_start.cmp(&other.window_start))
            .then_with(|| self.window_end.cmp(&other.window_end))
    }
}

impl PartialOrd for ScoredWindow {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub fn semantic_scan(
    compiled: &CompiledScanQuery,
    entries: &[(u64, &[u8], crate::semantic_query::ContentType)],
    query: &crate::semantic_query::SemanticQuery,
    top_k: usize,
) -> (Vec<ScoredWindow>, Vec<u32>) {
    semantic_scan_with_options(
        compiled,
        entries,
        query,
        top_k,
        SemanticScanOptions::default(),
    )
}

pub fn semantic_scan_with_options(
    compiled: &CompiledScanQuery,
    entries: &[(u64, &[u8], crate::semantic_query::ContentType)],
    query: &crate::semantic_query::SemanticQuery,
    top_k: usize,
    options: SemanticScanOptions,
) -> (Vec<ScoredWindow>, Vec<u32>) {
    let mut df_counts = vec![0u32; query.terms.len()];
    let mut entry_events = Vec::with_capacity(entries.len());

    for (wal_entry_id, hay, content_type) in entries {
        let events = collect_entry_events(compiled, hay);
        let mut term_seen = vec![false; query.terms.len()];
        for event in &events {
            if let Some(term_id) = event.term_id {
                let slot = term_id as usize;
                if slot < term_seen.len() && !term_seen[slot] {
                    df_counts[slot] += 1;
                    term_seen[slot] = true;
                }
            }
        }
        entry_events.push((
            *wal_entry_id,
            *content_type,
            hay.len(),
            events,
            compute_windows(hay, *content_type),
        ));
    }

    let idf: Vec<f32> = query
        .terms
        .iter()
        .enumerate()
        .map(|(idx, term)| {
            compute_idf(
                idx as TermId,
                df_counts.get(idx).copied().unwrap_or_default(),
                entries.len() as u32,
                static_df_frac(&term.canonical),
            )
        })
        .collect();

    if options.no_window_scoring {
        let mut windows = entry_events
            .into_iter()
            .filter_map(|(wal_entry_id, _content_type, hay_len, events, _windows)| {
                if events.is_empty() {
                    return None;
                }
                let has_anchor = events.iter().any(|event| event.is_anchor);
                if !has_anchor {
                    return None;
                }
                let window_start = events
                    .iter()
                    .map(|event| event.byte_start)
                    .min()
                    .unwrap_or(0);
                let window_end = events
                    .iter()
                    .map(|event| event.byte_end)
                    .max()
                    .unwrap_or(hay_len as u32)
                    .max(window_start + 1);
                Some(ScoredWindow {
                    score: events.len() as f32,
                    wal_entry_id,
                    window_start,
                    window_end,
                    events,
                    has_anchor,
                })
            })
            .collect::<Vec<_>>();
        windows.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.wal_entry_id.cmp(&right.wal_entry_id))
                .then_with(|| left.window_start.cmp(&right.window_start))
        });
        windows.truncate(top_k.max(1));
        return (windows, df_counts);
    }

    let mut heap: BinaryHeap<Reverse<ScoredWindow>> = BinaryHeap::new();
    let limit = top_k.max(1);

    for (wal_entry_id, _content_type, _hay_len, events, windows) in &entry_events {
        let mut deque: VecDeque<&MatchEvent> = VecDeque::new();
        let mut front = 0usize;
        for &(window_start, window_end) in windows {
            while deque
                .front()
                .is_some_and(|event| (event.byte_end as usize) <= window_start)
            {
                deque.pop_front();
            }
            while front < events.len() && (events[front].byte_start as usize) < window_end {
                deque.push_back(&events[front]);
                front += 1;
            }
            if deque.is_empty() {
                continue;
            }
            let bucket: Vec<MatchEvent> = deque.iter().map(|event| (*event).clone()).collect();
            let has_anchor = bucket.iter().any(|event| event.is_anchor);
            let window = WindowAccumulator {
                wal_entry_id: *wal_entry_id,
                window_start: window_start as u32,
                window_end: window_end as u32,
                events: bucket.clone(),
                has_anchor,
            };
            let score = score_window(&window, query, &idf);
            if score <= 0.0 {
                continue;
            }
            let scored = ScoredWindow {
                score,
                wal_entry_id: *wal_entry_id,
                window_start: window.window_start,
                window_end: window.window_end,
                events: bucket,
                has_anchor,
            };
            if heap.len() < limit {
                heap.push(Reverse(scored));
            } else if heap.peek().is_some_and(|min| scored.score > min.0.score) {
                heap.pop();
                heap.push(Reverse(scored));
            }
        }
    }

    let mut top_windows: Vec<ScoredWindow> = heap.into_iter().map(|entry| entry.0).collect();
    top_windows.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.wal_entry_id.cmp(&right.wal_entry_id))
            .then_with(|| left.window_start.cmp(&right.window_start))
    });
    (top_windows, df_counts)
}

fn collect_entry_events(compiled: &CompiledScanQuery, hay: &[u8]) -> Vec<MatchEvent> {
    let mut best: HashMap<(u32, u32, GroupId), (usize, MatchEvent)> = HashMap::new();
    for mat in compiled.ac.find_overlapping_iter(hay) {
        let meta = &compiled.patterns[mat.pattern().as_usize()];
        let start = mat.start();
        let end = mat.end();
        if !boundary_ok(hay, start, end, meta.boundary) {
            continue;
        }
        let event = MatchEvent {
            term_id: meta.term_id,
            phrase_id: meta.phrase_id,
            primary_group_id: meta.primary_group_id,
            weight: meta.weight,
            byte_start: start as u32,
            byte_end: end as u32,
            is_anchor: meta.is_anchor,
        };
        let key = (event.byte_start, event.byte_end, event.primary_group_id);
        let pattern_len = end.saturating_sub(start);
        match best.get(&key) {
            Some((existing_len, existing))
                if !should_replace(*existing_len, existing, pattern_len, &event) => {}
            _ => {
                best.insert(key, (pattern_len, event));
            }
        }
    }
    let mut events: Vec<MatchEvent> = best.into_values().map(|(_, event)| event).collect();
    events.sort_by(|a, b| {
        a.byte_start
            .cmp(&b.byte_start)
            .then_with(|| a.byte_end.cmp(&b.byte_end))
            .then_with(|| a.primary_group_id.cmp(&b.primary_group_id))
    });
    events
}

fn should_replace(
    existing_len: usize,
    existing: &MatchEvent,
    new_len: usize,
    new_event: &MatchEvent,
) -> bool {
    new_len > existing_len
        || (new_len == existing_len
            && (new_event.weight > existing.weight
                || (new_event.weight == existing.weight
                    && new_event.phrase_id.is_some()
                    && existing.phrase_id.is_none())))
}

pub fn compute_windows(
    hay: &[u8],
    content_type: crate::semantic_query::ContentType,
) -> Vec<(usize, usize)> {
    match content_type {
        crate::semantic_query::ContentType::Code => code_windows(hay),
        crate::semantic_query::ContentType::Prose => prose_windows(hay),
        crate::semantic_query::ContentType::Mixed => prose_windows(hay),
        crate::semantic_query::ContentType::Config
        | crate::semantic_query::ContentType::Log
        | crate::semantic_query::ContentType::Unknown => fixed_windows(hay.len()),
    }
}

fn fixed_windows(len: usize) -> Vec<(usize, usize)> {
    if len == 0 {
        return Vec::new();
    }
    if len <= WINDOW_BYTES {
        return vec![(0, len)];
    }
    let mut ranges = Vec::new();
    let mut start = 0usize;
    while start < len {
        let end = (start + WINDOW_BYTES).min(len);
        ranges.push((start, end));
        if end == len {
            break;
        }
        start += WINDOW_STRIDE;
    }
    ranges
}

fn prose_windows(hay: &[u8]) -> Vec<(usize, usize)> {
    let text = String::from_utf8_lossy(hay);
    let mut ranges = Vec::new();
    let mut start = 0usize;
    for paragraph in text.split("\n\n") {
        let para_len = paragraph.len();
        if para_len == 0 {
            start = start.saturating_add(2);
            continue;
        }
        let end = (start + para_len).min(hay.len());
        ranges.push((start, end));
        start = end.saturating_add(2).min(hay.len());
    }
    if ranges.is_empty() {
        fixed_windows(hay.len())
    } else {
        ranges
    }
}

fn code_windows(hay: &[u8]) -> Vec<(usize, usize)> {
    if hay.is_empty() {
        return Vec::new();
    }
    let text = String::from_utf8_lossy(hay);
    let mut raw_blocks = Vec::new();
    let mut start = 0usize;
    for block in text.split("\n\n") {
        let block_len = block.len();
        if block_len == 0 {
            start = start.saturating_add(2).min(hay.len());
            continue;
        }
        let end = (start + block_len).min(hay.len());
        raw_blocks.push((start, end));
        start = end.saturating_add(2).min(hay.len());
    }
    if raw_blocks.is_empty() {
        return fixed_windows(hay.len());
    }

    let mut merged = Vec::new();
    let mut current = raw_blocks[0];
    for block in raw_blocks.into_iter().skip(1) {
        if current.1.saturating_sub(current.0) < 128 {
            current.1 = block.1;
        } else {
            merged.push(current);
            current = block;
        }
    }
    merged.push(current);

    let mut windows = Vec::new();
    for (block_start, block_end) in merged {
        let block_len = block_end.saturating_sub(block_start);
        if block_len <= 1024 {
            windows.push((block_start, block_end));
            continue;
        }
        let mut segment_start = block_start;
        while segment_start < block_end {
            let segment_end = (segment_start + 1024).min(block_end);
            windows.push((segment_start, segment_end));
            if segment_end == block_end {
                break;
            }
            let stride = ((segment_end - segment_start) / 2).max(128);
            segment_start += stride;
        }
    }

    if windows.is_empty() {
        fixed_windows(hay.len())
    } else {
        windows
    }
}

fn is_identifier_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

fn is_code_separator(b: u8) -> bool {
    matches!(b, b'_' | b':' | b'.' | b'/' | b'\\' | b'-') || b.is_ascii_whitespace()
}

fn is_subtoken_left_boundary(hay: &[u8], start: usize) -> bool {
    if start == 0 {
        return true;
    }
    let prev = hay[start - 1];
    let current = hay[start];
    is_code_separator(prev)
        || (prev.is_ascii_lowercase() && current.is_ascii_uppercase())
        || (prev.is_ascii_digit() && current.is_ascii_alphabetic())
        || (prev.is_ascii_alphabetic() && current.is_ascii_digit())
}

fn is_subtoken_right_boundary(hay: &[u8], end: usize) -> bool {
    if end >= hay.len() {
        return true;
    }
    let prev = hay[end - 1];
    let next = hay[end];
    is_code_separator(next)
        || (prev.is_ascii_lowercase() && next.is_ascii_uppercase())
        || (prev.is_ascii_digit() && next.is_ascii_alphabetic())
        || (prev.is_ascii_alphabetic() && next.is_ascii_digit())
}

fn boundary_ok(hay: &[u8], start: usize, end: usize, mode: BoundaryMode) -> bool {
    match mode {
        BoundaryMode::None => true,
        BoundaryMode::Word => {
            let left = if start > 0 {
                !hay[start - 1].is_ascii_alphanumeric()
            } else {
                true
            };
            let right = if end < hay.len() {
                !hay[end].is_ascii_alphanumeric()
            } else {
                true
            };
            left && right
        }
        BoundaryMode::Identifier => {
            let left = if start > 0 {
                !is_identifier_char(hay[start - 1])
            } else {
                true
            };
            let right = if end < hay.len() {
                !is_identifier_char(hay[end])
            } else {
                true
            };
            left && right
        }
        BoundaryMode::CodeSubtoken => {
            is_subtoken_left_boundary(hay, start) && is_subtoken_right_boundary(hay, end)
        }
    }
}
