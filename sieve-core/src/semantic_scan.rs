use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap, VecDeque};

use crate::df_prior::static_df_frac;
use crate::semantic_query::{GroupId, PhraseId, TermId};
use crate::surface::BoundaryMode;
use crate::window_score::{compute_idf, score_window};

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

pub fn compile_scan_query(
    patterns: &[crate::surface::RealizedPattern],
) -> crate::Result<CompiledScanQuery> {
    if patterns.is_empty() {
        return Err(crate::SieveError::Message(
            "semantic scan requires at least one pattern".to_string(),
        ));
    }
    let ac = aho_corasick::AhoCorasickBuilder::new()
        .match_kind(aho_corasick::MatchKind::Standard)
        .ascii_case_insensitive(false)
        .build(patterns.iter().map(|pattern| pattern.bytes.as_slice()))
        .map_err(|err| crate::SieveError::Message(err.to_string()))?;
    Ok(CompiledScanQuery {
        ac,
        patterns: patterns
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

pub const WINDOW_BYTES: usize = 512;
pub const WINDOW_STRIDE: usize = 256;

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
    entries: &[(u64, &[u8])],
    query: &crate::semantic_query::SemanticQuery,
    top_k: usize,
) -> (Vec<ScoredWindow>, Vec<u32>) {
    let mut df_counts = vec![0u32; query.terms.len()];
    let mut entry_events = Vec::with_capacity(entries.len());

    for (wal_entry_id, hay) in entries {
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
        entry_events.push((*wal_entry_id, hay.len(), events));
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

    let mut heap: BinaryHeap<Reverse<ScoredWindow>> = BinaryHeap::new();
    let limit = top_k.max(1);

    for (wal_entry_id, hay_len, events) in &entry_events {
        let mut deque: VecDeque<&MatchEvent> = VecDeque::new();
        let mut front = 0usize;
        for (window_start, window_end) in window_ranges(*hay_len) {
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

fn window_ranges(len: usize) -> Vec<(usize, usize)> {
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

fn is_identifier_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
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
    }
}
