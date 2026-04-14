use crate::semantic_query::{GroupId, PhraseId, TermId};
use crate::surface::BoundaryMode;

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

pub struct WindowAccumulator {
    pub wal_entry_id: u64,
    pub window_start: u32,
    pub window_end: u32,
    pub events: Vec<MatchEvent>,
    pub has_anchor: bool,
}

pub fn semantic_scan(
    compiled: &CompiledScanQuery,
    entries: &[(u64, &[u8])],
    query: &crate::semantic_query::SemanticQuery,
) -> (Vec<WindowAccumulator>, Vec<u32>) {
    let mut windows = Vec::new();
    let mut df_counts = vec![0u32; query.terms.len()];

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

        for (window_start, window_end) in window_ranges(hay.len()) {
            let mut bucket = Vec::new();
            let mut has_anchor = false;
            for event in &events {
                if event.byte_start < window_end as u32 && event.byte_end > window_start as u32 {
                    has_anchor |= event.is_anchor;
                    bucket.push(MatchEvent {
                        term_id: event.term_id,
                        phrase_id: event.phrase_id,
                        primary_group_id: event.primary_group_id,
                        weight: event.weight,
                        byte_start: event.byte_start,
                        byte_end: event.byte_end,
                        is_anchor: event.is_anchor,
                    });
                }
            }
            if !bucket.is_empty() {
                windows.push(WindowAccumulator {
                    wal_entry_id: *wal_entry_id,
                    window_start: window_start as u32,
                    window_end: window_end as u32,
                    events: bucket,
                    has_anchor,
                });
            }
        }
    }

    (windows, df_counts)
}

fn collect_entry_events(compiled: &CompiledScanQuery, hay: &[u8]) -> Vec<MatchEvent> {
    let mut best: std::collections::HashMap<(u32, u32, GroupId), (usize, MatchEvent)> =
        std::collections::HashMap::new();
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
        BoundaryMode::Word | BoundaryMode::Identifier => {
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
