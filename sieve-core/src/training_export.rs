use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

use serde::Serialize;

use crate::aliases::AliasLexicon;
use crate::default_queries::DEFAULT_TRAINING_QUERIES;
use crate::event_rerank::{extract_rerank_input, EVENT_FEATS, GLOBAL_FEATS};
use crate::model::DEFAULT_SPARSE_MODEL_NAME;
use crate::semantic_query::{SemanticGroup, SemanticQuery};
use crate::{
    default_sieve_data_dir, plan_query, semantic_scan_scored_windows, Index, QueryPlan, Result,
};

#[derive(Debug, Clone, Serialize)]
pub struct TrainingExample {
    pub query: String,
    pub query_groups: Vec<TrainingGroup>,
    pub window: TrainingWindow,
}

#[derive(Debug, Clone, Serialize)]
pub struct TrainingGroup {
    pub group_id: u16,
    pub canonical: String,
    pub importance: f32,
    pub is_seed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct TrainingWindow {
    pub wal_entry_id: u64,
    pub window_start: u32,
    pub window_end: u32,
    pub raw_text: String,
    pub formula_score: f32,
    pub events: Vec<TrainingEvent>,
    pub global_features: TrainingGlobal,
}

#[derive(Debug, Clone, Serialize)]
pub struct TrainingEvent {
    pub relative_start: f32,
    pub relative_end: f32,
    pub relative_gap_prev: f32,
    pub query_weight: f32,
    pub idf: f32,
    pub group_importance: f32,
    pub normalized_ordinal: f32,
    pub is_anchor: bool,
    pub is_phrase: bool,
    pub is_identifier_variant: bool,
    pub repeated_group: bool,
    pub same_group_as_prev: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct TrainingGlobal {
    pub formula_score: f32,
    pub matched_group_coverage: f32,
    pub anchor_count: u8,
    pub phrase_count: u8,
    pub top_proximity: f32,
    pub ordered_pair_mass: f32,
    pub common_term_penalty: f32,
    pub total_group_mass: f32,
    pub event_count: u8,
    pub query_group_count: u8,
    pub max_event_gain: f32,
    pub mean_idf: f32,
}

pub fn export_training_data(
    index: &Index,
    queries: &[String],
    output_path: &Path,
    top_k_windows: usize,
) -> Result<usize> {
    let mut output = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(output_path)?;
    let aliases = AliasLexicon::built_in();
    let query_list: Vec<String> = if queries.is_empty() {
        DEFAULT_TRAINING_QUERIES
            .iter()
            .map(|query| (*query).to_string())
            .collect()
    } else {
        queries.to_vec()
    };
    let snapshot = index.snapshot_search_partition()?;
    let metadata: Vec<_> = index
        .metadata_snapshot()?
        .into_iter()
        .filter(|entry| snapshot.active_ids.contains(entry.wal_entry_id))
        .collect();
    let mut exported = 0usize;

    for raw_query in query_list {
        let semantic_query = compile_training_query(index, &raw_query, &aliases)?;
        let (windows, _scan_timing) = semantic_scan_scored_windows(
            index.wal_content_path(),
            &metadata,
            &semantic_query,
            crate::semantic_scan::SemanticScanOptions::default(),
        )?;
        for (window, scored) in windows.into_iter().take(top_k_windows.max(1)) {
            let entry_content = index
                .read_entry_content(window.wal_entry_id)
                .unwrap_or_default();
            let raw_text =
                slice_window_text(&entry_content, window.window_start, window.window_end);
            let accumulator = crate::semantic_scan::WindowAccumulator {
                wal_entry_id: window.wal_entry_id,
                window_start: window.window_start,
                window_end: window.window_end,
                events: window.events.clone(),
                has_anchor: window.has_anchor,
            };
            let input = extract_rerank_input(
                &accumulator,
                &semantic_query,
                &vec![1.0; semantic_query.terms.len().max(1)],
                scored.score as f32,
            );
            let events = build_training_events(&input);
            let global_features = build_training_global(&input, scored.score as f32);
            let example = TrainingExample {
                query: raw_query.clone(),
                query_groups: semantic_query
                    .groups
                    .iter()
                    .map(|group| TrainingGroup {
                        group_id: group.group_id,
                        canonical: group.canonical.clone(),
                        importance: group.importance,
                        is_seed: group.is_seed,
                    })
                    .collect(),
                window: TrainingWindow {
                    wal_entry_id: window.wal_entry_id,
                    window_start: window.window_start,
                    window_end: window.window_end,
                    raw_text,
                    formula_score: scored.score as f32,
                    events,
                    global_features,
                },
            };
            serde_json::to_writer(&mut output, &example)?;
            output.write_all(b"\n")?;
            exported += 1;
        }
    }

    output.flush()?;
    Ok(exported)
}

fn compile_training_query(
    _index: &Index,
    raw_query: &str,
    aliases: &AliasLexicon,
) -> Result<Arc<SemanticQuery>> {
    let manager = crate::model::ModelManager::new(&default_sieve_data_dir());
    let sparse = if manager.is_cached(DEFAULT_SPARSE_MODEL_NAME) {
        manager.ensure_sparse_model().ok().and_then(|handle| {
            crate::sparse::SpladeEncoder::load(&handle.model_path, &handle.tokenizer_path)
                .ok()
                .map(Arc::new)
        })
    } else {
        None
    };
    let plan = plan_query(
        raw_query,
        sparse.as_deref(),
        aliases,
        &crate::SearchOptions::default(),
        crate::semantic_query::ContentType::Prose,
    );
    Ok(match plan {
        QueryPlan::Semantic(query) => query,
        _ => Arc::new(fallback_semantic_query(raw_query)),
    })
}

fn build_training_events(input: &crate::event_rerank::EventRerankInput) -> Vec<TrainingEvent> {
    let mut events = Vec::new();
    for idx in 0..input.mask.len() {
        if input.mask[idx] == 0 {
            continue;
        }
        let row = input.events[idx];
        debug_assert_eq!(row.len(), EVENT_FEATS);
        events.push(TrainingEvent {
            relative_start: row[0],
            relative_end: row[1],
            relative_gap_prev: row[2],
            query_weight: row[3],
            idf: row[4],
            group_importance: row[5],
            normalized_ordinal: row[6],
            is_anchor: row[7] > 0.0,
            is_phrase: row[8] > 0.0,
            is_identifier_variant: row[9] > 0.0,
            repeated_group: row[10] > 0.0,
            same_group_as_prev: row[11] > 0.0,
        });
    }
    events
}

fn build_training_global(
    input: &crate::event_rerank::EventRerankInput,
    formula_score: f32,
) -> TrainingGlobal {
    debug_assert_eq!(input.global.len(), GLOBAL_FEATS);
    TrainingGlobal {
        formula_score,
        matched_group_coverage: input.global[1],
        anchor_count: (input.global[2] * 16.0).round() as u8,
        phrase_count: (input.global[3] * 16.0).round() as u8,
        top_proximity: input.global[4],
        ordered_pair_mass: input.global[5],
        common_term_penalty: input.global[6],
        total_group_mass: input.global[7],
        event_count: (input.global[8] * 16.0).round() as u8,
        query_group_count: (input.global[9] * 16.0).round() as u8,
        max_event_gain: input.global[10],
        mean_idf: input.global[11],
    }
}

fn slice_window_text(content: &str, window_start: u32, window_end: u32) -> String {
    let bytes = content.as_bytes();
    let start = window_start as usize;
    let end = (window_end as usize).min(bytes.len());
    if start >= end {
        String::new()
    } else {
        String::from_utf8_lossy(&bytes[start..end]).into_owned()
    }
}

fn fallback_semantic_query(raw_query: &str) -> SemanticQuery {
    let normalized = raw_query.to_lowercase();
    let tokens: Vec<String> = normalized
        .split_whitespace()
        .filter(|token| !token.is_empty())
        .map(str::to_string)
        .collect();
    let groups: Vec<SemanticGroup> = tokens
        .iter()
        .enumerate()
        .map(|(idx, token)| SemanticGroup {
            group_id: idx as u16,
            canonical: token.clone(),
            query_ordinal: idx as u8,
            is_seed: true,
            importance: 1.0,
            member_terms: vec![idx as u16],
        })
        .collect();
    let terms = tokens
        .iter()
        .enumerate()
        .map(|(idx, token)| crate::semantic_query::SemanticTerm {
            term_id: idx as u16,
            vocab_id: idx as u32,
            vocab_piece: token.clone(),
            canonical: token.clone(),
            raw_weight: 1.0,
            norm_weight: 1.0,
            group_id: idx as u16,
            is_anchor: true,
            source: crate::semantic_query::TermSource::OriginalToken,
            surface_variants: vec![crate::surface::SurfaceVariant {
                text: token.clone(),
                bytes: token.as_bytes().to_vec(),
                kind: crate::surface::VariantKind::Lower,
                boundary: crate::surface::BoundaryMode::Word,
                quality: 1.0,
            }],
        })
        .collect();
    let phrases = if tokens.len() >= 2 {
        vec![crate::semantic_query::PhrasePattern {
            phrase_id: 0,
            canonical: tokens[..2].join(" "),
            component_groups: vec![0, 1],
            raw_weight: 1.0,
            norm_weight: 1.0,
            is_anchor: true,
            surface_variants: vec![crate::surface::SurfaceVariant {
                text: tokens[..2].join(" "),
                bytes: tokens[..2].join(" ").into_bytes(),
                kind: crate::surface::VariantKind::Lower,
                boundary: crate::surface::BoundaryMode::Word,
                quality: 1.0,
            }],
        }]
    } else {
        Vec::new()
    };
    SemanticQuery {
        raw_query: raw_query.to_string(),
        normalized_query: normalized,
        content_type: crate::semantic_query::ContentType::Prose,
        tokens: Vec::new(),
        seeds: Vec::new(),
        groups,
        terms,
        phrases,
        query_order: (0..tokens.len() as u16).collect(),
        total_group_importance: tokens.len() as f32,
    }
}
