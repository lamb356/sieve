use std::collections::HashSet;
use std::path::Path;
use std::sync::Mutex;

use ndarray::{Array2, Array3, ArrayViewD};
use ort::{session::Session, value::TensorRef};

use crate::semantic_query::{GroupId, SemanticQuery};
use crate::semantic_scan::WindowAccumulator;
use crate::{Result, SieveError};

pub const MAX_EVENT_TOKENS: usize = 16;
pub const EVENT_FEATS: usize = 12;
pub const GLOBAL_FEATS: usize = 12;
pub const MAX_RERANK_WINDOWS: usize = 32;

#[derive(Debug, Clone, PartialEq)]
pub struct EventRerankInput {
    pub events: [[f32; EVENT_FEATS]; MAX_EVENT_TOKENS],
    pub mask: [u8; MAX_EVENT_TOKENS],
    pub global: [f32; GLOBAL_FEATS],
}

#[derive(Debug)]
pub struct EventReranker {
    session: Mutex<Session>,
}

impl EventReranker {
    pub fn load(model_path: &Path) -> Result<Self> {
        let session = Session::builder()
            .map_err(map_external_error)?
            .commit_from_file(model_path)
            .map_err(map_external_error)?;
        Ok(Self {
            session: Mutex::new(session),
        })
    }

    pub fn rerank(
        &self,
        windows: &mut [WindowAccumulator],
        query: &SemanticQuery,
        idf: &[f32],
        formula_scores: &[f32],
    ) -> Result<Vec<f32>> {
        let limit = windows
            .len()
            .min(MAX_RERANK_WINDOWS)
            .min(formula_scores.len());
        if limit == 0 {
            return Ok(Vec::new());
        }

        let mut events = Array3::<f32>::zeros((limit, MAX_EVENT_TOKENS, EVENT_FEATS));
        let mut masks = Array2::<f32>::zeros((limit, MAX_EVENT_TOKENS));
        let mut globals = Array2::<f32>::zeros((limit, GLOBAL_FEATS));

        for idx in 0..limit {
            let input = extract_rerank_input(&windows[idx], query, idf, formula_scores[idx]);
            for token_idx in 0..MAX_EVENT_TOKENS {
                for feat_idx in 0..EVENT_FEATS {
                    events[(idx, token_idx, feat_idx)] = input.events[token_idx][feat_idx];
                }
                masks[(idx, token_idx)] = input.mask[token_idx] as f32;
            }
            for feat_idx in 0..GLOBAL_FEATS {
                globals[(idx, feat_idx)] = input.global[feat_idx];
            }
        }

        let mut session = self.session.lock().map_err(|_| SieveError::LockPoisoned)?;
        let outputs = session
            .run(ort::inputs![
                TensorRef::from_array_view(events.view()).map_err(map_external_error)?,
                TensorRef::from_array_view(masks.view()).map_err(map_external_error)?,
                TensorRef::from_array_view(globals.view()).map_err(map_external_error)?
            ])
            .map_err(map_external_error)?;
        let scores: ArrayViewD<'_, f32> =
            outputs[0].try_extract_array().map_err(map_external_error)?;
        Ok(scores.iter().copied().collect())
    }
}

pub fn should_rerank_event_windows(
    query: &SemanticQuery,
    candidate_count: usize,
    event_model_available: bool,
) -> bool {
    event_model_available && query.groups.len() >= 2 && candidate_count > 8
}

pub fn extract_rerank_input(
    window: &WindowAccumulator,
    query: &SemanticQuery,
    idf: &[f32],
    formula_score: f32,
) -> EventRerankInput {
    let window_len = (window.window_end.saturating_sub(window.window_start)).max(1) as f32;
    let mut scored_events: Vec<_> = window
        .events
        .iter()
        .map(|event| {
            let term_idx = event.term_id.unwrap_or_default() as usize;
            let idf_value = idf.get(term_idx).copied().unwrap_or(1.0);
            (event, event.weight * idf_value)
        })
        .collect();
    scored_events.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored_events.truncate(MAX_EVENT_TOKENS);
    scored_events.sort_by(|a, b| {
        a.0.byte_start
            .cmp(&b.0.byte_start)
            .then_with(|| a.0.byte_end.cmp(&b.0.byte_end))
    });

    let mut events = [[0.0f32; EVENT_FEATS]; MAX_EVENT_TOKENS];
    let mut mask = [0u8; MAX_EVENT_TOKENS];
    let num_groups = query.groups.len().max(1) as f32;
    let query_group_mass = query.total_group_importance.max(1.0);
    let mut seen_groups = HashSet::new();
    let mut matched_groups = HashSet::new();
    let mut anchor_count = 0u8;
    let mut phrase_count = 0u8;
    let mut max_event_gain = 0.0f32;
    let mut total_group_mass = 0.0f32;
    let mut idf_sum = 0.0f32;
    let mut prev_end = None;
    let mut prev_group = None;
    let mut top_proximity = 0.0f32;
    let mut ordered_pair_mass = 0.0f32;
    let mut group_positions: Vec<(GroupId, f32)> = Vec::new();

    for (idx, (event, gain)) in scored_events.iter().enumerate() {
        mask[idx] = 1;
        let term_idx = event.term_id.unwrap_or_default() as usize;
        let term = event.term_id.and_then(|id| query.terms.get(id as usize));
        let group = query
            .groups
            .iter()
            .find(|group| group.group_id == event.primary_group_id);
        let query_weight = term.map(|term| term.norm_weight).unwrap_or(event.weight);
        let idf_value = idf.get(term_idx).copied().unwrap_or(1.0);
        let group_importance = group.map(|group| group.importance).unwrap_or(0.0);
        let normalized_ordinal = group
            .map(|group| group.query_ordinal as f32 / num_groups)
            .unwrap_or(0.0);
        let repeated_group = seen_groups.contains(&event.primary_group_id);
        let same_group_as_prev = prev_group.is_some_and(|prev| prev == event.primary_group_id);
        let relative_start =
            event.byte_start.saturating_sub(window.window_start) as f32 / window_len;
        let relative_end = event.byte_end.saturating_sub(window.window_start) as f32 / window_len;
        let gap = prev_end
            .map(|prev| event.byte_start.saturating_sub(prev) as f32 / window_len)
            .unwrap_or(0.0);
        let identifier_variant = term
            .map(|term| {
                term.surface_variants.iter().any(|variant| {
                    matches!(
                        variant.kind,
                        crate::surface::VariantKind::Snake
                            | crate::surface::VariantKind::Camel
                            | crate::surface::VariantKind::Pascal
                            | crate::surface::VariantKind::Kebab
                            | crate::surface::VariantKind::ScreamingSnake
                    )
                })
            })
            .unwrap_or(false);

        events[idx] = [
            relative_start,
            relative_end,
            gap,
            query_weight,
            idf_value,
            group_importance,
            normalized_ordinal,
            if event.is_anchor { 1.0 } else { 0.0 },
            if event.phrase_id.is_some() { 1.0 } else { 0.0 },
            if identifier_variant { 1.0 } else { 0.0 },
            if repeated_group { 1.0 } else { 0.0 },
            if same_group_as_prev { 1.0 } else { 0.0 },
        ];

        seen_groups.insert(event.primary_group_id);
        matched_groups.insert(event.primary_group_id);
        if event.is_anchor {
            anchor_count = anchor_count.saturating_add(1);
        }
        if event.phrase_id.is_some() {
            phrase_count = phrase_count.saturating_add(1);
        }
        max_event_gain = max_event_gain.max(*gain);
        total_group_mass += group_importance;
        idf_sum += idf_value;
        let center = (event.byte_start as f32 + event.byte_end as f32) * 0.5;
        for (_, prev_center) in &group_positions {
            let proximity = 1.0 / (1.0 + ((center - *prev_center).abs() / window_len));
            top_proximity = top_proximity.max(proximity);
        }
        group_positions.push((event.primary_group_id, center));
        if let Some(prev) = prev_group {
            if prev != event.primary_group_id {
                ordered_pair_mass += (query_weight * idf_value).sqrt();
            }
        }
        prev_end = Some(event.byte_end);
        prev_group = Some(event.primary_group_id);
    }

    let event_count = scored_events.len() as u8;
    let matched_group_coverage = matched_groups.len() as f32 / num_groups;
    let common_term_penalty = scored_events
        .iter()
        .map(|(event, _)| {
            let term_idx = event.term_id.unwrap_or_default() as usize;
            1.0 - idf.get(term_idx).copied().unwrap_or(1.0)
        })
        .sum::<f32>();
    let mean_idf = if event_count == 0 {
        0.0
    } else {
        idf_sum / event_count as f32
    };

    let global = [
        formula_score,
        matched_group_coverage,
        anchor_count as f32 / MAX_EVENT_TOKENS as f32,
        phrase_count as f32 / MAX_EVENT_TOKENS as f32,
        top_proximity,
        ordered_pair_mass,
        common_term_penalty,
        total_group_mass / query_group_mass,
        event_count as f32 / MAX_EVENT_TOKENS as f32,
        query.groups.len() as f32 / 16.0,
        max_event_gain,
        mean_idf,
    ];

    EventRerankInput {
        events,
        mask,
        global,
    }
}

fn map_external_error(error: impl std::fmt::Display) -> SieveError {
    SieveError::Io(std::io::Error::other(error.to_string()))
}
