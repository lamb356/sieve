#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResultId {
    pub wal_entry_id: u64,
    pub byte_start: u32,
    pub byte_end: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResultSource {
    RawScan,
    ScanFallback,
    LexicalBm25,
    SpladeBm25,
    SemanticScan,
    EventReranked,
    HotVector,
    DeltaFallback,
    QueryPromoted,
    Fused,
}

impl ResultSource {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::RawScan => "scan",
            Self::ScanFallback => "scan:fallback",
            Self::LexicalBm25 => "bm25",
            Self::SpladeBm25 => "splade-bm25",
            Self::SemanticScan => "semantic",
            Self::EventReranked => "event-reranked",
            Self::HotVector => "vec",
            Self::DeltaFallback => "delta",
            Self::QueryPromoted => "promoted",
            Self::Fused => "fused",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CoverageState {
    Complete,
    Partial(f32),
    Unavailable,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LayerResults {
    pub source: ResultSource,
    pub weight: f64,
    pub results: Vec<ScoredResult>,
    pub coverage: CoverageState,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScoredResult {
    pub result_id: ResultId,
    pub source_path: String,
    pub line_range: (usize, usize),
    pub chunk_id: u32,
    pub snippet: String,
    pub score: f64,
    pub source_layer: ResultSource,
    pub wal_entry_id: u64,
}

pub fn rrf_fuse(result_sets: Vec<Vec<ScoredResult>>, k: f64) -> Vec<ScoredResult> {
    weighted_rrf_fuse(
        result_sets
            .into_iter()
            .map(|results| (ResultSource::Fused, 1.0, results))
            .collect(),
        k,
    )
}

pub fn weighted_rrf_fuse(
    result_sets: Vec<(ResultSource, f64, Vec<ScoredResult>)>,
    k: f64,
) -> Vec<ScoredResult> {
    let result_sets = result_sets
        .into_iter()
        .map(|(source, weight, results)| LayerResults {
            source,
            weight,
            results,
            coverage: CoverageState::Complete,
        })
        .collect();
    coverage_aware_rrf_fuse(result_sets, k)
}

pub fn compute_layer_weight(
    base_weight: f64,
    coverage: &CoverageState,
    layer_confidence: f64,
) -> f64 {
    let coverage_factor = match coverage {
        CoverageState::Complete => 1.0,
        CoverageState::Partial(frac) => frac.clamp(0.0, 1.0) as f64,
        CoverageState::Unavailable => 0.0,
    };
    let confidence_bonus = if matches!(coverage, CoverageState::Complete) {
        1.0 + layer_confidence.max(0.0)
    } else {
        1.0
    };
    base_weight * coverage_factor * confidence_bonus
}

pub fn coverage_aware_rrf_fuse(result_sets: Vec<LayerResults>, k: f64) -> Vec<ScoredResult> {
    let complete_count = result_sets
        .iter()
        .filter(|set| matches!(set.coverage, CoverageState::Complete) && !set.results.is_empty())
        .count();
    let mut fused: Vec<(ScoredResult, f64, bool, bool)> = Vec::new();

    for layer in result_sets {
        if matches!(layer.coverage, CoverageState::Unavailable) || layer.results.is_empty() {
            continue;
        }
        let layer_confidence =
            if complete_count == 1 && matches!(layer.coverage, CoverageState::Complete) {
                score_gap_confidence(&layer.results)
            } else {
                0.0
            };
        let layer_weight = compute_layer_weight(layer.weight, &layer.coverage, layer_confidence);
        if layer_weight <= 0.0 {
            continue;
        }

        for (rank, result) in layer.results.into_iter().enumerate() {
            let contribution = layer_weight / (k + rank as f64 + 1.0);
            if let Some((existing, best_raw_score, seen_complete, seen_partial)) =
                fused.iter_mut().find(|(existing, _, _, _)| {
                    existing.result_id.wal_entry_id == result.result_id.wal_entry_id
                        && range_iou(existing.result_id, result.result_id) > 0.5
                })
            {
                existing.score += contribution;
                existing.source_layer = ResultSource::Fused;
                *seen_complete |= matches!(layer.coverage, CoverageState::Complete);
                *seen_partial |= matches!(layer.coverage, CoverageState::Partial(_));
                if result.score > *best_raw_score
                    || (result.score == *best_raw_score
                        && result.snippet.len() > existing.snippet.len())
                {
                    let cumulative_score = existing.score;
                    let source_layer = existing.source_layer;
                    existing.result_id = result.result_id;
                    existing.source_path = result.source_path;
                    existing.line_range = result.line_range;
                    existing.chunk_id = result.chunk_id;
                    existing.snippet = result.snippet;
                    existing.wal_entry_id = result.wal_entry_id;
                    existing.score = cumulative_score;
                    existing.source_layer = source_layer;
                    *best_raw_score = result.score;
                }
            } else {
                let raw_score = result.score;
                let mut fused_result = result;
                fused_result.score = contribution;
                fused_result.source_layer = if layer.source == ResultSource::Fused {
                    fused_result.source_layer
                } else {
                    ResultSource::Fused
                };
                fused.push((
                    fused_result,
                    raw_score,
                    matches!(layer.coverage, CoverageState::Complete),
                    matches!(layer.coverage, CoverageState::Partial(_)),
                ));
            }
        }
    }

    if complete_count == 1 {
        for (result, _raw_score, seen_complete, seen_partial) in &mut fused {
            if *seen_complete && !*seen_partial {
                result.score *= 1.5;
            } else if *seen_complete {
                result.score *= 1.15;
            }
        }
    } else {
        for (result, _raw_score, seen_complete, seen_partial) in &mut fused {
            if *seen_complete && !*seen_partial {
                result.score *= 1.35;
            }
        }
    }

    let mut results: Vec<ScoredResult> =
        fused.into_iter().map(|(result, _, _, _)| result).collect();
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.source_path.cmp(&b.source_path))
            .then_with(|| a.line_range.0.cmp(&b.line_range.0))
    });
    results
}

fn score_gap_confidence(results: &[ScoredResult]) -> f64 {
    if results.is_empty() {
        return 0.0;
    }
    if results.len() == 1 {
        return 1.0;
    }
    let first = results[0].score;
    let second = results[1].score;
    if !first.is_finite() || !second.is_finite() {
        return 0.0;
    }
    let denom = first.abs().max(1.0);
    ((first - second) / denom).clamp(0.0, 1.0)
}

fn range_iou(left: ResultId, right: ResultId) -> f64 {
    if left.wal_entry_id != right.wal_entry_id {
        return 0.0;
    }
    let left_start = left.byte_start.min(left.byte_end);
    let left_end = left.byte_start.max(left.byte_end);
    let right_start = right.byte_start.min(right.byte_end);
    let right_end = right.byte_start.max(right.byte_end);
    let intersection_start = left_start.max(right_start);
    let intersection_end = left_end.min(right_end);
    if intersection_end <= intersection_start {
        return 0.0;
    }
    let union_start = left_start.min(right_start);
    let union_end = left_end.max(right_end);
    let union = union_end.saturating_sub(union_start) as f64;
    if union == 0.0 {
        return 0.0;
    }
    (intersection_end.saturating_sub(intersection_start) as f64) / union
}
