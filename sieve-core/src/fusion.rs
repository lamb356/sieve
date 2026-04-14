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
    SemanticScan,
    HotVector,
    DeltaFallback,
    Fused,
}

impl ResultSource {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::RawScan => "scan",
            Self::ScanFallback => "scan:fallback",
            Self::LexicalBm25 => "bm25",
            Self::SemanticScan => "semantic",
            Self::HotVector => "vec",
            Self::DeltaFallback => "delta",
            Self::Fused => "fused",
        }
    }
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
    let mut fused: Vec<(ScoredResult, f64)> = Vec::new();

    for (source, weight, result_set) in result_sets {
        for (rank, result) in result_set.into_iter().enumerate() {
            let contribution = weight / (k + rank as f64 + 1.0);
            if let Some((existing, best_raw_score)) = fused.iter_mut().find(|(existing, _)| {
                existing.result_id.wal_entry_id == result.result_id.wal_entry_id
                    && range_iou(existing.result_id, result.result_id) > 0.5
            }) {
                existing.score += contribution;
                existing.source_layer = ResultSource::Fused;
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
                fused_result.source_layer = if source == ResultSource::Fused {
                    fused_result.source_layer
                } else {
                    ResultSource::Fused
                };
                fused.push((fused_result, raw_score));
            }
        }
    }

    let mut results: Vec<ScoredResult> = fused.into_iter().map(|(result, _)| result).collect();
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.source_path.cmp(&b.source_path))
            .then_with(|| a.line_range.0.cmp(&b.line_range.0))
    });
    results
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
