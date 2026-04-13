use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResultSource {
    RawScan,
    ScanFallback,
    LexicalBm25,
    Fused,
}

impl ResultSource {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::RawScan => "scan",
            Self::ScanFallback => "scan:fallback",
            Self::LexicalBm25 => "bm25",
            Self::Fused => "fused",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScoredResult {
    pub source_path: String,
    pub line_range: (usize, usize),
    pub snippet: String,
    pub score: f64,
    pub source_layer: ResultSource,
    pub wal_entry_id: u64,
}

pub fn rrf_fuse(result_sets: Vec<Vec<ScoredResult>>, k: f64) -> Vec<ScoredResult> {
    let mut fused: HashMap<(String, usize), ScoredResult> = HashMap::new();

    for result_set in result_sets {
        for (rank, result) in result_set.into_iter().enumerate() {
            let key = (result.source_path.clone(), result.line_range.0);
            let contribution = 1.0 / (k + rank as f64 + 1.0);
            fused
                .entry(key)
                .and_modify(|existing| {
                    existing.score += contribution;
                    existing.source_layer = ResultSource::Fused;
                    if result.snippet.len() > existing.snippet.len() {
                        existing.snippet = result.snippet.clone();
                    }
                    existing.line_range = result.line_range;
                    existing.wal_entry_id = result.wal_entry_id;
                })
                .or_insert_with(|| ScoredResult {
                    score: contribution,
                    ..result
                });
        }
    }

    let mut results: Vec<ScoredResult> = fused.into_values().collect();
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.source_path.cmp(&b.source_path))
            .then_with(|| a.line_range.0.cmp(&b.line_range.0))
    });
    results
}
