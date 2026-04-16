use sieve_core::fusion::{
    compute_layer_weight, coverage_aware_rrf_fuse, rrf_fuse, weighted_rrf_fuse, CoverageState,
    LayerResults, ResultId, ResultSource, ScoredResult,
};

#[test]
fn test_rrf_fusion() {
    let scan = vec![
        ScoredResult {
            result_id: ResultId {
                wal_entry_id: 1,
                byte_start: 0,
                byte_end: 10,
            },
            source_path: "a.rs".into(),
            line_range: (1, 1),
            chunk_id: 0,
            snippet: "alpha".into(),
            score: 1.0,
            source_layer: ResultSource::RawScan,
            wal_entry_id: 1,
        },
        ScoredResult {
            result_id: ResultId {
                wal_entry_id: 2,
                byte_start: 0,
                byte_end: 10,
            },
            source_path: "b.rs".into(),
            line_range: (1, 1),
            chunk_id: 0,
            snippet: "beta".into(),
            score: 1.0,
            source_layer: ResultSource::RawScan,
            wal_entry_id: 2,
        },
    ];

    let bm25 = vec![
        ScoredResult {
            result_id: ResultId {
                wal_entry_id: 2,
                byte_start: 0,
                byte_end: 10,
            },
            source_path: "b.rs".into(),
            line_range: (1, 1),
            chunk_id: 0,
            snippet: "beta".into(),
            score: 10.0,
            source_layer: ResultSource::LexicalBm25,
            wal_entry_id: 2,
        },
        ScoredResult {
            result_id: ResultId {
                wal_entry_id: 3,
                byte_start: 0,
                byte_end: 10,
            },
            source_path: "c.rs".into(),
            line_range: (1, 1),
            chunk_id: 0,
            snippet: "gamma".into(),
            score: 9.0,
            source_layer: ResultSource::LexicalBm25,
            wal_entry_id: 3,
        },
    ];

    let fused = rrf_fuse(vec![scan, bm25], 5.0);
    let paths: Vec<_> = fused.iter().map(|r| r.source_path.as_str()).collect();

    assert_eq!(paths[0], "b.rs");
    assert!(paths.contains(&"a.rs"));
    assert!(paths.contains(&"c.rs"));
}

#[test]
fn test_rrf_dedup() {
    let fused = rrf_fuse(
        vec![
            vec![ScoredResult {
                result_id: ResultId {
                    wal_entry_id: 9,
                    byte_start: 30,
                    byte_end: 40,
                },
                source_path: "same.rs".into(),
                line_range: (4, 4),
                chunk_id: 0,
                snippet: "same".into(),
                score: 1.0,
                source_layer: ResultSource::RawScan,
                wal_entry_id: 9,
            }],
            vec![ScoredResult {
                result_id: ResultId {
                    wal_entry_id: 9,
                    byte_start: 30,
                    byte_end: 40,
                },
                source_path: "same.rs".into(),
                line_range: (4, 4),
                chunk_id: 0,
                snippet: "same".into(),
                score: 9.0,
                source_layer: ResultSource::LexicalBm25,
                wal_entry_id: 9,
            }],
        ],
        5.0,
    );

    assert_eq!(fused.len(), 1);
    assert_eq!(fused[0].source_path, "same.rs");
    assert_eq!(fused[0].source_layer, ResultSource::Fused);
}

#[test]
fn test_weighted_rrf_basic() {
    let fused = weighted_rrf_fuse(
        vec![
            (
                ResultSource::RawScan,
                0.9,
                vec![ScoredResult {
                    result_id: ResultId {
                        wal_entry_id: 1,
                        byte_start: 0,
                        byte_end: 10,
                    },
                    source_path: "a.rs".into(),
                    line_range: (1, 1),
                    chunk_id: 0,
                    snippet: "alpha".into(),
                    score: 0.0,
                    source_layer: ResultSource::RawScan,
                    wal_entry_id: 1,
                }],
            ),
            (
                ResultSource::LexicalBm25,
                1.0,
                vec![ScoredResult {
                    result_id: ResultId {
                        wal_entry_id: 2,
                        byte_start: 10,
                        byte_end: 20,
                    },
                    source_path: "b.rs".into(),
                    line_range: (2, 2),
                    chunk_id: 0,
                    snippet: "beta".into(),
                    score: 0.0,
                    source_layer: ResultSource::LexicalBm25,
                    wal_entry_id: 2,
                }],
            ),
            (
                ResultSource::HotVector,
                1.0,
                vec![ScoredResult {
                    result_id: ResultId {
                        wal_entry_id: 2,
                        byte_start: 10,
                        byte_end: 20,
                    },
                    source_path: "b.rs".into(),
                    line_range: (2, 2),
                    chunk_id: 0,
                    snippet: "beta but longer".into(),
                    score: 0.0,
                    source_layer: ResultSource::HotVector,
                    wal_entry_id: 2,
                }],
            ),
        ],
        20.0,
    );

    assert_eq!(fused[0].source_path, "b.rs");
    assert_eq!(fused[0].source_layer, ResultSource::Fused);
    assert!(fused[0].score > fused[1].score);
}

#[test]
fn test_rrf_iou_dedup() {
    let fused = weighted_rrf_fuse(
        vec![
            (
                ResultSource::LexicalBm25,
                1.0,
                vec![ScoredResult {
                    result_id: ResultId {
                        wal_entry_id: 7,
                        byte_start: 100,
                        byte_end: 220,
                    },
                    source_path: "same.rs".into(),
                    line_range: (10, 16),
                    chunk_id: 0,
                    snippet: "short".into(),
                    score: 0.0,
                    source_layer: ResultSource::LexicalBm25,
                    wal_entry_id: 7,
                }],
            ),
            (
                ResultSource::HotVector,
                1.0,
                vec![ScoredResult {
                    result_id: ResultId {
                        wal_entry_id: 7,
                        byte_start: 120,
                        byte_end: 240,
                    },
                    source_path: "same.rs".into(),
                    line_range: (11, 17),
                    chunk_id: 0,
                    snippet: "longer overlapping snippet".into(),
                    score: 0.0,
                    source_layer: ResultSource::HotVector,
                    wal_entry_id: 7,
                }],
            ),
        ],
        20.0,
    );

    assert_eq!(fused.len(), 1);
    assert_eq!(fused[0].source_path, "same.rs");
    assert_eq!(fused[0].result_id.wal_entry_id, 7);
    assert_eq!(fused[0].result_id.byte_start, 120);
    assert_eq!(fused[0].result_id.byte_end, 240);
    assert_eq!(fused[0].snippet, "longer overlapping snippet");
}

#[test]
fn test_rrf_no_dedup_different_entries() {
    let fused = weighted_rrf_fuse(
        vec![
            (
                ResultSource::LexicalBm25,
                1.0,
                vec![ScoredResult {
                    result_id: ResultId {
                        wal_entry_id: 7,
                        byte_start: 0,
                        byte_end: 100,
                    },
                    source_path: "left.rs".into(),
                    line_range: (1, 4),
                    chunk_id: 0,
                    snippet: "left".into(),
                    score: 0.0,
                    source_layer: ResultSource::LexicalBm25,
                    wal_entry_id: 7,
                }],
            ),
            (
                ResultSource::HotVector,
                1.0,
                vec![ScoredResult {
                    result_id: ResultId {
                        wal_entry_id: 8,
                        byte_start: 10,
                        byte_end: 110,
                    },
                    source_path: "right.rs".into(),
                    line_range: (1, 4),
                    chunk_id: 0,
                    snippet: "right".into(),
                    score: 0.0,
                    source_layer: ResultSource::HotVector,
                    wal_entry_id: 8,
                }],
            ),
        ],
        20.0,
    );

    assert_eq!(fused.len(), 2);
}

fn scored(path: &str, wal_entry_id: u64, rank_hint: f64, source: ResultSource) -> ScoredResult {
    ScoredResult {
        result_id: ResultId {
            wal_entry_id,
            byte_start: 0,
            byte_end: 32,
        },
        source_path: path.into(),
        line_range: (1, 1),
        chunk_id: 0,
        snippet: path.into(),
        score: rank_hint,
        source_layer: source,
        wal_entry_id,
    }
}

#[test]
fn test_coverage_aware_fusion_complete_layer_dominates() {
    let dense = LayerResults {
        source: ResultSource::HotVector,
        weight: 1.00,
        coverage: CoverageState::Complete,
        results: vec![scored("correct.rs", 1, 10.0, ResultSource::HotVector)],
    };
    let scan = LayerResults {
        source: ResultSource::RawScan,
        weight: 0.90,
        coverage: CoverageState::Complete,
        results: vec![
            scored("wrong-a.rs", 2, 1.0, ResultSource::RawScan),
            scored("wrong-b.rs", 3, 0.9, ResultSource::RawScan),
            scored("wrong-c.rs", 4, 0.8, ResultSource::RawScan),
        ],
    };

    let fused = coverage_aware_rrf_fuse(vec![dense, scan], 20.0);
    assert_eq!(fused[0].source_path, "correct.rs");
}

#[test]
fn test_coverage_aware_fusion_partial_layer_downweighted() {
    let weight = compute_layer_weight(1.0, &CoverageState::Partial(0.3), 0.0);
    assert!((weight - 0.3).abs() < 1e-6);
}

#[test]
fn test_coverage_aware_fusion_unavailable_layer_skipped() {
    let fused = coverage_aware_rrf_fuse(
        vec![
            LayerResults {
                source: ResultSource::HotVector,
                weight: 1.00,
                coverage: CoverageState::Unavailable,
                results: vec![scored("dense.rs", 1, 10.0, ResultSource::HotVector)],
            },
            LayerResults {
                source: ResultSource::RawScan,
                weight: 0.90,
                coverage: CoverageState::Complete,
                results: vec![scored("scan.rs", 2, 1.0, ResultSource::RawScan)],
            },
        ],
        20.0,
    );

    assert_eq!(fused.len(), 1);
    assert_eq!(fused[0].source_path, "scan.rs");
}

#[test]
fn test_coverage_aware_fixes_steady_state_regression() {
    let dense = LayerResults {
        source: ResultSource::HotVector,
        weight: 1.00,
        coverage: CoverageState::Complete,
        results: vec![scored("correct.rs", 1, 20.0, ResultSource::HotVector)],
    };
    let lexical = LayerResults {
        source: ResultSource::LexicalBm25,
        weight: 1.00,
        coverage: CoverageState::Partial(0.4),
        results: vec![
            scored("wrong-1.rs", 2, 5.0, ResultSource::LexicalBm25),
            scored("wrong-2.rs", 3, 4.5, ResultSource::LexicalBm25),
            scored("wrong-3.rs", 4, 4.0, ResultSource::LexicalBm25),
            scored("wrong-4.rs", 5, 3.5, ResultSource::LexicalBm25),
            scored("wrong-5.rs", 6, 3.0, ResultSource::LexicalBm25),
        ],
    };
    let scan = LayerResults {
        source: ResultSource::RawScan,
        weight: 0.90,
        coverage: CoverageState::Complete,
        results: vec![
            scored("wrong-1.rs", 2, 2.0, ResultSource::RawScan),
            scored("wrong-2.rs", 3, 1.9, ResultSource::RawScan),
            scored("wrong-3.rs", 4, 1.8, ResultSource::RawScan),
            scored("wrong-4.rs", 5, 1.7, ResultSource::RawScan),
            scored("wrong-5.rs", 6, 1.6, ResultSource::RawScan),
        ],
    };

    let fused = coverage_aware_rrf_fuse(vec![dense, lexical, scan], 20.0);
    let rank = fused
        .iter()
        .position(|result| result.source_path == "correct.rs")
        .unwrap();
    assert!(rank < 5, "correct result rank was {}", rank + 1);
}

#[test]
fn test_coverage_aware_fusion_t0_scan_dominates() {
    let fused = coverage_aware_rrf_fuse(
        vec![
            LayerResults {
                source: ResultSource::HotVector,
                weight: 1.00,
                coverage: CoverageState::Unavailable,
                results: Vec::new(),
            },
            LayerResults {
                source: ResultSource::RawScan,
                weight: 0.90,
                coverage: CoverageState::Complete,
                results: vec![scored("fresh.rs", 7, 1.0, ResultSource::RawScan)],
            },
        ],
        20.0,
    );
    assert_eq!(fused[0].source_path, "fresh.rs");
}

#[test]
fn test_weighted_rrf_respects_explicit_weights() {
    let fused = weighted_rrf_fuse(
        vec![
            (
                ResultSource::RawScan,
                5.0,
                vec![scored("scan.rs", 1, 1.0, ResultSource::RawScan)],
            ),
            (
                ResultSource::LexicalBm25,
                0.1,
                vec![scored("bm25.rs", 2, 1.0, ResultSource::LexicalBm25)],
            ),
        ],
        20.0,
    );

    assert_eq!(fused[0].source_path, "scan.rs");
    assert!(fused[0].score > fused[1].score);
}
