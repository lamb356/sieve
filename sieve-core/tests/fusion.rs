use sieve_core::fusion::{rrf_fuse, weighted_rrf_fuse, ResultId, ResultSource, ScoredResult};

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
