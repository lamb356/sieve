use sieve_core::fusion::{rrf_fuse, ResultSource, ScoredResult};

#[test]
fn test_rrf_fusion() {
    let scan = vec![
        ScoredResult {
            source_path: "a.rs".into(),
            line_range: (1, 1),
            snippet: "alpha".into(),
            score: 1.0,
            source_layer: ResultSource::Scan,
            wal_entry_id: 1,
        },
        ScoredResult {
            source_path: "b.rs".into(),
            line_range: (1, 1),
            snippet: "beta".into(),
            score: 1.0,
            source_layer: ResultSource::Scan,
            wal_entry_id: 2,
        },
    ];

    let bm25 = vec![
        ScoredResult {
            source_path: "b.rs".into(),
            line_range: (1, 1),
            snippet: "beta".into(),
            score: 10.0,
            source_layer: ResultSource::Bm25,
            wal_entry_id: 2,
        },
        ScoredResult {
            source_path: "c.rs".into(),
            line_range: (1, 1),
            snippet: "gamma".into(),
            score: 9.0,
            source_layer: ResultSource::Bm25,
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
                source_path: "same.rs".into(),
                line_range: (4, 4),
                snippet: "same".into(),
                score: 1.0,
                source_layer: ResultSource::Scan,
                wal_entry_id: 9,
            }],
            vec![ScoredResult {
                source_path: "same.rs".into(),
                line_range: (4, 4),
                snippet: "same".into(),
                score: 9.0,
                source_layer: ResultSource::Bm25,
                wal_entry_id: 9,
            }],
        ],
        5.0,
    );

    assert_eq!(fused.len(), 1);
    assert_eq!(fused[0].source_path, "same.rs");
    assert_eq!(fused[0].source_layer, ResultSource::Fused);
}
