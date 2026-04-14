#![cfg(feature = "semantic")]

use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use serde_json::Value;
use sieve_core::default_queries::DEFAULT_TRAINING_QUERIES;
use sieve_core::event_rerank::{
    extract_rerank_input, should_rerank_event_windows, EVENT_FEATS, GLOBAL_FEATS, MAX_EVENT_TOKENS,
};
use sieve_core::fusion::ResultSource;
use sieve_core::lexical::{
    anchor_boost, build_semantic_tantivy_query, expansion_boost, phrase_boost,
    search_semantic_lexical, semantic_tantivy_clauses, TantivyFieldKind,
};
use sieve_core::semantic_query::{
    GroupId, PhrasePattern, SemanticGroup, SemanticQuery, SemanticTerm, TermSource,
};
use sieve_core::surface::{BoundaryMode, SurfaceVariant, VariantKind};
use sieve_core::training_export::export_training_data;
use sieve_core::{Index, SearchOptions, SearchSnapshot};
use tempfile::tempdir;

fn home_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn set_home(path: &Path) {
    std::env::set_var("HOME", path);
}

fn term(
    term_id: u16,
    canonical: &str,
    group_id: GroupId,
    norm_weight: f32,
    is_anchor: bool,
    variants: &[&str],
) -> SemanticTerm {
    SemanticTerm {
        term_id,
        vocab_id: term_id as u32,
        vocab_piece: canonical.to_string(),
        canonical: canonical.to_string(),
        raw_weight: norm_weight,
        norm_weight,
        group_id,
        is_anchor,
        source: if is_anchor {
            TermSource::OriginalToken
        } else {
            TermSource::SparseExpansion
        },
        surface_variants: variants
            .iter()
            .map(|variant| SurfaceVariant {
                text: (*variant).to_string(),
                bytes: variant.as_bytes().to_vec(),
                kind: if variant.contains('_') {
                    VariantKind::Snake
                } else {
                    VariantKind::Lower
                },
                boundary: BoundaryMode::Identifier,
                quality: 0.9,
            })
            .collect(),
    }
}

fn phrase(phrase_id: u16, canonical: &str, groups: &[GroupId], norm_weight: f32) -> PhrasePattern {
    PhrasePattern {
        phrase_id,
        canonical: canonical.to_string(),
        component_groups: groups.to_vec(),
        raw_weight: norm_weight,
        norm_weight,
        is_anchor: true,
        surface_variants: vec![SurfaceVariant {
            text: canonical.to_string(),
            bytes: canonical.as_bytes().to_vec(),
            kind: VariantKind::Lower,
            boundary: BoundaryMode::Word,
            quality: 1.0,
        }],
    }
}

fn mock_semantic_query() -> SemanticQuery {
    SemanticQuery {
        raw_query: "error handling".to_string(),
        normalized_query: "error handling".to_string(),
        seeds: Vec::new(),
        groups: vec![
            SemanticGroup {
                group_id: 0,
                canonical: "error".to_string(),
                query_ordinal: 0,
                is_seed: true,
                importance: 1.0,
                member_terms: vec![0, 2, 3],
            },
            SemanticGroup {
                group_id: 1,
                canonical: "handling".to_string(),
                query_ordinal: 1,
                is_seed: true,
                importance: 0.9,
                member_terms: vec![1, 4],
            },
        ],
        terms: vec![
            term(0, "error", 0, 1.0, true, &["error"]),
            term(1, "handling", 1, 0.95, true, &["handling"]),
            term(2, "retry", 0, 0.72, false, &["retry"]),
            term(3, "exception", 0, 0.61, false, &["exception"]),
            term(
                4,
                "graceful_degradation",
                1,
                0.58,
                false,
                &["graceful_degradation"],
            ),
        ],
        phrases: vec![phrase(0, "error handling", &[0, 1], 0.82)],
        query_order: vec![0, 1],
        total_group_importance: 1.9,
    }
}

#[test]
fn test_splade_tantivy_query_construction() {
    let query = mock_semantic_query();
    let clauses = semantic_tantivy_clauses(&query);
    assert!(clauses.iter().any(|clause| clause.is_anchor));
    assert!(clauses.iter().any(|clause| !clause.is_anchor));
    assert!(clauses.iter().any(|clause| clause.is_phrase));
    assert!(clauses
        .iter()
        .any(|clause| clause.field == TantivyFieldKind::Body));
    assert!(clauses
        .iter()
        .any(|clause| clause.field == TantivyFieldKind::Ident));

    let schema = sieve_core::lexical::lexical_schema();
    let built = build_semantic_tantivy_query(&query, &schema).unwrap();
    let debug = format!("{built:?}");
    assert!(debug.contains("Must") || debug.contains("must"));
    assert!(debug.contains("Should") || debug.contains("should"));
}

#[test]
fn test_splade_tantivy_boost_values() {
    assert!(anchor_boost(0.8) > expansion_boost(0.8));
    assert!(phrase_boost(0.8) > anchor_boost(0.8));
}

#[test]
fn test_splade_tantivy_finds_expanded_terms() {
    let dir = tempdir().unwrap();
    let index = Index::open_or_create(dir.path()).unwrap();
    index
        .add_text(
            "expanded.rs",
            "error retry with exception mapping keeps request flow alive\n",
        )
        .unwrap();
    sieve_core::lexical::build_pending_shards(&index).unwrap();

    let matches =
        search_semantic_lexical(&dir.path().join("segments"), &mock_semantic_query(), 10).unwrap();
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].source_path, "expanded.rs");
}

#[test]
fn test_splade_tantivy_anchor_required() {
    let dir = tempdir().unwrap();
    let index = Index::open_or_create(dir.path()).unwrap();
    index
        .add_text("noise.rs", "retry exception graceful_degradation\n")
        .unwrap();
    sieve_core::lexical::build_pending_shards(&index).unwrap();

    let matches =
        search_semantic_lexical(&dir.path().join("segments"), &mock_semantic_query(), 10).unwrap();
    assert!(matches.is_empty());
}

#[test]
fn test_disjoint_partition() {
    let dir = tempdir().unwrap();
    let index = Index::open_or_create(dir.path()).unwrap();
    for i in 0..5 {
        index
            .add_text(format!("src/{i}.rs"), format!("error handling case {i}\n"))
            .unwrap();
    }
    sieve_core::lexical::build_pending_shards(&index).unwrap();
    index
        .add_text("src/fresh-a.rs", "error handling fresh a\n")
        .unwrap();
    index
        .add_text("src/fresh-b.rs", "error handling fresh b\n")
        .unwrap();

    let snapshot: SearchSnapshot = index.snapshot_search_partition().unwrap();
    assert_eq!(snapshot.indexed_ids.len(), 5);
    assert_eq!(snapshot.fresh_ids.len(), 2);
    assert_eq!(snapshot.active_ids.len(), 7);
    assert_eq!(
        snapshot.indexed_ids.intersection_len(&snapshot.fresh_ids),
        0
    );
}

#[test]
fn test_dual_backend_both_contribute() {
    let dir = tempdir().unwrap();
    let _guard = home_lock().lock().unwrap_or_else(|e| e.into_inner());
    set_home(dir.path());
    let index = Index::open_or_create(dir.path()).unwrap();
    index
        .add_text(
            "stable.rs",
            "error handling with retry and exception routing\n",
        )
        .unwrap();
    sieve_core::lexical::build_pending_shards(&index).unwrap();
    index
        .add_text(
            "fresh.rs",
            "error handling stays fresh via graceful_degradation fallback\n",
        )
        .unwrap();

    let outcome = index
        .search_semantic_query(
            &mock_semantic_query(),
            SearchOptions {
                top_k: Some(10),
                ..Default::default()
            },
        )
        .unwrap();
    let layers: HashSet<_> = outcome.source_sets.iter().map(|set| set.source).collect();
    assert!(layers.contains(&ResultSource::SemanticScan));
    assert!(layers.contains(&ResultSource::SpladeBm25));
}

#[test]
fn test_fresh_only_semantic_query_skips_stable_layers() {
    let dir = tempdir().unwrap();
    let _guard = home_lock().lock().unwrap_or_else(|e| e.into_inner());
    set_home(dir.path());
    let index = Index::open_or_create(dir.path()).unwrap();
    index
        .add_text(
            "stable.rs",
            "error handling with retry and exception routing\n",
        )
        .unwrap();
    sieve_core::lexical::build_pending_shards(&index).unwrap();
    index
        .add_text(
            "fresh.rs",
            "error handling stays fresh via graceful_degradation fallback\n",
        )
        .unwrap();

    let outcome = index
        .search_semantic_query(
            &mock_semantic_query(),
            SearchOptions {
                top_k: Some(10),
                fresh_only: true,
                ..Default::default()
            },
        )
        .unwrap();
    assert!(outcome
        .debug
        .as_ref()
        .is_some_and(|debug| debug.plan_mode == "semantic:fresh-only"));
    assert!(outcome
        .source_sets
        .iter()
        .all(|set| matches!(set.source, ResultSource::SemanticScan)));
    assert!(!outcome.results.is_empty());
    assert!(outcome
        .results
        .iter()
        .all(|result| result.source_path == "fresh.rs"));
    assert!(outcome.results.iter().all(|result| {
        matches!(
            result.source_layer,
            ResultSource::RawScan | ResultSource::SemanticScan | ResultSource::Fused
        )
    }));
}

#[test]
fn test_semantic_query_keeps_indexed_and_fresh_sources_separate() {
    let dir = tempdir().unwrap();
    let _guard = home_lock().lock().unwrap_or_else(|e| e.into_inner());
    set_home(dir.path());
    let index = Index::open_or_create(dir.path()).unwrap();
    index
        .add_text(
            "stable.rs",
            "error handling with retry and exception routing\n",
        )
        .unwrap();
    sieve_core::lexical::build_pending_shards(&index).unwrap();
    index
        .add_text(
            "fresh.rs",
            "error handling stays fresh via graceful_degradation fallback\n",
        )
        .unwrap();

    let outcome = index
        .search_semantic_query(
            &mock_semantic_query(),
            SearchOptions {
                top_k: Some(10),
                ..Default::default()
            },
        )
        .unwrap();
    let sources: HashSet<_> = outcome.source_sets.iter().map(|set| set.source).collect();
    assert!(sources.contains(&ResultSource::SpladeBm25));
    assert!(sources.contains(&ResultSource::SemanticScan));
}

#[test]
fn test_event_reranker_disabled_without_flag() {
    let dir = tempdir().unwrap();
    let _guard = home_lock().lock().unwrap_or_else(|e| e.into_inner());
    set_home(dir.path());
    let index = Index::open_or_create(dir.path()).unwrap();
    index
        .add_text(
            "stable.rs",
            "error handling with retry and exception routing\n",
        )
        .unwrap();
    sieve_core::lexical::build_pending_shards(&index).unwrap();
    index
        .add_text(
            "fresh.rs",
            "error handling stays fresh via graceful_degradation fallback\n",
        )
        .unwrap();

    let outcome = index
        .search_semantic_query(
            &mock_semantic_query(),
            SearchOptions {
                top_k: Some(32),
                experimental_rerank: false,
                ..Default::default()
            },
        )
        .unwrap();
    assert!(outcome
        .source_sets
        .iter()
        .all(|set| set.source != ResultSource::EventReranked));
}

#[test]
fn test_training_export_format() {
    let dir = tempdir().unwrap();
    let index = Index::open_or_create(dir.path()).unwrap();
    index
        .add_text("a.rs", "error handling with retry and exception mapping\n")
        .unwrap();
    let output = dir.path().join("training.jsonl");
    let exported = export_training_data(
        &index,
        &[
            "error handling".to_string(),
            "retry with backoff".to_string(),
            "graceful shutdown".to_string(),
        ],
        &output,
        8,
    )
    .unwrap();
    assert!(exported > 0);
    let lines: Vec<_> = fs::read_to_string(&output)
        .unwrap()
        .lines()
        .map(str::to_string)
        .collect();
    assert!(!lines.is_empty());
    for line in lines {
        let json: Value = serde_json::from_str(&line).unwrap();
        assert!(json.get("query").is_some());
        let window = json.get("window").unwrap();
        assert!(window.get("events").is_some());
        assert!(window.get("global_features").is_some());
        assert!(window.get("raw_text").is_some());
    }
}

#[test]
fn test_training_export_excludes_deleted_entries() {
    let dir = tempdir().unwrap();
    let index = Index::open_or_create(dir.path()).unwrap();
    index
        .add_text("stale.rs", "error handling in deleted file\n")
        .unwrap();
    index
        .add_text("live.rs", "error handling in live file\n")
        .unwrap();
    let keep: HashSet<String> = ["live.rs".to_string()].into_iter().collect();
    index.prune_manifest_to_paths(&keep).unwrap();
    index.save_manifest().unwrap();

    let output = dir.path().join("training.jsonl");
    export_training_data(&index, &["error handling".to_string()], &output, 8).unwrap();
    let lines = fs::read_to_string(&output).unwrap();
    assert!(lines.contains("live file"));
    assert!(!lines.contains("deleted file"));
}

#[test]
fn test_training_export_event_features() {
    let dir = tempdir().unwrap();
    let index = Index::open_or_create(dir.path()).unwrap();
    index
        .add_text(
            "a.rs",
            "error handling retry exception graceful_degradation\n",
        )
        .unwrap();
    let output = dir.path().join("training.jsonl");
    export_training_data(&index, &["error handling".to_string()], &output, 4).unwrap();
    let line = fs::read_to_string(&output)
        .unwrap()
        .lines()
        .next()
        .unwrap()
        .to_string();
    let json: Value = serde_json::from_str(&line).unwrap();
    let events = json["window"]["events"].as_array().unwrap();
    let global = json["window"]["global_features"].as_object().unwrap();
    assert!(!events.is_empty());
    for event in events {
        assert_eq!(event.as_object().unwrap().len(), EVENT_FEATS);
    }
    assert_eq!(global.len(), GLOBAL_FEATS);
}

#[test]
fn test_event_reranker_feature_extraction() {
    let query = mock_semantic_query();
    let window = sieve_core::semantic_scan::WindowAccumulator {
        wal_entry_id: 1,
        window_start: 0,
        window_end: 64,
        events: vec![
            sieve_core::semantic_scan::MatchEvent {
                term_id: Some(0),
                phrase_id: None,
                primary_group_id: 0,
                weight: 1.0,
                byte_start: 0,
                byte_end: 5,
                is_anchor: true,
            },
            sieve_core::semantic_scan::MatchEvent {
                term_id: Some(1),
                phrase_id: None,
                primary_group_id: 1,
                weight: 0.95,
                byte_start: 10,
                byte_end: 18,
                is_anchor: true,
            },
        ],
        has_anchor: true,
    };
    let input = extract_rerank_input(&window, &query, &[1.0; 8], 2.75);
    assert_eq!(input.events.len(), MAX_EVENT_TOKENS);
    assert_eq!(input.events[0].len(), EVENT_FEATS);
    assert_eq!(input.global.len(), GLOBAL_FEATS);
    assert_eq!(input.mask[0], 1);
    assert_eq!(input.mask[1], 1);
    assert_eq!(input.mask[2], 0);
}

#[test]
fn test_event_reranker_graceful_skip() {
    let dir = tempdir().unwrap();
    let _guard = home_lock().lock().unwrap_or_else(|e| e.into_inner());
    set_home(dir.path());
    let index = Index::open_or_create(dir.path()).unwrap();
    index
        .add_text("fallback.rs", "error handling with retry and exception\n")
        .unwrap();
    let results = index
        .search_semantic_query(
            &mock_semantic_query(),
            SearchOptions {
                top_k: Some(10),
                ..Default::default()
            },
        )
        .unwrap()
        .results;
    assert!(!results.is_empty());
    assert!(results
        .iter()
        .all(|result| result.source_layer != ResultSource::EventReranked));
}

#[test]
fn test_event_reranker_criteria() {
    let single_group = SemanticQuery {
        groups: vec![SemanticGroup {
            group_id: 0,
            canonical: "error".to_string(),
            query_ordinal: 0,
            is_seed: true,
            importance: 1.0,
            member_terms: vec![0],
        }],
        ..mock_semantic_query()
    };
    assert!(!should_rerank_event_windows(&single_group, 16, true));
    assert!(should_rerank_event_windows(
        &mock_semantic_query(),
        16,
        true
    ));
}

#[test]
fn test_search_without_any_models_still_works() {
    let dir = tempdir().unwrap();
    let _guard = home_lock().lock().unwrap_or_else(|e| e.into_inner());
    set_home(dir.path());
    let index = Index::open_or_create(dir.path()).unwrap();
    index
        .add_text("plain.rs", "fn error_handling() { retry(); }\n")
        .unwrap();
    sieve_core::lexical::build_pending_shards(&index).unwrap();

    let results = index
        .search("error_handling", SearchOptions { top_k: Some(10), ..Default::default() })
        .unwrap();
    assert!(!results.is_empty());
    assert!(results.iter().all(|result| {
        matches!(
            result.source_layer,
            ResultSource::RawScan | ResultSource::LexicalBm25 | ResultSource::Fused
        )
    }));
}

#[test]
fn test_default_training_queries_nonempty() {
    assert_eq!(DEFAULT_TRAINING_QUERIES.len(), 50);
    let deduped: HashSet<_> = DEFAULT_TRAINING_QUERIES.iter().copied().collect();
    assert_eq!(deduped.len(), DEFAULT_TRAINING_QUERIES.len());
}
