#![cfg(feature = "semantic")]

use sieve_core::aliases::AliasLexicon;
use sieve_core::df_prior::static_df_frac;
use sieve_core::model::ModelManager;
use sieve_core::semantic_query::{
    PhrasePattern, SemanticGroup, SemanticQuery, SemanticTerm, TermSource,
};
use sieve_core::semantic_scan::{compile_scan_query, semantic_scan, MatchEvent, WindowAccumulator};
use sieve_core::surface::{
    realize_surfaces, BoundaryMode, RealizedPattern, SurfaceVariant, VariantKind,
};
use sieve_core::window_score::{compute_idf, score_window};
use sieve_core::{default_sieve_data_dir, plan_query, QueryPlan};

fn sample_query() -> SemanticQuery {
    SemanticQuery {
        raw_query: "failure handling".to_string(),
        normalized_query: "failure handling".to_string(),
        seeds: Vec::new(),
        groups: vec![
            SemanticGroup {
                group_id: 0,
                canonical: "failure".to_string(),
                query_ordinal: 0,
                is_seed: true,
                importance: 1.0,
                member_terms: vec![0],
            },
            SemanticGroup {
                group_id: 1,
                canonical: "handling".to_string(),
                query_ordinal: 1,
                is_seed: true,
                importance: 0.9,
                member_terms: vec![1],
            },
        ],
        terms: vec![
            SemanticTerm {
                term_id: 0,
                vocab_id: 1,
                vocab_piece: "failure".to_string(),
                canonical: "failure".to_string(),
                raw_weight: 2.0,
                norm_weight: 1.0,
                group_id: 0,
                is_anchor: true,
                source: TermSource::OriginalToken,
                surface_variants: vec![SurfaceVariant {
                    text: "failure".to_string(),
                    bytes: b"failure".to_vec(),
                    kind: VariantKind::Lower,
                    boundary: BoundaryMode::Identifier,
                    quality: 0.8,
                }],
            },
            SemanticTerm {
                term_id: 1,
                vocab_id: 2,
                vocab_piece: "handling".to_string(),
                canonical: "handling".to_string(),
                raw_weight: 1.8,
                norm_weight: 0.9,
                group_id: 1,
                is_anchor: true,
                source: TermSource::OriginalToken,
                surface_variants: vec![SurfaceVariant {
                    text: "handling".to_string(),
                    bytes: b"handling".to_vec(),
                    kind: VariantKind::Lower,
                    boundary: BoundaryMode::Identifier,
                    quality: 0.8,
                }],
            },
        ],
        phrases: vec![PhrasePattern {
            phrase_id: 0,
            canonical: "failure handling".to_string(),
            component_groups: vec![0, 1],
            raw_weight: 1.5,
            norm_weight: 0.75,
            is_anchor: true,
            surface_variants: vec![SurfaceVariant {
                text: "failure_handling".to_string(),
                bytes: b"failure_handling".to_vec(),
                kind: VariantKind::Snake,
                boundary: BoundaryMode::Identifier,
                quality: 0.9,
            }],
        }],
        query_order: vec![0, 1],
        total_group_importance: 1.9,
    }
}

#[test]
fn test_df_prior_lookup() {
    assert!(static_df_frac("status") > static_df_frac("qzmtplk"));
}

#[test]
fn test_alias_lexicon_bidirectional() {
    let aliases = AliasLexicon::built_in();
    assert!(aliases.same_alias_family("api", "application programming interface"));
    assert!(aliases.same_alias_family("application programming interface", "api"));
}

#[test]
fn test_surface_realization_rejects_fragments() {
    let mut query = sample_query();
    query.terms.push(SemanticTerm {
        term_id: 2,
        vocab_id: 3,
        vocab_piece: "##ing".to_string(),
        canonical: "##ing".to_string(),
        raw_weight: 1.0,
        norm_weight: 0.7,
        group_id: 1,
        is_anchor: false,
        source: TermSource::SparseExpansion,
        surface_variants: Vec::new(),
    });
    let patterns = realize_surfaces(&mut query, &|_| 0.001);
    assert!(patterns.iter().all(|pattern| pattern.bytes != b"ing"));
    assert!(patterns.iter().all(|pattern| pattern.bytes != b"##ing"));
}

#[test]
fn test_surface_code_variants() {
    let mut query = sample_query();
    let patterns = realize_surfaces(&mut query, &|_| 0.001);
    let pattern_bytes: Vec<Vec<u8>> = patterns.into_iter().map(|pattern| pattern.bytes).collect();
    assert!(pattern_bytes.contains(&b"failure_handling".to_vec()));
    assert!(pattern_bytes.contains(&b"FAILURE_HANDLING".to_vec()));
    assert!(pattern_bytes.contains(&b"failure-handling".to_vec()));
    assert!(pattern_bytes.contains(&b"failureHandling".to_vec()));
    assert!(pattern_bytes.contains(&b"FailureHandling".to_vec()));
}

#[test]
fn test_surface_quality_filter() {
    let mut query = sample_query();
    query.terms[0].norm_weight = 0.1;
    let patterns = realize_surfaces(&mut query, &|_| 0.2);
    assert!(patterns.iter().all(|pattern| pattern.bytes != b"failure"));
}

#[test]
fn test_aho_corasick_compilation() {
    let patterns = vec![RealizedPattern {
        pattern_id: 0,
        term_id: Some(0),
        phrase_id: None,
        primary_group_id: 0,
        bytes: b"failure".to_vec(),
        weight: 1.0,
        is_anchor: true,
        boundary: BoundaryMode::Identifier,
    }];
    let compiled = compile_scan_query(&patterns).unwrap();
    assert_eq!(compiled.patterns.len(), 1);
}

#[test]
fn test_window_accumulator() {
    let patterns = vec![RealizedPattern {
        pattern_id: 0,
        term_id: Some(0),
        phrase_id: None,
        primary_group_id: 0,
        bytes: b"failure".to_vec(),
        weight: 1.0,
        is_anchor: true,
        boundary: BoundaryMode::Identifier,
    }];
    let compiled = compile_scan_query(&patterns).unwrap();
    let query = sample_query();
    let (windows, dfs) = semantic_scan(
        &compiled,
        &[(7, b"failure handling in a module".as_slice())],
        &query,
    );
    assert!(!windows.is_empty());
    assert_eq!(dfs.len(), query.terms.len());
    assert!(windows.iter().any(|window| window.has_anchor));
}

#[test]
fn test_windowed_scoring_basic() {
    let query = sample_query();
    let idf = vec![compute_idf(0, 1, 2, 0.001), compute_idf(1, 1, 2, 0.001)];
    let window = WindowAccumulator {
        wal_entry_id: 7,
        window_start: 0,
        window_end: 64,
        events: vec![
            MatchEvent {
                term_id: Some(0),
                phrase_id: None,
                primary_group_id: 0,
                weight: 1.0,
                byte_start: 0,
                byte_end: 7,
                is_anchor: true,
            },
            MatchEvent {
                term_id: Some(1),
                phrase_id: None,
                primary_group_id: 1,
                weight: 0.9,
                byte_start: 8,
                byte_end: 16,
                is_anchor: true,
            },
        ],
        has_anchor: true,
    };
    assert!(score_window(&window, &query, &idf) > 0.0);
}

#[test]
fn test_windowed_scoring_coverage() {
    let query = sample_query();
    let idf = vec![1.0, 1.0];
    let one_group = WindowAccumulator {
        wal_entry_id: 1,
        window_start: 0,
        window_end: 64,
        events: vec![MatchEvent {
            term_id: Some(0),
            phrase_id: None,
            primary_group_id: 0,
            weight: 1.0,
            byte_start: 0,
            byte_end: 7,
            is_anchor: true,
        }],
        has_anchor: true,
    };
    let two_groups = WindowAccumulator {
        wal_entry_id: 1,
        window_start: 0,
        window_end: 64,
        events: vec![
            MatchEvent {
                term_id: Some(0),
                phrase_id: None,
                primary_group_id: 0,
                weight: 1.0,
                byte_start: 0,
                byte_end: 7,
                is_anchor: true,
            },
            MatchEvent {
                term_id: Some(1),
                phrase_id: None,
                primary_group_id: 1,
                weight: 0.9,
                byte_start: 40,
                byte_end: 48,
                is_anchor: true,
            },
        ],
        has_anchor: true,
    };
    assert!(score_window(&two_groups, &query, &idf) > score_window(&one_group, &query, &idf));
}

#[test]
fn test_windowed_scoring_proximity() {
    let query = sample_query();
    let idf = vec![1.0, 1.0];
    let near = WindowAccumulator {
        wal_entry_id: 1,
        window_start: 0,
        window_end: 64,
        events: vec![
            MatchEvent {
                term_id: Some(0),
                phrase_id: None,
                primary_group_id: 0,
                weight: 1.0,
                byte_start: 0,
                byte_end: 7,
                is_anchor: true,
            },
            MatchEvent {
                term_id: Some(1),
                phrase_id: None,
                primary_group_id: 1,
                weight: 0.9,
                byte_start: 8,
                byte_end: 16,
                is_anchor: true,
            },
        ],
        has_anchor: true,
    };
    let far = WindowAccumulator {
        wal_entry_id: 1,
        window_start: 0,
        window_end: 512,
        events: vec![
            MatchEvent {
                term_id: Some(0),
                phrase_id: None,
                primary_group_id: 0,
                weight: 1.0,
                byte_start: 0,
                byte_end: 7,
                is_anchor: true,
            },
            MatchEvent {
                term_id: Some(1),
                phrase_id: None,
                primary_group_id: 1,
                weight: 0.9,
                byte_start: 300,
                byte_end: 308,
                is_anchor: true,
            },
        ],
        has_anchor: true,
    };
    assert!(score_window(&near, &query, &idf) > score_window(&far, &query, &idf));
}

#[test]
#[ignore]
fn test_query_plan_selects_semantic() {
    let manager = ModelManager::new(&default_sieve_data_dir());
    let sparse = manager.ensure_sparse_model().unwrap();
    let encoder =
        sieve_core::sparse::SpladeEncoder::load(&sparse.model_path, &sparse.tokenizer_path)
            .unwrap();
    let aliases = AliasLexicon::built_in();
    let plan = plan_query("failure handling", Some(&encoder), &aliases);
    assert!(matches!(plan, QueryPlan::Semantic(_)));
}

#[test]
#[ignore]
fn test_semantic_scan_end_to_end() {
    let mut query = sample_query();
    let patterns = realize_surfaces(&mut query, &|term| static_df_frac(term));
    let compiled = compile_scan_query(&patterns).unwrap();
    let entries = vec![(
        11,
        b"module failure_handling path\nretryable error path\n".as_slice(),
    )];
    let (windows, dfs) = semantic_scan(&compiled, &entries, &query);
    assert!(!windows.is_empty());
    assert_eq!(dfs.len(), query.terms.len());
    assert!(windows.iter().any(|window| window.has_anchor));
}

#[test]
#[ignore]
fn test_semantic_scan_zero_preprocess() {
    let mut query = sample_query();
    let patterns = realize_surfaces(&mut query, &|term| static_df_frac(term));
    let compiled = compile_scan_query(&patterns).unwrap();
    let raw = b"\0failure_handling::module\0retry";
    let (windows, _) = semantic_scan(&compiled, &[(9, raw.as_slice())], &query);
    assert!(!windows.is_empty());
    assert!(windows
        .iter()
        .flat_map(|window| window.events.iter())
        .all(|event| event.byte_end as usize <= raw.len()));
}
