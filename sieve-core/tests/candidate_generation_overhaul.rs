#![cfg(feature = "semantic")]

use sieve_core::aliases::AliasLexicon;
use sieve_core::lexical::{search_semantic_lexical, semantic_tantivy_clauses, TantivyFieldKind};
use sieve_core::model::{select_sparse_route, SparseRoute};
use sieve_core::semantic_query::{
    compile_semantic_query_with, detect_content_type, tokenize_query, ContentType,
    SemanticCompileOptions,
};
use sieve_core::semantic_scan::{compile_scan_query, compute_windows, semantic_scan};
use sieve_core::surface::{realize_code_subtoken, realize_surfaces, BoundaryMode, RealizedPattern};
use sieve_core::Index;
use tempfile::tempdir;

fn mock_vocab(id: u32) -> Option<String> {
    Some(
        match id {
            1 => "error",
            2 => "handler",
            3 => "server",
            4 => "http",
            5 => "retry",
            _ => return None,
        }
        .to_string(),
    )
}

#[test]
fn test_detect_content_type() {
    assert_eq!(
        detect_content_type(std::path::Path::new("src/lib.rs")),
        ContentType::Code
    );
    assert_eq!(
        detect_content_type(std::path::Path::new("README.md")),
        ContentType::Prose
    );
    assert_eq!(
        detect_content_type(std::path::Path::new("Cargo.toml")),
        ContentType::Config
    );
    assert_eq!(
        detect_content_type(std::path::Path::new("app.log")),
        ContentType::Log
    );
    assert_eq!(
        detect_content_type(std::path::Path::new("weird.xyz")),
        ContentType::Unknown
    );
}

#[test]
fn test_code_subtoken_matches_inside_camel_case() {
    let patterns = vec![RealizedPattern {
        pattern_id: 0,
        term_id: Some(0),
        phrase_id: None,
        primary_group_id: 0,
        bytes: b"Error".to_vec(),
        weight: 1.0,
        is_anchor: true,
        boundary: BoundaryMode::CodeSubtoken,
    }];
    let compiled = compile_scan_query(&patterns).unwrap();
    let query = sieve_core::semantic_query::SemanticQuery::seed_only("error", ContentType::Code);
    let (windows, _) = semantic_scan(
        &compiled,
        &[(
            1,
            b"fn handleErrorGracefully()".as_slice(),
            ContentType::Code,
        )],
        &query,
        8,
    );
    assert!(!windows.is_empty());
}

#[test]
fn test_code_subtoken_matches_inside_snake_case() {
    let patterns = vec![RealizedPattern {
        pattern_id: 0,
        term_id: Some(0),
        phrase_id: None,
        primary_group_id: 0,
        bytes: b"error".to_vec(),
        weight: 1.0,
        is_anchor: true,
        boundary: BoundaryMode::CodeSubtoken,
    }];
    let compiled = compile_scan_query(&patterns).unwrap();
    let query = sieve_core::semantic_query::SemanticQuery::seed_only("error", ContentType::Code);
    let (windows, _) = semantic_scan(
        &compiled,
        &[(
            1,
            b"fn handle_error_gracefully()".as_slice(),
            ContentType::Code,
        )],
        &query,
        8,
    );
    assert!(!windows.is_empty());
}

#[test]
fn test_code_subtoken_matches_inside_namespace() {
    let patterns = vec![RealizedPattern {
        pattern_id: 0,
        term_id: Some(0),
        phrase_id: None,
        primary_group_id: 0,
        bytes: b"error".to_vec(),
        weight: 1.0,
        is_anchor: true,
        boundary: BoundaryMode::CodeSubtoken,
    }];
    let compiled = compile_scan_query(&patterns).unwrap();
    let query = sieve_core::semantic_query::SemanticQuery::seed_only("error", ContentType::Code);
    let (windows, _) = semantic_scan(
        &compiled,
        &[(1, b"use std::error::Error".as_slice(), ContentType::Code)],
        &query,
        8,
    );
    assert!(!windows.is_empty());
}

#[test]
fn test_code_subtoken_does_not_false_positive_on_prose() {
    let patterns = vec![RealizedPattern {
        pattern_id: 0,
        term_id: Some(0),
        phrase_id: None,
        primary_group_id: 0,
        bytes: b"error".to_vec(),
        weight: 1.0,
        is_anchor: true,
        boundary: BoundaryMode::CodeSubtoken,
    }];
    let compiled = compile_scan_query(&patterns).unwrap();
    let query = sieve_core::semantic_query::SemanticQuery::seed_only("error", ContentType::Prose);
    let (windows, _) = semantic_scan(
        &compiled,
        &[(1, b"the terrorism report".as_slice(), ContentType::Prose)],
        &query,
        8,
    );
    assert!(windows.is_empty());
}

#[test]
fn test_surface_realizer_emits_initial_cap_for_code() {
    let variants = realize_code_subtoken("error");
    assert!(variants.iter().any(|variant| variant == "Error"));
}

#[test]
fn test_code_query_keeps_short_tokens() {
    let tokens = tokenize_query("read fs io", ContentType::Code);
    let texts: Vec<_> = tokens.iter().map(|token| token.text.as_str()).collect();
    assert!(texts.contains(&"fs"));
    assert!(texts.contains(&"io"));
}

#[test]
fn test_code_query_keeps_namespace() {
    let tokens = tokenize_query("use std::fs", ContentType::Code);
    let texts: Vec<_> = tokens.iter().map(|token| token.text.as_str()).collect();
    assert!(texts.contains(&"std::fs"));
    assert!(texts.contains(&"std"));
    assert!(texts.contains(&"fs"));
}

#[test]
fn test_code_query_keeps_numbers() {
    let tokens = tokenize_query("handle 429 retry", ContentType::Code);
    let texts: Vec<_> = tokens.iter().map(|token| token.text.as_str()).collect();
    assert!(texts.contains(&"429"));
}

#[test]
fn test_code_query_splits_compound() {
    let tokens = tokenize_query("error_handler", ContentType::Code);
    let texts: Vec<_> = tokens.iter().map(|token| token.text.as_str()).collect();
    assert!(texts.contains(&"error_handler"));
    assert!(texts.contains(&"error"));
    assert!(texts.contains(&"handler"));
}

#[test]
fn test_prose_query_drops_single_char() {
    let tokens = tokenize_query("a big error", ContentType::Prose);
    let texts: Vec<_> = tokens.iter().map(|token| token.text.as_str()).collect();
    assert!(!texts.contains(&"a"));
    assert!(texts.contains(&"big"));
    assert!(texts.contains(&"error"));
}

#[test]
fn test_content_type_routes_to_correct_model() {
    let code_route = select_sparse_route(ContentType::Code, true, true);
    assert_eq!(code_route.route, SparseRoute::CodeSplade);
    let prose_route = select_sparse_route(ContentType::Prose, true, true);
    assert_eq!(prose_route.route, SparseRoute::GenericSplade);
}

#[test]
fn test_splade_code_fallback_to_generic() {
    let route = select_sparse_route(ContentType::Code, false, true);
    assert_eq!(route.route, SparseRoute::GenericSplade);
    assert!(route.warned_fallback);
}

#[test]
fn test_subtoken_field_indexes_camel_case() {
    let dir = tempdir().unwrap();
    let index = Index::open_or_create(dir.path()).unwrap();
    index
        .add_text("src/error.rs", "struct ErrorHandler;\n")
        .unwrap();
    sieve_core::lexical::build_pending_shards(&index).unwrap();

    let aliases = AliasLexicon::built_in();
    let mut query = compile_semantic_query_with(
        "error handling",
        &|_| Ok(vec![(1, 2.0), (2, 1.5)]),
        &mock_vocab,
        &aliases,
        SemanticCompileOptions::default(),
        ContentType::Code,
    )
    .unwrap();
    let _patterns = realize_surfaces(&mut query, &|_| 0.001);

    let clauses = semantic_tantivy_clauses(&query);
    assert!(clauses
        .iter()
        .any(|clause| clause.field == TantivyFieldKind::Subtoken));

    let matches = search_semantic_lexical(&dir.path().join("segments"), &query, 10).unwrap();
    assert!(matches.iter().any(|m| m.source_path == "src/error.rs"));
}

#[test]
fn test_code_windows_split_on_functions() {
    let content = format!(
        "def one():\n{}\n\ndef two():\n{}\n",
        "    return 1\n".repeat(16),
        "    return 2\n".repeat(16)
    );
    let windows = compute_windows(content.as_bytes(), ContentType::Code);
    assert!(windows.len() >= 2);
}

#[test]
fn test_prose_windows_use_paragraphs() {
    let content = b"alpha paragraph\nline two\n\nnext paragraph\n";
    let windows = compute_windows(content, ContentType::Prose);
    assert!(windows.len() >= 2);
}
