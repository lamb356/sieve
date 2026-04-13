use sieve_core::lexical::{
    build_pending_shards, load_indexed_entries, search_lexical, LexicalMatch,
};
use sieve_core::{Index, SearchOptions};
use tempfile::tempdir;

fn source_paths(matches: &[LexicalMatch]) -> Vec<String> {
    matches.iter().map(|m| m.source_path.clone()).collect()
}

#[test]
fn test_tantivy_shard_create_and_query() {
    let dir = tempdir().unwrap();
    let index = Index::open_or_create(dir.path()).unwrap();
    index
        .add_text("src/a.rs", "fn authentication_middleware() {}\n")
        .unwrap();
    index
        .add_text("src/b.rs", "fn login_handler() {}\n")
        .unwrap();

    let built = build_pending_shards(&index).unwrap();
    assert_eq!(built, 1);

    let matches = search_lexical(&dir.path().join("segments"), "authentication", 10).unwrap();
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].source_path, "src/a.rs");
    assert!(matches[0].snippet.contains("authentication_middleware"));
    assert!(matches[0].bm25_score > 0.0);
}

#[test]
fn test_shard_persistence() {
    let dir = tempdir().unwrap();
    {
        let index = Index::open_or_create(dir.path()).unwrap();
        index
            .add_text("src/a.rs", "fn authentication_middleware() {}\n")
            .unwrap();
        build_pending_shards(&index).unwrap();
    }

    let matches = search_lexical(&dir.path().join("segments"), "authentication", 10).unwrap();
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].source_path, "src/a.rs");
}

#[test]
fn test_hybrid_search() {
    let dir = tempdir().unwrap();
    let index = Index::open_or_create(dir.path()).unwrap();
    index.add_text("old.rs", "TODO from shard\n").unwrap();
    build_pending_shards(&index).unwrap();

    index.add_text("new.rs", "TODO from wal only\n").unwrap();

    let results = index
        .search("TODO", SearchOptions { top_k: Some(10) })
        .unwrap();
    let paths: Vec<_> = results.iter().map(|r| r.source_path.as_str()).collect();
    assert!(paths.contains(&"old.rs"));
    assert!(paths.contains(&"new.rs"));
}

#[test]
fn test_incremental_index() {
    let dir = tempdir().unwrap();
    let index = Index::open_or_create(dir.path()).unwrap();
    index.add_text("a.rs", "alpha lexical term\n").unwrap();
    let built = build_pending_shards(&index).unwrap();
    assert_eq!(built, 1);
    assert_eq!(
        load_indexed_entries(&dir.path().join("segments"))
            .unwrap()
            .len(),
        1
    );

    index.add_text("b.rs", "beta lexical term\n").unwrap();
    let built = build_pending_shards(&index).unwrap();
    assert_eq!(built, 1);

    let indexed = load_indexed_entries(&dir.path().join("segments")).unwrap();
    assert_eq!(indexed.len(), 2);

    let matches = search_lexical(&dir.path().join("segments"), "beta", 10).unwrap();
    assert_eq!(source_paths(&matches), vec!["b.rs".to_string()]);
}

#[test]
fn test_regex_search() {
    let dir = tempdir().unwrap();
    let index = Index::open_or_create(dir.path()).unwrap();
    index.add_text("a.rs", "TODO item\n").unwrap();
    build_pending_shards(&index).unwrap();

    let results = index
        .search("/TODO|FIXME/", SearchOptions { top_k: Some(10) })
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].source_path, "a.rs");
    assert_eq!(results[0].source_layer.as_str(), "scan");
}

#[test]
fn test_lexical_only_result_localizes_line_and_snippet() {
    let dir = tempdir().unwrap();
    let index = Index::open_or_create(dir.path()).unwrap();
    index
        .add_text("split.txt", "line1\nauthentication\nmiddleware\nline4\n")
        .unwrap();
    build_pending_shards(&index).unwrap();

    let results = index
        .search(
            "authentication middleware",
            SearchOptions { top_k: Some(10) },
        )
        .unwrap();

    assert!(!results.is_empty());
    assert_eq!(results[0].source_path, "split.txt");
    assert_eq!(results[0].line_number, 2);
    assert!(results[0].snippet.contains("authentication"));
}

#[test]
fn test_symbol_query_shows_fallback_tag() {
    let dir = tempdir().unwrap();
    let index = Index::open_or_create(dir.path()).unwrap();
    index.add_text("symbols.rs", "fn foo::bar() {}\n").unwrap();
    build_pending_shards(&index).unwrap();

    let results = index
        .search("foo::bar", SearchOptions { top_k: Some(10) })
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].source_path, "symbols.rs");
    assert!(results[0].snippet.contains("foo::bar"));
    assert_eq!(results[0].source_layer.as_str(), "scan:fallback");
}
