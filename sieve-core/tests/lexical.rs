use sieve_core::fusion::ResultSource;
use sieve_core::lexical::{
    build_pending_shards, load_indexed_entries, search_lexical, LexicalMatch,
};
use sieve_core::{Index, SearchOptions};
use tantivy::schema::Value;
use tantivy::{Index as TantivyIndex, TantivyDocument};
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
    assert!(results.iter().any(|result| {
        result.source_path == "split.txt"
            && result.line_number == 2
            && result.snippet.contains("authentication")
    }));
}

#[test]
fn test_exact_phrase_query_preserves_matches_under_semantic_build() {
    let dir = tempdir().unwrap();
    let index = Index::open_or_create(dir.path()).unwrap();
    index
        .add_text(
            "quoted.txt",
            "first line\nauthentication middleware\nthird line\n",
        )
        .unwrap();
    build_pending_shards(&index).unwrap();

    let results = index
        .search(
            "\"authentication middleware\"",
            SearchOptions { top_k: Some(10) },
        )
        .unwrap();

    assert!(results.iter().any(|result| {
        result.source_path == "quoted.txt"
            && result.line_number == 2
            && result.snippet.contains("authentication middleware")
    }));
}

#[test]
fn test_lexical_line_numbers_remain_absolute_after_chunking() {
    let dir = tempdir().unwrap();
    let index = Index::open_or_create(dir.path()).unwrap();
    let mut lines: Vec<String> = (1..=220).map(|i| format!("line-{i}")).collect();
    lines[179] = "semantic target lives here".to_string();
    index
        .add_text("absolute.txt", format!("{}\n", lines.join("\n")))
        .unwrap();
    build_pending_shards(&index).unwrap();

    let results = index
        .search("target lives", SearchOptions { top_k: Some(10) })
        .unwrap();

    assert!(results.iter().any(|result| {
        result.source_path == "absolute.txt"
            && result.line_number == 180
            && result.line_range == (180, 180)
    }));
}

#[test]
fn test_symbol_query_uses_tantivy_identifier_fallback() {
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
    assert_ne!(results[0].source_layer, ResultSource::ScanFallback);
    assert_ne!(results[0].source_layer, ResultSource::RawScan);
}

#[test]
fn test_tantivy_chunk_indexing() {
    let dir = tempdir().unwrap();
    let index = Index::open_or_create(dir.path()).unwrap();
    let content = ["alpha", &"x".repeat(600), "omega"].join("\n");
    index.add_text("src/chunky.rs", content).unwrap();

    build_pending_shards(&index).unwrap();

    let shard = dir.path().join("segments/seg_0001");
    let tantivy = TantivyIndex::open_in_dir(&shard).unwrap();
    let reader = tantivy.reader().unwrap();
    let searcher = reader.searcher();
    let schema = tantivy.schema();
    let source_path = schema.get_field("source_path").unwrap();
    let wal_entry_id = schema.get_field("wal_entry_id").unwrap();
    let chunk_id = schema.get_field("chunk_id").unwrap();
    let byte_start = schema.get_field("byte_start").unwrap();
    let byte_end = schema.get_field("byte_end").unwrap();
    let stored_text = schema.get_field("stored_text").unwrap();

    let mut docs: Vec<_> = searcher
        .segment_readers()
        .iter()
        .flat_map(|segment| {
            let store = segment.get_store_reader(10).unwrap();
            (0..segment.max_doc())
                .map(move |doc_id| store.get::<TantivyDocument>(doc_id).unwrap())
                .collect::<Vec<_>>()
        })
        .collect();
    docs.sort_by_key(|doc| {
        doc.get_first(chunk_id)
            .and_then(|v| v.as_u64())
            .unwrap_or_default()
    });

    assert!(docs.len() > 1);
    for (idx, doc) in docs.iter().enumerate() {
        assert_eq!(
            doc.get_first(source_path).and_then(|v| v.as_str()),
            Some("src/chunky.rs")
        );
        assert_eq!(
            doc.get_first(wal_entry_id).and_then(|v| v.as_u64()),
            Some(0)
        );
        assert_eq!(
            doc.get_first(chunk_id).and_then(|v| v.as_u64()),
            Some(idx as u64)
        );
        assert!(doc.get_first(byte_end).and_then(|v| v.as_u64()).unwrap() > 0);
        assert!(
            doc.get_first(byte_end).and_then(|v| v.as_u64()).unwrap()
                > doc.get_first(byte_start).and_then(|v| v.as_u64()).unwrap()
        );
        assert!(!doc
            .get_first(stored_text)
            .and_then(|v| v.as_str())
            .unwrap()
            .is_empty());
    }
}

#[test]
fn test_tantivy_ident_field() {
    let dir = tempdir().unwrap();
    let index = Index::open_or_create(dir.path()).unwrap();
    index
        .add_text("src/symbols.rs", "fn foo::bar_baz() {}\n")
        .unwrap();
    build_pending_shards(&index).unwrap();

    let matches = search_lexical(&dir.path().join("segments"), "foo::bar_baz", 10).unwrap();
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].source_path, "src/symbols.rs");
    assert!(matches[0].snippet.contains("foo::bar_baz"));
}

#[test]
fn test_merge_small_shards_preserves_ident_field() {
    let dir = tempdir().unwrap();
    let index = Index::open_or_create(dir.path()).unwrap();
    for i in 0..11 {
        index
            .add_text(
                format!("src/{i}.rs"),
                format!("fn foo::bar_baz_{i}() {{}}\n"),
            )
            .unwrap();
        build_pending_shards(&index).unwrap();
    }

    let matches = search_lexical(&dir.path().join("segments"), "foo::bar_baz_10", 10).unwrap();
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].source_path, "src/10.rs");
    assert_eq!(matches[0].chunk_id, 0);
}
