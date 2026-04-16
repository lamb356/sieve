use std::fs::OpenOptions;
use std::io::Write;

use sieve_core::{Index, QueryPromotedDense, SearchOptions};
use tempfile::tempdir;

#[test]
fn add_text_is_immediately_searchable_via_raw_scan() {
    let tmp = tempdir().unwrap();
    let index = Index::open_or_create(tmp.path()).unwrap();

    index
        .add_text(
            "doc-1.txt",
            "alpha\nbeta authentication middleware\ngamma\n",
        )
        .unwrap();

    let manifest_path = tmp.path().join("sources").join("manifest.json");
    std::fs::write(&manifest_path, "").unwrap();

    let reopened = Index::open_or_create(tmp.path()).unwrap();
    let results = reopened
        .search("authentication", SearchOptions::default())
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].source_path, "doc-1.txt");
    assert_eq!(results[0].line_number, 2);
    assert!(results[0].snippet.contains("authentication middleware"));

    let wal_dir = tmp.path().join("wal");
    assert!(wal_dir.join("wal.meta").exists());
    assert!(wal_dir.join("wal.content").exists());
}

#[test]
fn search_returns_empty_when_term_is_absent() {
    let tmp = tempdir().unwrap();
    let index = Index::open_or_create(tmp.path()).unwrap();
    index.add_text("doc-1.txt", "alpha\nbeta\ngamma\n").unwrap();

    let results = index
        .search("middleware", SearchOptions::default())
        .unwrap();

    assert!(results.is_empty());
}

#[test]
fn search_ignores_truncated_trailing_meta_record() {
    let tmp = tempdir().unwrap();
    let index = Index::open_or_create(tmp.path()).unwrap();
    index
        .add_text("doc-1.txt", "alpha\nauthentication middleware\n")
        .unwrap();

    let meta_path = tmp.path().join("wal").join("wal.meta");
    let mut meta = OpenOptions::new().append(true).open(meta_path).unwrap();
    meta.write_all(b"{\"source_path\":\"broken").unwrap();
    meta.flush().unwrap();

    let reopened = Index::open_or_create(tmp.path()).unwrap();
    let results = reopened
        .search("authentication", SearchOptions::default())
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].source_path, "doc-1.txt");
}

#[test]
fn search_ignores_truncated_invalid_utf8_meta_tail() {
    let tmp = tempdir().unwrap();
    let index = Index::open_or_create(tmp.path()).unwrap();
    index
        .add_text("doc-1.txt", "alpha\nauthentication middleware\n")
        .unwrap();

    let meta_path = tmp.path().join("wal").join("wal.meta");
    let mut meta = OpenOptions::new().append(true).open(meta_path).unwrap();
    meta.write_all(b"{\"source_path\":\"broken\",\"content_hash\":\"")
        .unwrap();
    meta.write_all(&[0xF0]).unwrap();
    meta.flush().unwrap();

    let reopened = Index::open_or_create(tmp.path()).unwrap();
    let results = reopened
        .search("authentication", SearchOptions::default())
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].source_path, "doc-1.txt");
}

#[test]
fn search_ignores_unterminated_but_parseable_meta_tail() {
    let tmp = tempdir().unwrap();
    let index = Index::open_or_create(tmp.path()).unwrap();
    index
        .add_text("doc-1.txt", "alpha\nauthentication middleware\n")
        .unwrap();

    let meta_path = tmp.path().join("wal").join("wal.meta");
    let mut meta = OpenOptions::new().append(true).open(meta_path).unwrap();
    meta.write_all(
        br#"{"source_path":"tail-doc","content_hash":"x","byte_offset":0,"byte_length":10,"committed_at":1}"#,
    )
    .unwrap();
    meta.flush().unwrap();

    let reopened = Index::open_or_create(tmp.path()).unwrap();
    let results = reopened
        .search("authentication", SearchOptions::default())
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].source_path, "doc-1.txt");
}

#[test]
fn open_drops_metadata_entries_that_overshoot_content_file() {
    let tmp = tempdir().unwrap();
    {
        let index = Index::open_or_create(tmp.path()).unwrap();
        index
            .add_text("doc-1.txt", "alpha\nauthentication middleware\n")
            .unwrap();
    }

    let meta_path = tmp.path().join("wal").join("wal.meta");
    let mut meta = OpenOptions::new().append(true).open(&meta_path).unwrap();
    meta.write_all(
        br#"{"source_path":"overshoot","content_hash":"x","byte_offset":999999999,"byte_length":10,"committed_at":1}
"#,
    )
    .unwrap();
    meta.flush().unwrap();

    let reopened = Index::open_or_create(tmp.path()).unwrap();
    let results = reopened
        .search("authentication", SearchOptions::default())
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].source_path, "doc-1.txt");

    let meta_contents = std::fs::read_to_string(meta_path).unwrap();
    assert_eq!(meta_contents.lines().count(), 1);
    assert!(meta_contents.contains("doc-1.txt"));
    assert!(!meta_contents.contains("overshoot"));
}

#[test]
fn search_sees_content_written_after_open() {
    let dir = tempdir().unwrap();
    let index = Index::open_or_create(dir.path()).unwrap();

    index.add_text("file1.rs", "fn hello() {}\n").unwrap();
    let results = index.search("hello", SearchOptions::default()).unwrap();
    assert_eq!(results.len(), 1);

    index.add_text("file2.rs", "fn goodbye() {}\n").unwrap();

    let results = index.search("goodbye", SearchOptions::default()).unwrap();
    assert_eq!(
        results.len(),
        1,
        "mmap must see content written after initial open"
    );

    let results = index.search("fn ", SearchOptions::default()).unwrap();
    assert_eq!(results.len(), 2, "both entries must be visible");
}

#[test]
fn empty_query_returns_no_results() {
    let tmp = tempdir().unwrap();
    let index = Index::open_or_create(tmp.path()).unwrap();
    index
        .add_text("doc-1.txt", "alpha\nauthentication middleware\n")
        .unwrap();

    let results = index.search("", SearchOptions::default()).unwrap();

    assert!(results.is_empty());
}

#[test]
fn large_corpus_search_completes_without_ooming() {
    let tmp = tempdir().unwrap();
    let index = Index::open_or_create(tmp.path()).unwrap();
    let chunk = "abcdefghij".repeat(1024);

    for i in 0..10_000 {
        index
            .add_text(format!("doc-{i}.txt"), format!("{}\n", chunk))
            .unwrap();
    }

    let reopened = Index::open_or_create(tmp.path()).unwrap();
    let results = reopened
        .search(
            "needle-that-does-not-exist",
            SearchOptions {
                query_promoted_dense: QueryPromotedDense {
                    max_promoted_chunks: 0,
                    max_promoted_ms: 0,
                },
                ..Default::default()
            },
        )
        .unwrap();

    assert!(results.is_empty());
}
