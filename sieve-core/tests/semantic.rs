#![cfg(feature = "semantic")]

use std::fs;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use sieve_core::fusion::{rrf_fuse, ResultId, ResultSource, ScoredResult};
use sieve_core::model::{
    ModelManager, DEFAULT_CODE_SPARSE_MODEL_NAME, DEFAULT_MODEL_NAME, DEFAULT_SPARSE_MODEL_NAME,
};
use sieve_core::vectors::{snippet_from_byte_range, HotVectorStore, VectorMeta};
use sieve_core::{filter_stale_only_source_sets, Index, SearchOptions, SourceResultSet};
use tempfile::tempdir;

fn home_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn set_home(path: &Path) {
    std::env::set_var("HOME", path);
}

fn unit_vector(dim: usize, hot_index: usize) -> Vec<f32> {
    let mut vector = vec![0.0; dim];
    vector[hot_index] = 1.0;
    vector
}

fn sample_meta(id: u64, path: &str) -> VectorMeta {
    VectorMeta {
        wal_entry_id: id,
        source_path: path.to_string(),
        line_range: (id as usize + 1, id as usize + 1),
        chunk_id: 0,
        byte_range: (0, 8),
    }
}

#[test]
fn test_hot_vector_store_append_and_knn() {
    let dir = tempdir().unwrap();
    let mut store = HotVectorStore::open_or_create(dir.path(), 384).unwrap();
    let vectors: Vec<Vec<f32>> = (0..100)
        .map(|index| unit_vector(384, index % 384))
        .collect();
    let metas: Vec<VectorMeta> = (0..100)
        .map(|index| sample_meta(index as u64, &format!("file-{index}.rs")))
        .collect();
    store.append(&vectors, &metas).unwrap();

    let query = unit_vector(384, 7);
    let matches = store.search_knn(&query, 3).unwrap();
    assert!(!matches.is_empty());
    assert_eq!(matches[0].source_path, "file-7.rs");
    assert!(matches[0].score > 0.99);
}

#[test]
fn test_hot_vector_store_persistence() {
    let dir = tempdir().unwrap();
    {
        let mut store = HotVectorStore::open_or_create(dir.path(), 384).unwrap();
        store
            .append(&[unit_vector(384, 3)], &[sample_meta(1, "persist.rs")])
            .unwrap();
    }
    let store = HotVectorStore::open_or_create(dir.path(), 384).unwrap();
    let matches = store.search_knn(&unit_vector(384, 3), 1).unwrap();
    assert_eq!(matches[0].source_path, "persist.rs");
}

#[test]
fn test_hot_vector_store_mmap_sees_new_writes() {
    let dir = tempdir().unwrap();
    let mut store = HotVectorStore::open_or_create(dir.path(), 384).unwrap();
    store
        .append(&[unit_vector(384, 1)], &[sample_meta(1, "first.rs")])
        .unwrap();
    assert_eq!(store.search_knn(&unit_vector(384, 1), 5).unwrap().len(), 1);
    store
        .append(&[unit_vector(384, 2)], &[sample_meta(2, "second.rs")])
        .unwrap();
    let matches = store.search_knn(&unit_vector(384, 2), 5).unwrap();
    assert_eq!(matches[0].source_path, "second.rs");
    assert_eq!(store.len(), 2);
}

#[test]
fn test_delta_fallback_skips_when_over_budget() {
    let dir = tempdir().unwrap();
    let _guard = home_lock().lock().unwrap();
    set_home(dir.path());
    let index = Index::open_or_create(dir.path()).unwrap();
    for i in 0..100 {
        index
            .add_text(format!("src/{i}.rs"), format!("token-{i}\n"))
            .unwrap();
    }
    assert!(index.delta_fallback_over_budget(10).unwrap());
    let status = index.semantic_status().unwrap();
    assert_eq!(status.vectors, 0);
    assert_eq!(status.total_chunks, 100);
}

#[test]
fn test_search_without_model_degrades_gracefully() {
    let dir = tempdir().unwrap();
    let _guard = home_lock().lock().unwrap();
    set_home(dir.path());
    let index = Index::open_or_create(dir.path()).unwrap();
    index
        .add_text(
            "retry.rs",
            "fn retry_with_backoff() -> Result<()> { Ok(()) }\n",
        )
        .unwrap();
    let results = index
        .search(
            "retry_with_backoff",
            SearchOptions {
                top_k: Some(5),
                ..Default::default()
            },
        )
        .unwrap();
    assert!(!results.is_empty());
    assert!(results[0].source_layer != ResultSource::HotVector);
    assert!(results[0].source_layer != ResultSource::DeltaFallback);
}

#[test]
fn test_rrf_fuses_all_layers() {
    let scan = vec![ScoredResult {
        result_id: ResultId {
            wal_entry_id: 1,
            byte_start: 0,
            byte_end: 4,
        },
        source_path: "scan.rs".into(),
        line_range: (1, 1),
        chunk_id: 0,
        snippet: "scan".into(),
        score: 1.0,
        source_layer: ResultSource::RawScan,
        wal_entry_id: 1,
    }];
    let bm25 = vec![ScoredResult {
        result_id: ResultId {
            wal_entry_id: 2,
            byte_start: 0,
            byte_end: 4,
        },
        source_path: "combo.rs".into(),
        line_range: (2, 2),
        chunk_id: 0,
        snippet: "bm25".into(),
        score: 2.0,
        source_layer: ResultSource::LexicalBm25,
        wal_entry_id: 2,
    }];
    let vecs = vec![ScoredResult {
        result_id: ResultId {
            wal_entry_id: 2,
            byte_start: 0,
            byte_end: 4,
        },
        source_path: "combo.rs".into(),
        line_range: (2, 2),
        chunk_id: 0,
        snippet: "vec".into(),
        score: 3.0,
        source_layer: ResultSource::HotVector,
        wal_entry_id: 2,
    }];
    let delta = vec![ScoredResult {
        result_id: ResultId {
            wal_entry_id: 3,
            byte_start: 0,
            byte_end: 5,
        },
        source_path: "delta.rs".into(),
        line_range: (3, 3),
        chunk_id: 0,
        snippet: "delta".into(),
        score: 4.0,
        source_layer: ResultSource::DeltaFallback,
        wal_entry_id: 3,
    }];
    let fused = rrf_fuse(vec![scan, bm25, vecs, delta], 5.0);
    assert_eq!(fused[0].source_path, "combo.rs");
    assert_eq!(fused[0].source_layer, ResultSource::Fused);
}

#[test]
fn test_semantic_status_without_model_reports_zero_coverage() {
    let dir = tempdir().unwrap();
    let _guard = home_lock().lock().unwrap();
    set_home(dir.path());
    let index = Index::open_or_create(dir.path()).unwrap();
    index.add_text("a.rs", "alpha\n").unwrap();
    let status = index.semantic_status().unwrap();
    assert!(!status.model_cached);
    assert_eq!(status.vectors, 0);
    assert_eq!(status.total_chunks, 1);
}

#[test]
fn test_semantic_status_with_vectors_reports_counts() {
    let dir = tempdir().unwrap();
    let _guard = home_lock().lock().unwrap();
    set_home(dir.path());
    let index = Index::open_or_create(dir.path()).unwrap();
    index.add_text("a.rs", "alpha\n").unwrap();
    index.add_text("b.rs", "beta\n").unwrap();
    let mut store = HotVectorStore::open_or_create(&dir.path().join("vectors"), 384).unwrap();
    store
        .append(
            &[unit_vector(384, 0), unit_vector(384, 1)],
            &[sample_meta(0, "a.rs"), sample_meta(1, "b.rs")],
        )
        .unwrap();
    let status = index.semantic_status().unwrap();
    assert_eq!(status.vectors, 2);
    assert_eq!(status.total_chunks, 2);
    assert_eq!(status.dimension, 384);
}

#[test]
fn test_model_manager_is_cached_false_until_both_files_present() {
    let dir = tempdir().unwrap();
    let manager = ModelManager::new(dir.path());
    let model_dir = manager.model_dir(DEFAULT_MODEL_NAME);
    fs::create_dir_all(&model_dir).unwrap();
    assert!(!manager.is_cached(DEFAULT_MODEL_NAME));
    fs::write(model_dir.join("model.onnx"), b"onnx").unwrap();
    assert!(!manager.is_cached(DEFAULT_MODEL_NAME));
    fs::write(model_dir.join("tokenizer.json"), b"{}").unwrap();
    assert!(manager.is_cached(DEFAULT_MODEL_NAME));
}

#[test]
fn test_vector_store_rejects_dimension_mismatch() {
    let dir = tempdir().unwrap();
    let mut store = HotVectorStore::open_or_create(dir.path(), 384).unwrap();
    let error = store
        .append(&[vec![1.0, 2.0]], &[sample_meta(1, "bad.rs")])
        .unwrap_err();
    assert!(error.to_string().contains("dimension mismatch"));
}

#[test]
fn test_vector_store_len_tracks_appends() {
    let dir = tempdir().unwrap();
    let mut store = HotVectorStore::open_or_create(dir.path(), 384).unwrap();
    assert_eq!(store.len(), 0);
    store
        .append(&[unit_vector(384, 4)], &[sample_meta(1, "len.rs")])
        .unwrap();
    assert_eq!(store.len(), 1);
}

#[test]
fn test_dense_embedding_chunk_aware() {
    let dir = tempdir().unwrap();
    let mut store = HotVectorStore::open_or_create(dir.path(), 384).unwrap();
    let metas = vec![
        VectorMeta {
            wal_entry_id: 1,
            chunk_id: 0,
            source_path: "chunky.rs".into(),
            byte_range: (0, 512),
            line_range: (1, 8),
        },
        VectorMeta {
            wal_entry_id: 1,
            chunk_id: 1,
            source_path: "chunky.rs".into(),
            byte_range: (256, 768),
            line_range: (5, 12),
        },
        VectorMeta {
            wal_entry_id: 1,
            chunk_id: 2,
            source_path: "chunky.rs".into(),
            byte_range: (512, 900),
            line_range: (9, 16),
        },
    ];
    store
        .append(
            &[
                unit_vector(384, 0),
                unit_vector(384, 1),
                unit_vector(384, 2),
            ],
            &metas,
        )
        .unwrap();

    let matches = store.search_knn(&unit_vector(384, 2), 3).unwrap();
    assert_eq!(matches[0].wal_entry_id, 1);
    assert_eq!(matches[0].chunk_id, 2);
    assert_eq!(matches[0].byte_range, (512, 900));
    assert_eq!(matches[1].chunk_id, 0);
    assert_eq!(matches[2].chunk_id, 1);
}

#[test]
fn test_vector_snippet_from_byte_range() {
    let content = "header line\nexact semantic window\ntrailer line\n";
    let start = content.find("exact").unwrap() as u32;
    let end = (content.find("trailer").unwrap() - 1) as u32;

    let snippet = snippet_from_byte_range(content, (start, end)).unwrap();
    assert_eq!(snippet, "exact semantic window");
}

#[test]
fn test_model_registry_stubbed() {
    let dir = tempdir().unwrap();
    let manager = ModelManager::new(dir.path());
    let model_dir = manager.model_dir(DEFAULT_MODEL_NAME);
    fs::create_dir_all(&model_dir).unwrap();
    fs::write(model_dir.join("model.onnx"), b"onnx").unwrap();
    fs::write(model_dir.join("tokenizer.json"), b"{}").unwrap();

    let registry = manager.registry().unwrap();
    assert!(registry.dense.is_some());
    assert!(registry.sparse.is_none());
    assert!(registry.event_reranker.is_none());

    let event = manager.ensure_event_model().unwrap();
    assert!(event.is_none());
}

#[test]
fn test_model_manager_sparse_uses_manual_copy_layout() {
    let dir = tempdir().unwrap();
    let manager = ModelManager::new(dir.path());
    let sparse_dir = manager.model_dir(DEFAULT_SPARSE_MODEL_NAME);
    fs::create_dir_all(sparse_dir.join("splade-tokenizer")).unwrap();
    fs::write(sparse_dir.join("splade.onnx"), b"onnx").unwrap();
    fs::write(sparse_dir.join("splade.onnx.data"), b"weights").unwrap();
    fs::write(
        sparse_dir.join("splade-tokenizer").join("tokenizer.json"),
        b"{}",
    )
    .unwrap();

    assert!(manager.is_cached(DEFAULT_SPARSE_MODEL_NAME));

    let handle = manager.ensure_sparse_model().unwrap();
    assert_eq!(handle.model_path, sparse_dir.join("splade.onnx"));
    assert_eq!(
        handle.tokenizer_path,
        sparse_dir.join("splade-tokenizer").join("tokenizer.json")
    );
}

#[test]
fn test_model_manager_code_sparse_uses_manual_copy_layout() {
    let dir = tempdir().unwrap();
    let manager = ModelManager::new(dir.path());
    let sparse_dir = manager.model_dir(DEFAULT_CODE_SPARSE_MODEL_NAME);
    fs::create_dir_all(sparse_dir.join("splade-tokenizer")).unwrap();
    fs::write(sparse_dir.join("splade.onnx"), b"onnx").unwrap();
    fs::write(sparse_dir.join("splade.onnx.data"), b"weights").unwrap();
    fs::write(
        sparse_dir.join("splade-tokenizer").join("tokenizer.json"),
        b"{}",
    )
    .unwrap();

    assert!(manager.is_cached(DEFAULT_CODE_SPARSE_MODEL_NAME));

    let handle = manager.ensure_code_sparse_model().unwrap();
    assert_eq!(handle.model_path, sparse_dir.join("splade.onnx"));
    assert_eq!(
        handle.tokenizer_path,
        sparse_dir.join("splade-tokenizer").join("tokenizer.json")
    );
}

#[test]
fn test_search_end_to_end_with_chunks() {
    let dir = tempdir().unwrap();
    let index = Index::open_or_create(dir.path()).unwrap();
    let content = format!(
        "{}\nneedle phrase lives here\n{}",
        "left".repeat(120),
        "right".repeat(120)
    );
    index.add_text("notes/chunks.txt", content).unwrap();
    sieve_core::lexical::build_pending_shards(&index).unwrap();

    let results = index
        .search(
            "needle phrase",
            SearchOptions {
                top_k: Some(10),
                ..Default::default()
            },
        )
        .unwrap();

    assert!(!results.is_empty());
    assert!(results.iter().any(|result| {
        result.source_path == "notes/chunks.txt"
            && result.chunk_id <= 2
            && result.byte_range.0 < result.byte_range.1
            && result.snippet.contains("needle phrase lives here")
    }));
}

#[test]
fn test_filter_stale_only_source_sets_keeps_sets_with_fresh_results() {
    let fresh_scan = SourceResultSet {
        source: ResultSource::RawScan,
        weight: 0.90,
        results: vec![ScoredResult {
            result_id: ResultId {
                wal_entry_id: 100,
                byte_start: 0,
                byte_end: 8,
            },
            source_path: "fresh.rs".into(),
            line_range: (1, 1),
            chunk_id: 0,
            snippet: "fresh".into(),
            score: 1.0,
            source_layer: ResultSource::RawScan,
            wal_entry_id: 100,
        }],
    };
    let stable_bm25 = SourceResultSet {
        source: ResultSource::SpladeBm25,
        weight: 1.10,
        results: vec![ScoredResult {
            result_id: ResultId {
                wal_entry_id: 200,
                byte_start: 0,
                byte_end: 8,
            },
            source_path: "stable.rs".into(),
            line_range: (1, 1),
            chunk_id: 0,
            snippet: "stable".into(),
            score: 10.0,
            source_layer: ResultSource::SpladeBm25,
            wal_entry_id: 200,
        }],
    };
    let filtered = filter_stale_only_source_sets(
        vec![fresh_scan.clone(), stable_bm25],
        &std::collections::HashSet::from([100_u64]),
        false,
    );
    assert_eq!(filtered, vec![fresh_scan]);
}

#[test]
#[ignore = "requires downloaded ONNX model"]
fn test_embedder_loads_and_produces_vectors() {}

#[test]
#[ignore = "requires downloaded ONNX model"]
fn test_delta_fallback_covers_unembedded() {}

#[test]
#[ignore = "requires downloaded ONNX model"]
fn test_semantic_search_finds_meaning_not_words() {}

#[test]
#[ignore = "requires network download"]
fn test_download_model_command() {}
