#[cfg(feature = "semantic")]
mod path2_smoke_tests {
    use std::sync::Mutex;

    use sieve_core::fusion::ResultSource;
    use sieve_core::{Index, SearchOptions};
    use tempfile::tempdir;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn fixture_path(name: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join(name)
    }

    #[test]
    fn test_path2_end_to_end_smoke_random_encoder_weights() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::set_var(
            "SIEVE_ENCODER_QUERY_ONNX",
            fixture_path("query_encoder.onnx"),
        );
        std::env::set_var("SIEVE_ENCODER_DOC_ONNX", fixture_path("doc_encoder.onnx"));

        let dir = tempdir().unwrap();
        let index = Index::open_or_create(dir.path()).unwrap();
        for doc_id in 0..100 {
            let content = if doc_id == 42 {
                "pub fn known_correct_answer() -> &'static str { \"needle42 target\" }\n"
                    .to_string()
            } else {
                format!(
                    "pub fn synthetic_doc_{doc_id}() -> usize {{ {doc_id} }}\n// filler token_{doc_id}\n"
                )
            };
            index
                .add_text(&format!("src/doc_{doc_id:03}.rs"), content)
                .unwrap();
        }
        sieve_core::lexical::build_pending_shards(&index).unwrap();
        let embedded = index.embed_pending(16).unwrap();
        assert!(
            embedded >= 100,
            "expected one or more chunks per document, got {embedded}"
        );

        let mut found_known_top5 = false;
        for query_id in 0..10 {
            let query = if query_id == 0 {
                "known_correct_answer needle42 target".to_string()
            } else {
                format!("synthetic_doc_{query_id} token_{query_id}")
            };
            let outcome = index
                .search_with_outcome(
                    &query,
                    SearchOptions {
                        top_k: Some(10),
                        ..Default::default()
                    },
                )
                .unwrap();
            assert!(
                outcome.results.len() >= 5,
                "query {query_id} returned only {} candidates",
                outcome.results.len()
            );
            assert!(outcome
                .results
                .iter()
                .all(|result| result.score.is_finite()));
            assert!(
                outcome
                    .source_sets
                    .iter()
                    .any(|set| set.source == ResultSource::HotVector),
                "Path 2 dense vector source did not contribute for query {query_id}"
            );
            if query_id == 0
                && outcome
                    .results
                    .iter()
                    .take(5)
                    .any(|result| result.source_path == "src/doc_042.rs")
            {
                found_known_top5 = true;
            }
        }
        assert!(found_known_top5, "known-correct answer was not in top-5");

        std::env::remove_var("SIEVE_ENCODER_QUERY_ONNX");
        std::env::remove_var("SIEVE_ENCODER_DOC_ONNX");
    }
}
