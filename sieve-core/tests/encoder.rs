#[cfg(feature = "semantic")]
mod encoder_tests {
    use std::sync::Arc;
    use std::thread;

    use sieve_core::embed::{Embedder, DOC_SEQ_LEN, QUERY_SEQ_LEN};

    fn fixture_path(name: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join(name)
    }

    fn fixture_encoder() -> Embedder {
        Embedder::load(
            &fixture_path("query_encoder.onnx"),
            &fixture_path("doc_encoder.onnx"),
        )
        .expect("fixture encoder loads")
    }

    fn assert_finite_vector(vector: &[f32]) {
        assert_eq!(vector.len(), 4);
        assert!(vector.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn test_encode_query_smoke() {
        let encoder = fixture_encoder();
        let vector = encoder.encode_query("fn alpha() -> usize { 42 }").unwrap();
        assert_finite_vector(&vector);
        assert!(vector.iter().any(|value| *value != 0.0));
        assert_eq!(encoder.dimension(), 4);
        assert_eq!(encoder.query_seq_len(), QUERY_SEQ_LEN);
    }

    #[test]
    fn test_encode_documents_batched() {
        let encoder = fixture_encoder();
        let docs: Vec<String> = (0..10)
            .map(|idx| format!("pub fn doc_{idx}() -> usize {{ {idx} }}"))
            .collect();
        let vectors = encoder.encode_documents(&docs).unwrap();
        assert_eq!(vectors.len(), 10);
        for vector in &vectors {
            assert_finite_vector(vector);
        }
    }

    #[test]
    fn test_encode_handles_empty_string() {
        let encoder = fixture_encoder();
        let vector = encoder.encode_query("").unwrap();
        assert_finite_vector(&vector);
    }

    #[test]
    fn test_encode_handles_oversized_input() {
        let encoder = fixture_encoder();
        let oversized = "x".repeat(DOC_SEQ_LEN * 2 + 17);
        let docs = vec![oversized];
        let vectors = encoder.encode_documents(&docs).unwrap();
        assert_eq!(vectors.len(), 1);
        assert_finite_vector(&vectors[0]);
        assert_eq!(encoder.doc_seq_len(), DOC_SEQ_LEN);
    }

    #[test]
    fn test_encode_concurrent() {
        let encoder = Arc::new(fixture_encoder());
        let mut handles = Vec::new();
        for thread_idx in 0..10 {
            let encoder = Arc::clone(&encoder);
            handles.push(thread::spawn(move || {
                for query_idx in 0..100 {
                    let vector = encoder
                        .encode_query(&format!("thread {thread_idx} query {query_idx}"))
                        .unwrap();
                    assert_finite_vector(&vector);
                }
            }));
        }
        for handle in handles {
            handle.join().unwrap();
        }
    }
}
