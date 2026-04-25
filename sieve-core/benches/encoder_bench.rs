#[cfg(feature = "semantic")]
use criterion::{criterion_group, criterion_main, Criterion};
#[cfg(feature = "semantic")]
use sieve_core::embed::Embedder;

#[cfg(feature = "semantic")]
fn fixture_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

#[cfg(feature = "semantic")]
fn bench_single_query_encoding(c: &mut Criterion) {
    let encoder = Embedder::load(
        &fixture_path("query_encoder.onnx"),
        &fixture_path("doc_encoder.onnx"),
    )
    .expect("fixture ONNX encoder loads");
    c.bench_function("encoder/single_query", |b| {
        b.iter(|| {
            let vector = encoder
                .encode_query("find the known correct answer in this synthetic Rust corpus")
                .expect("query encoding succeeds");
            criterion::black_box(vector);
        })
    });
}

#[cfg(feature = "semantic")]
criterion_group!(benches, bench_single_query_encoding);
#[cfg(feature = "semantic")]
criterion_main!(benches);

#[cfg(not(feature = "semantic"))]
fn main() {}
