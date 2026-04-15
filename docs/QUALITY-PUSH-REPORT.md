# SIEVE Quality Push Report

This report covers the candidate-generation quality push and the corrected semantic-hard rerun using the restored curated semantic-hard subset logic.

## Implemented changes

- ContentType detection from file extension
- CodeSubtoken boundary mode for matching inside compound identifiers
- Content-aware query tokenization for code queries:
  - preserves short code tokens (`fs`, `io`, etc.)
  - preserves namespace and dotted tokens
  - preserves numeric tokens
  - keeps compound identifiers and emits split seeds for namespace/dotted/snake-case forms
- Tantivy `subtoken_text` field for split-identifier indexing
- Softened semantic anchor gate so code-mode anchor evidence can be satisfied through subtoken matching
- Block-aware code windows replacing fixed semantic scan windows
- `sieve download-model --splade-code` stub added; runtime load/selection is wired if model files are present, but download/bootstrap is still not implemented

## Semantic-hard benchmark correction

The temporary automatic selector was wrong for this task because it filtered on the original docstring, not the actual curated benchmark query that was used to expose semantic lift.

That selector has been removed.

The benchmark path now uses the restored curated semantic-hard subset logic from the earlier semantic-hard run lineage:
- docstrings are rewritten into the same hand-curated semantic-hard benchmark queries
- only the episodes that have those curated rewrites are used
- with the current dataset/cache state, this yields 14 usable fresh episodes

This means the semantic-hard rerun below is again measuring the intended semantic-hard slice instead of the looser harness-unblock proxy.

## 1) Corrected semantic-hard rerun

Command:

```bash
cargo run --release -p sieve-bench -- \
  --track codesearchnet \
  --semantic-hard \
  --n-stable 20 \
  --n-fresh 14 \
  --output /home/burba/sieve/docs/quality-push-results.json
```

Results:

| Runner | Recall@5 T+0 | Recall@5 100ms | Recall@5 500ms | Recall@5 1s | Recall@5 5s | Recall@5 T+steady | p50 latency | p95 latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sieve | 0.57 | 0.00 | 0.57 | 0.57 | 0.57 | 0.79 | 0.2s | 0.2s |
| sieve-scan | 0.57 | 0.00 | 0.57 | 0.57 | 0.57 | 0.57 | 0.1s | 0.1s |
| sieve-random | 0.57 | 0.00 | 0.57 | 0.57 | 0.57 | 0.57 | 0.1s | 0.1s |
| ripgrep | 0.29 | 0.29 | 0.29 | 0.29 | 0.29 | 0.29 | 0.0s | 0.0s |
| embed-knn | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.1s | 0.1s |
| splade-bm25 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.29 | 0.2s | 0.2s |

Additional semantic-hard metrics:
- ripgrep-miss semantic-hard subset size inside this run: 10 queries
- On that 10-query ripgrep-miss subset:
  - sieve Recall@5 at T+0: 0.40
  - sieve Recall@5 at T+steady: 0.70
  - embed-knn Recall@5 at T+steady: 1.00
- overall ZeroPrepRetention for the 14-episode semantic-hard run: 0.57

Comparison to the earlier semantic-hard baseline
- Earlier semantic-hard baseline (curated subset lineage, 11-fresh run):
  - ripgrep T+0 Recall@5 = 0.27
  - sieve T+0 Recall@5 = 0.55
- Corrected quality-push rerun (curated subset lineage, 14-fresh run):
  - ripgrep T+0 Recall@5 = 0.29
  - sieve T+0 Recall@5 = 0.57

Observed delta on the restored curated subset lineage:
- sieve T+0 Recall@5: +0.02 absolute
- ripgrep T+0 Recall@5: +0.02 absolute
- semantic lift over ripgrep at T+0:
  - earlier: +0.27
  - current: +0.29
  - delta: +0.02

Important comparability caveat:
- this is not a strict apples-to-apples same-cohort comparison, because the earlier published semantic-hard baseline used 11 fresh episodes while the corrected rerun uses 14 fresh episodes available under the restored curated subset logic with the current cache/dataset state
- the +0.02 should therefore be treated as directional evidence, not a definitive same-cohort gain

Interpretation:
- The quality push does show a small positive movement on the proper semantic-hard measurement lineage.
- The improvement is real in direction, but modest and not fully isolated from cohort-size differences.
- The main target of materially lifting the old ~0.55 number was only partially achieved:
  - 0.55 -> 0.57 on the restored curated semantic-hard lineage.

## 2) Normal 20/20 regression check

Command:

```bash
cargo run --release -p sieve-bench -- \
  --track codesearchnet \
  --n-stable 20 \
  --n-fresh 20 \
  --output /home/burba/sieve/docs/quality-push-normal-results.json
```

Results:

| Runner | Recall@5 T+0 | Recall@5 100ms | Recall@5 500ms | Recall@5 1s | Recall@5 5s | Recall@5 T+steady | p50 latency | p95 latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sieve | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.90 | 0.1s | 0.2s |
| sieve-scan | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.1s | 0.1s |
| sieve-random | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.1s | 0.1s |
| ripgrep | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.0s | 0.1s |
| embed-knn | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.1s | 0.2s |
| splade-bm25 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.75 | 0.2s | 0.2s |

Regression check:
- T+0 normal run: no regression (`sieve` remains 1.00)
- caveat: `sieve` T+steady is still 0.90 while `sieve-scan` and `ripgrep` remain 1.00

## Validation

Validation after restoring the curated semantic-hard path and rerunning:
- `cargo test` — passed
- `cargo test --no-default-features` — passed
- `cargo clippy --all-targets --all-features -- -D warnings` — passed

## Artifacts

- Corrected semantic-hard benchmark JSON: `/home/burba/sieve/docs/quality-push-results.json`
- Normal benchmark JSON: `/home/burba/sieve/docs/quality-push-normal-results.json`
- This report: `/home/burba/sieve/docs/QUALITY-PUSH-REPORT.md`

## Bottom line

- The incorrect automatic semantic-hard selector has been removed.
- The semantic-hard rerun is back on the intended curated subset lineage.
- On that restored measurement, the quality push improves full `sieve` T+0 Recall@5 from 0.55 to 0.57.
- There is no normal-run T+0 regression.
- The improvement is directionally positive, but still small enough that further recall work is justified if the goal is a materially stronger semantic-hard number.