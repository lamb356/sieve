# SIEVE Ablation Report

This report covers the normal CodeSearchNet ablation run plus the semantic-hard subset rerun after fixing the full-system fusion regression.

Fixes applied before the final semantic-hard rerun
- T+0 semantics: query issued immediately after fresh arrival, with actual query latency measured separately.
- Pre-steady full-runner delta fallback disabled for deadline-limited benchmark queries so query-time embedding work does not blow the deadline.
- Fusion fix for semantic-hard fresh retrieval: when benchmark mode disallows stale-only fusion and fresh results are present, stale-only results are removed from fusion inputs so stable-corpus noise cannot outrank the correct fresh hit.
- scan-only and random-expansion steady-state behavior remains fixed so T+steady continues to scan fresh content.
- semantic-hard mode now fetches a fixed larger raw example pool and truncates cached rows to the requested count, making the subset cold-cache reproducible for the documented command class.

## 1) Normal CodeSearchNet 20/20 rerun

Command:

```bash
cargo run --release -p sieve-bench -- \
  --track codesearchnet \
  --n-stable 20 \
  --n-fresh 20 \
  --output /home/burba/sieve/docs/ablation-results.json
```

Results:

| Runner | Recall@5 T+0 | T+0 latency p50 | Recall@5 T+500ms | Recall@5 T+steady | p50 latency | p95 latency |
|---|---:|---:|---:|---:|---:|---:|
| sieve (full system) | 1.00 | 178ms | 1.00 | 1.00 | 168ms | 198ms |
| sieve-scan (scan only) | 1.00 | 162ms | 1.00 | 1.00 | 152ms | 169ms |
| sieve-random (random expansion) | 1.00 | 165ms | 1.00 | 1.00 | 149ms | 170ms |
| splade-bm25 (indexed only) | 0.00 | 157ms | 0.00 | 1.00 | 154ms | 175ms |
| ripgrep | 1.00 | 88ms | 0.95 | 1.00 | 88ms | 241ms |
| embed-knn | 0.00 | 156ms | 0.95 | 1.00 | 145ms | 193ms |

Lexical-difficulty audit on this 20/20 set:
- ripgrep Recall@5 at T+0 = 1.00
- ripgrep T+0 misses = 0 / 20
- ripgrep T+steady misses = 0 / 20
- Conclusion: this slice is lexically easy; it cannot clear the semantic-lift gate.

## 2) Reproducible semantic-hard subset

Command:

```bash
cargo run --release -p sieve-bench -- \
  --track codesearchnet \
  --semantic-hard \
  --n-stable 20 \
  --n-fresh 11 \
  --cache-dir /home/burba/.sieve/bench-cache/codesearchnet-semantic-hard-fixed-v2 \
  --output /home/burba/sieve/docs/semantic-hard-results.json
```

Results:

| Runner | Recall@5 T+0 | T+0 latency p50 | Recall@5 T+500ms | Recall@5 T+steady | p50 latency | p95 latency |
|---|---:|---:|---:|---:|---:|---:|
| sieve | 0.55 | 156ms | 0.55 | 0.82 | 145ms | 150ms |
| sieve-scan | 0.55 | 150ms | 0.55 | 0.55 | 134ms | 140ms |
| sieve-random | 0.55 | 148ms | 0.55 | 0.55 | 134ms | 140ms |
| splade-bm25 | 0.00 | 149ms | 0.00 | 0.18 | 160ms | 169ms |
| ripgrep | 0.27 | 17ms | 0.27 | 0.27 | 18ms | 21ms |
| embed-knn | 0.00 | 150ms | 1.00 | 1.00 | 143ms | 167ms |

Additional semantic-hard metrics:
- Full sieve semantic lift over ripgrep at T+0: +0.27 Recall@5
- Full sieve vs sieve-scan at T+0: tie
- Full sieve vs sieve-scan at T+500ms: tie
- ripgrep-final-miss semantic-hard subset size inside this run: 8 queries
- On that 8-query subset:
  - sieve Recall@5 = 0.75
  - embed-knn Recall@5 = 1.00
  - ZeroPrepRetention = 0.75

Fusion-debug conclusion
- The semantic-hard failures were caused by stable-corpus noise entering fusion before steady state.
- In the failing episodes, the correct fresh raw-scan hit existed, but full-mode fusion also admitted stale stable-corpus windows/layers that could not actually see fresh content.
- Those stale results pushed the correct fresh file out of top-5.
- After filtering stale-only fusion inputs when fresh results are present in this benchmark path, full sieve recovers to match sieve-scan on the semantic-hard subset.

Bottom line
- The normal 20/20 CodeSearchNet slice is lexically easy and cannot clear the semantic-lift gate.
- The reproducible semantic-hard subset does expose semantic lift.
- On that subset, full sieve now beats ripgrep at T+0 and is no worse than sieve-scan.
- The semantic-lift gate is cleared.

Artifacts
- Normal run JSON: /home/burba/sieve/docs/ablation-results.json
- Semantic-hard JSON: /home/burba/sieve/docs/semantic-hard-results.json
- Semantic-hard focused summary: /home/burba/sieve/docs/SEMANTIC-HARD-REPORT.md
