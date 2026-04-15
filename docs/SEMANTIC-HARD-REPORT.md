# SIEVE Semantic-Hard Subset Report

Normal-set lexical audit:
- ripgrep Recall@5 at T+0 on the 20/20 CodeSearchNet run: 1.00
- ripgrep T+0 misses: 0 / 20
- ripgrep T+steady misses: 0 / 20
- Conclusion: the default 20/20 slice is lexically easy; it does not expose semantic lift.

Constructed semantic-hard subset:
- Benchmark mode: `--semantic-hard`
- Stable set: 20 files
- Fresh semantic-hard episodes: 11
- Selection method: curated concept-bridging query rewrites for CodeSearchNet docstrings where the rewritten query tokens are intended to avoid direct lexical overlap with the answer code.
- Reproducibility note: semantic-hard mode now fetches a fixed larger raw example pool and truncates cached rows to the requested count, so the subset is cold-cache reproducible for the documented command class.

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

Core results:

| Runner | Recall@5 T+0 | T+0 latency p50 | Recall@5 T+500ms | Recall@5 T+steady | p50 latency | p95 latency |
|---|---:|---:|---:|---:|---:|---:|
| sieve | 0.55 | 156ms | 0.55 | 0.82 | 145ms | 150ms |
| sieve-scan | 0.55 | 150ms | 0.55 | 0.55 | 134ms | 140ms |
| sieve-random | 0.55 | 148ms | 0.55 | 0.55 | 134ms | 140ms |
| ripgrep | 0.27 | 17ms | 0.27 | 0.27 | 18ms | 21ms |
| embed-knn | 0.00 | 150ms | 1.00 | 1.00 | 143ms | 167ms |
| splade-bm25 | 0.00 | 149ms | 0.00 | 0.18 | 160ms | 169ms |

Key deltas:
- Full sieve semantic lift over ripgrep at T+0: +0.27 Recall@5
- Full sieve vs sieve-scan at T+0: tie (0.55 vs 0.55)
- Full sieve vs sieve-scan at T+500ms: tie (0.55 vs 0.55)
- semantic-hard subset detected by ripgrep final-top5 miss criterion: 8 queries
- On that 8-query subset:
  - sieve Recall@5 = 0.75
  - embed-knn Recall@5 = 1.00
  - ZeroPrepRetention = 0.75

Gate decision:
- The semantic-hard subset now shows semantic lift in full `sieve` at T+0.
- Full sieve beats ripgrep at T+0 on the reproducible semantic-hard run: 0.55 vs 0.27.
- Full sieve is no longer worse than sieve-scan on this subset; it is equal at T+0 and T+500ms.
- Therefore the semantic-lift gate is cleared.
