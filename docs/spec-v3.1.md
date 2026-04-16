# SIEVE Build Spec V3.1

**Version**: 2026-04-16-v3.1  
**Status**: Active — supersedes `SIEVE-BUILD-SPEC-V3.md`, `RESEARCH-CONTRACT-V2.pdf`, `SIEVE-FUTURE-SPEC.md`, and `SIEVE-FUTURE-SPEC-REVISION-NOTES.md`  
**Governing rule**: Every batch must answer *what specific miss category does this close, and how do we measure it*. No speculative work, no polish, no "nice to haves."

-----

## 1. Current State

### 1.1 What is proven

The core novelty claim holds. Ablation evidence from commits `a993fb4` and `4ed6d6b`:

- Sieve full system T+0 Recall@5 = 1.00 on CodeSearchNet semantic-hard subset
- Sieve scan-only T+0 Recall@5 = 0.57 on the same subset
- ripgrep T+0 Recall@5 = 0.29 on the same subset
- Semantic lift over ripgrep at T+0 = +0.71 (full system) / +0.28 (scan only)
- ZeroPrepRetention@5 = 1.00
- Full benchmark T+0 / T+steady = 1.00 / 1.00
- Normal benchmark T+0 / T+steady = 1.00 / 1.00
- Semantic-hard T+steady = 0.93, which is a fusion/steady-state regression, not a scan-floor regression

Kill criteria from the research-contract phase were cleared. The T+0 raw-byte semantic scan primitive is proven.

### 1.2 What exists in the repo on main

- Query-compiled SPLADE → Aho-Corasick byte automaton scan path
- Hot dense vectors (`bge-small-en-v1.5`, 384d, ONNX, flat kNN)
- Bounded delta fallback upgraded to first-class query-promoted dense
- Tantivy BM25 lexical shards with split-identifier field for subtoken indexing
- Two-file WAL (metadata + mmap'd content)
- Coverage-aware weighted RRF fusion
- Ablation infrastructure: `--no-embed`, `--explain`, `--no-expand`, `--no-window-scoring`, `--no-df-filter`, `--scan-only`, plus `sieve-random` and `splade-bm25` benchmark runners
- Semantic-hard subset filtering with curated rewrite logic
- T+0 through T+steady deadline protocol
- Content-type detection, code-subtoken matching, block-aware code windows, code-aware query tokenization

### 1.3 What is not done

- **Benchmark power/breadth is insufficient** — current semantic-hard slice is too small and too narrow to justify heavy model work
- **Candidate generation ceiling remains low** — scan-only plateaued at 0.57 and must move substantially before reranking matters
- **Indexed sparse path is underpowered** — Tantivy semantic path is still over-gated and underperforming
- **Fusion is not yet the final quality layer** — semantic-hard T+steady still regresses to 0.93
- **Ops hardening is incomplete** — scheduler, cache reuse, cancellation, WAL graduation, compaction, and watch-mode correctness are not finished
- **Reranker training data is dirty** — exported IDF/DF features are flattened and content type is not preserved end-to-end
- **Long-document cross-window reasoning is not implemented**
- **Local adaptation is not shipped** — `sieve train calibrate` does not exist
- **External release surfaces are not done** — MCP, production README, crates publish, paper, demo

### 1.4 What this document supersedes

The following prior specs are archived reference only. Hermes must not use them for scope or build decisions.

- `SIEVE-BUILD-SPEC-V3.md`
- `RESEARCH-CONTRACT-V2.pdf`
- `SIEVE-FUTURE-SPEC.md`
- `SIEVE-FUTURE-SPEC-REVISION-NOTES.md`
- `README.md` on main — stale architecture state
- `spec-v0.3.md` — predates current architecture

-----

## 2. Governing Principles

### 2.1 Miss-category discipline

Every batch names the exact failure mode it closes. If a proposed task does not map to a named miss category with a named measurement, it waits.

### 2.2 Bottleneck discipline

Never optimize above the current bottleneck. Reranker improvements are pointless if candidate generation is weak. Scheduler improvements are pointless if fusion is the current quality limiter. Work the limiting factor first.

### 2.3 Novelty preservation

The T+0 raw-byte scan path with zero document-side preprocessing is the non-negotiable core. No new tier is allowed to replace it or dilute its claim. Dense, indexed sparse, or future multi-vector tiers sit alongside scan, never in place of scan.

### 2.4 Measurement before optimization

Every batch ends with a benchmark rerun. If the number did not move, the batch did not succeed regardless of what shipped.

### 2.5 Benchmark breadth before expensive model work

No new learned component is justified on a tiny or single-track benchmark. Expand the benchmark before heavy candidate-generation or reranker model work.

### 2.6 Local-first product, cloud-only for training compute

Sieve is a local-first product. The trained model artifact ships with the binary. Training compute is rented from the cloud when needed. No cloud inference, no user data leaves the user's machine. See Section 4.5 for the full training infrastructure policy.

-----

## 3. The Nine Batches

Batch order is the build order. Hermes does not pull tasks forward from a later batch without explicit approval.

### Batch 0 — Benchmark Expansion

**Goal**: Expand the benchmark until a +3 point Recall@5 improvement is statistically distinguishable from noise at `p < 0.05`.

**Primary miss category**: **Measurement underpower** — the current benchmark is too small or too narrow to justify expensive quality work.

**Training compute**: None. 2080 Ti handles all work.

**Tasks**:

1. **Expand episode count and variance estimation.**
   - Per track, per run: `n_stable = 1000`, `n_fresh = 200`
   - Number of runs: `5`
   - Total fresh episodes across primary tracks: `2000`
   - Stable/fresh partitions must vary by seed; seeds are committed in benchmark artifacts

2. **Add exactly one additional benchmark track beyond CodeSearchNet Python.**
   - **Track name**: `codesearchnet-java`
   - **Source**: CodeSearchNet Java test split
   - **Construction**:
     - Same D0 / D_fresh episode protocol as Python
     - Same semantic-hard curation logic lineage as commit `4ed6d6b`
     - One judged-relevant fresh file per episode
     - Same T+0 / T+500ms / T+steady deadline ladder
   - **Why this track**:
     - Low integration cost because it shares corpus structure with the existing harness
     - High value because Java stresses CamelCase, type-heavy identifiers, imports, and API surfaces differently from Python

3. **Add confidence intervals and significance tests.**
   - Single-system Recall@5 CIs: Wilson 95% intervals
   - Paired system-difference CIs: stratified paired bootstrap, `10_000` resamples
   - Primary significance test for paired Recall@5 differences: exact McNemar
   - Report both effect size and `p` value for every primary comparison

4. **Add miss-categorization infrastructure.**
   - Benchmark JSON must emit, for every miss:
     - query
     - relevant path
     - top-k paths
     - emitted expansion terms
     - top window surfaces
     - `miss_category`
     - `miss_evidence`
   - Initial schema:
     - `DLatent`
     - `DHard`
     - `IndexedSparseGate`
     - `FusionMisorder`
     - `Other`
   - Batch 0 adds the schema and evidence plumbing; Batch 2 uses it operationally

5. **Standardize latency reporting.**
   - Every report row must include:
     - `track`
     - `runner`
     - `deadline`
     - `n`
     - `recall_at_5`
     - `ci95_low`
     - `ci95_high`
     - `latency_ms_p50`
     - `latency_ms_p95`
   - Benchmark JSON must include the same fields under a stable schema

**Success metric**:

- Benchmark harness demonstrates enough power to distinguish a `+0.03` Recall@5 difference at `p < 0.05`
- Paired Recall@5-difference 95% CI half-width is `<= 0.015` on the pooled primary benchmark
- Both `codesearchnet-python` and `codesearchnet-java` report primary metrics and latency in the standardized format

**Required tests**:

- `test_benchmark_runs_five_seeded_partitions`
- `test_codesearchnet_java_track_builds`
- `test_bootstrap_ci_emitted_for_primary_metrics`
- `test_exact_mcnemar_emitted_for_primary_comparisons`
- `test_miss_schema_includes_category_and_evidence`
- `test_latency_report_includes_p50_p95`

**Commit criteria**:

- All tests pass
- Batch 0 artifacts committed under `docs/benchmarks/`
- Pooled benchmark report includes Python + Java, 5 runs, CIs, and p-values
- Independent review confirms the harness can detect a `+3` point Recall@5 gain at `p < 0.05`

---

### Batch 1 — Foundation Quality

**Goal**: Raise the scan floor and repair the indexed sparse foundation.

**Primary miss category**: **Foundation underreach** — recoverable quality is being lost in the base sparse paths before fusion or reranking.

**Training compute**: None. All SPLADE-Code work uses existing pretrained models via ONNX. 2080 Ti handles inference.

**Task A — SPLADE-Code integration**

- **Miss category**: **Term emission failures**
- **Tasks**:
  1. Wire `ensure_code_sparse_model` to a real ONNX load path
  2. Route code-like queries to SPLADE-Code and prose-like queries to generic SPLADE
  3. Preserve fallback to generic SPLADE with explicit warning if SPLADE-Code is unavailable
- **Success metric**:
  - Scan-only semantic-hard Recall@5 improves from the Batch 0 baseline to `>= 0.63` on code-like queries, or by `>= +0.05` absolute, whichever is larger
  - No prose-track scan-only regression larger than `0.01`

**Task B — Corpus-aware DF + CODE_DF_PRIOR**

- **Miss category**: **Scoring/efficiency failures**
- **Tasks**:
  1. Pass Tantivy `doc_freq` and `total_num_docs` into the scan scorer
  2. Replace fresh-only DF estimation in scan scoring with collection-level statistics when available
  3. Add `CODE_DF_PRIOR` for high-DF code terms (`error`, `handler`, `config`, `impl`, `async`, `result`, etc.)
- **Success metrics**:
  - Events per query reduced by `>= 20%`
  - Scan-path `p95` latency reduced by `>= 15%`
  - On the incumbent-hit subset (queries where relevant is already in top-5 before Task B), the relevant result remains in top-5 on `>= 98%` of queries and median relevant rank does not worsen

**Task C — Indexed sparse-path repair**

- **Miss category**: **Anchor over-gating**
- **Tasks**:
  1. Soften the hard anchor requirement in the Tantivy semantic path
  2. Allow expansion-led retrieval when diversity/coverage conditions are met
  3. Make anchor presence a strong preference, not a hard `Must` clause
  4. Preserve anti-noise controls: minimum term diversity, minimum weighted mass, and post-retrieval plausibility checks
- **Success metric**:
  - `splade-bm25` semantic-hard T+steady Recall@5 improves from the current `0.18` toward `>= 0.40`

**Required tests**:

- `test_splade_code_loads_and_routes`
- `test_splade_code_fallback_to_generic`
- `test_corpus_df_matches_tantivy_stats`
- `test_code_df_prior_applied`
- `test_indexed_sparse_can_retrieve_without_anchor_when_diversity_sufficient`
- `test_anchor_bias_is_preference_not_must`

**Commit criteria**:

- All tests pass
- Batch 1 benchmark rerun reports Task A, Task B, and Task C metrics separately
- No T+0 full-system regression larger than `0.01`
- Independent review confirms indexed sparse-path repair is actually moving `splade-bm25`, not just changing score scales

---

### Batch 2 — Taxonomy → Rescan → Lexicalizer (Conditional)

**Goal**: Push scan-only Recall@5 toward `0.72+` only if remaining misses are still recoverable through better query-side lexicalization.

**Primary miss category**: **D-latent misses** — a byte-level bridge term exists in the file, but the query-side compiler failed to emit it.

**Training compute**: None for taxonomy + rescan. If the lexicalizer is justified and built, training happens on Colab Pro+ (A100). See Section 4.5 for the Colab workflow. Only start Colab subscription if the lexicalizer threshold is met.

**Tasks**:

1. **Categorize misses first.**
   - Use Batch 0 miss-schema infrastructure
   - For every scan-only semantic-hard miss, assign:
     - `DLatent`
     - `DHard`
     - `Other`
   - Emit category counts and examples in benchmark output
   - No lexicalizer work starts until this report exists

2. **Add bounded feedback rescan.**
   - Run initial scan
   - If the scan is weak (`low max score`, `few candidates`, or `low group coverage`), extract salient code surfaces from top-K windows
   - Add feedback terms under a hard global cap
   - Permit exactly one feedback iteration
   - Never activate on already-confident scans

3. **Build the learned lexicalizer only if justified by the miss report.**
   - Justification threshold:
     - after feedback rescan, `DLatent > 30%` of remaining semantic-hard misses
   - If the threshold is not met, the lexicalizer is out of scope for this release cycle
   - If justified:
     - lexicalizer takes NL query + dense query embedding
     - emits weighted code surfaces: identifiers, API names, imports, error strings, path fragments, one short code sketch
     - query-side only; preserves zero-preprocessing claim
   - **Training path if built**:
     - Generate training data locally on 2080 Ti overnight (distillation from teacher)
     - Train on Colab Pro+ A100 with checkpointing every 30 min
     - Export to ONNX locally
     - Validate against eval harness on 2080 Ti before commit

**Success metrics**:

- After miss categorization + bounded rescan:
  - Scan-only semantic-hard Recall@5 reaches `>= 0.68`, or
  - `DLatent` miss count drops by `>= 40%` versus the post-Batch-1 baseline
- If the lexicalizer is justified and built:
  - Scan-only semantic-hard Recall@5 reaches `>= 0.72`
  - Remaining `DLatent` misses drop by `>= 60%` versus the post-Batch-1 baseline
  - Scan-path `p95` latency does not increase by more than `25%`

**Required tests**:

- `test_miss_categorizer_tags_dlatent_dhard_other`
- `test_feedback_rescan_activates_only_on_weak_scan`
- `test_feedback_rescan_bounded_to_one_iteration`
- `test_feedback_terms_respect_global_cap`
- `test_lexicalizer_only_enabled_when_dlatent_threshold_met`
- `test_lexicalizer_fallback_when_model_missing`

**Commit criteria**:

- All tests pass
- Batch 2 benchmark rerun reports pre-rescan, post-rescan, and post-lexicalizer numbers separately
- Miss-categorization report committed under `docs/benchmarks/`
- If lexicalizer is built, latency impact is explicitly reported
- If lexicalizer is trained on Colab, training artifacts and seed are committed under `docs/training/batch2/`

**Kill criteria**:

- If `DLatent <= 30%` of remaining misses after bounded rescan, do not build the lexicalizer in this cycle
- If lexicalizer training/prototype fails to reduce remaining `DLatent` misses by at least `30%`, archive it for post-v1 work
- If `DHard >= 70%` of remaining misses after bounded rescan, stop candidate-generation work and move to Batch 3

---

### Batch 3 — Fusion Refinement

**Goal**: Remove the T+steady regression and make the system spend latency budget only when scan confidence is weak.

**Primary miss category**: **Cross-layer ranking conflict** — correct evidence exists, but static fusion and file/window competition misorder results or waste deadline budget.

**Training compute**: None. Fusion refinement is pure algorithm work.

**Tasks**:

1. **Deadline-aware fusion.**
   - Compute scan confidence from:
     - rare surfaces
     - concentrated windows
     - steep score gap
     - file diversity
   - High-confidence scan returns early
   - Weak-confidence scan spends budget on targeted dense promotion before final fusion

2. **File-level fusion before window selection.**
   - Dedup by stable `file_id`
   - Aggregate evidence across windows before selecting final windows
   - Prevent one file with many mediocre windows from crowding out a file with one strong window

3. **Explicit fusion dedup semantics.**
   - Every fused candidate retains per-source ranks and scores:
     ```rust
     pub struct FusedCandidate {
         pub result_id: ResultId,
         pub source_ranks: SmallVec<[(SourceKind, u16); 4]>,
         pub source_scores: SmallVec<[(SourceKind, f32); 4]>,
     }
     ```
   - Deduplication happens before weighted RRF
   - Overlap across sources is treated as evidence, not ambiguity

4. **Calibrate remaining planner heuristics and dense-budget policies.**
   - Tune:
     - query-promoted dense budget
     - minimum scan-confidence thresholds for early return
     - weak-scan triggers for spending deadline on dense rescue
   - Persist in config/metadata, not source literals

5. **Provisional/final result contract.**
   - Return:
     ```rust
     pub struct SearchResponse {
         pub results: Vec<r>,
         pub semantic_coverage: f32,
         pub revision_id: u64,
         pub finalized_at_ms: Option<u64>,
         pub provisional: bool,
     }
     ```
   - Agents and IDEs may act on provisional results immediately and subscribe to final refinement

6. **Revisit the semantic-hard T+steady regression.**
   - Measure first after Batches 1–2
   - If the regression persists, add a dense-dominance rule:
     - when dense coverage is `Complete`
     - and dense top-result confidence exceeds threshold
     - dense becomes ranking backbone and other layers inject unique candidates only

**Success metrics**:

- Semantic-hard T+steady Recall@5 reaches `>= 0.95`
- No T+0 regression larger than `0.01`
- p50 latency on confident-scan queries improves by `>= 20%`
- Dense-budget policy reduces wasted query-promoted dense work on confident scans by `>= 25%`

**Required tests**:

- `test_deadline_aware_fusion_early_returns_on_confident_scan`
- `test_deadline_aware_fusion_spends_budget_on_weak_scan`
- `test_file_level_fusion_dedups_windows`
- `test_fusion_dedup_retains_source_evidence`
- `test_dense_budget_policy_loaded_from_config`
- `test_provisional_result_contract`

**Commit criteria**:

- All tests pass
- Batch 3 benchmark rerun shows semantic-hard T+steady `>= 0.95`
- T+0 metrics are unchanged within tolerance
- Latency report includes confident-scan early-return savings

---

### Batch 4 — Ops Hardening

**Goal**: Make the freshness path correct under real churn: bulk writes, repeated edits, background embedding, WAL graduation, and watch-mode ingest.

**Primary miss category**: **Freshness-path correctness under churn** — the system can rank well in a benchmark but misbehave when files change rapidly.

**Training compute**: None. Ops work.

**Tasks**:

1. **Semantic debt scheduler (three-queue).**
   - `hot`: recent writes, active editor buffers, recently queried files
   - `query_promoted`: unembedded chunks from the current query's top scan candidates
   - `cold_backfill`: everything else
   - Priority function: `expected_recall_gain / estimated_embed_ms`
   - Interactive queries preempt background workers

2. **Content-hash caching for embeddings.**
   - Cache by BLAKE3 chunk hash
   - Stable chunking so small file edits do not invalidate full-file embeddings

3. **Versioned cancellation.**
   - Every embedding job carries a content hash
   - Stale jobs are cancelled when content changes
   - Only publish embeddings whose hash matches the latest committed WAL state

4. **`sieve watch` command.**
   - `notify`-based filesystem watch
   - debounced shard building
   - `.gitignore` respected
   - clean Ctrl+C exit
   - status lines to `stderr`
   - **Why this stays here**:
     - `watch` is the highest-churn ingest surface
     - it is the easiest operational way to validate scheduler, cancellation, and compaction correctness
     - it is not just polish; it is the main real-world stress path for freshness guarantees

5. **WAL graduation and compaction.**
   - Add `indexed_generation: Option<u64>` to WAL metadata
   - Fresh scan covers only entries with `indexed_generation == None`
   - Compaction triggers by bytes, count, or age
   - Crash-safe graduation markers
   - Invariant: once graduated, entries are served by Tantivy, never scanned

6. **Optional SIMD scan backend evaluation.**
   - Evaluate `aho-corasick` packed mode for small pattern sets
   - Evaluate Hyperscan/VectorScan on x86
   - Ship only if `>= 1.3x` speedup on x86 with no semantic regression
   - Classic AC path remains the portable default

**Success metrics**:

- 100-file burst test completes with no stale embeddings published
- `sieve watch` reflects file changes within `<= 2s`
- No WAL-graduation invariant violations under stress
- No benchmark regression on primary tracks
- If SIMD backend ships, x86 scan throughput improves by `>= 1.3x`

**Required tests**:

- `test_scheduler_prioritizes_query_promoted_over_cold`
- `test_scheduler_cancels_stale_jobs_on_content_change`
- `test_content_hash_cache_reuses_embeddings`
- `test_watch_respects_gitignore`
- `test_watch_debounces_shard_building`
- `test_wal_graduation_invariant`

**Commit criteria**:

- All tests pass
- Bulk-write stress test passes
- `sieve watch` manual smoke test passes
- No T+0 or T+steady benchmark regression
- SIMD decision recorded explicitly: shipped or archived

---

### Batch 5 — Reranker Data Hygiene + Adaptive-Depth Reranker

**Goal**: Improve ordering only where the right window is already retrieved but misranked.

**Primary miss category**: **Top-K misordering** — candidate generation succeeded, but the correct window is below rank 5.

**Training compute**: Colab Pro+ (A100). This is the first required training run. Follow the Section 4.5 workflow for data generation on 2080 Ti, training on Colab, validation on 2080 Ti.

**Task A — Blocking data-hygiene prerequisite**

- **Miss category**: **Dirty supervision**
- **Tasks**:
  1. Fix `training_export.rs` to emit real IDF/DF-derived features instead of all-ones vectors
  2. Preserve actual `ContentType` through export; remove `ContentType::Prose` fallback where ground truth is available
  3. Regenerate the reranker dataset and retrain the current non-recurrent baseline before any adaptive-depth work
- **Success metrics**:
  - `100%` of exported training rows include real IDF/DF features when corpus stats exist
  - `100%` of exported rows preserve content type from source examples
  - Corrected-export baseline matches or beats the old reranker on nDCG@10 before adaptive-depth work proceeds

**Task B — Adaptive-depth reranker**

- **Miss category**: **Ambiguity within retrieved candidates**
- **Tasks**:
  1. Shared-block recurrent-depth model over event tokens plus globals
  2. Candidate-set-level halting based on low KL change **and** high top-candidate confidence
  3. Training on hard candidate sets where the correct window is already in top-32 but misordered
  4. Compute penalty for late halting
  5. Preserve activation gating so the reranker only runs on genuinely ambiguous candidate sets
- **Training workflow**:
  1. Generate training data locally on 2080 Ti (distillation from BGE-reranker-large or equivalent teacher). Run overnight.
  2. Upload triples to Google Drive.
  3. Train adaptive-depth reranker on Colab Pro+ A100 with checkpointing every 30 minutes.
  4. Budget 5-15 training runs for hyperparameter search.
  5. Download final weights. Export to ONNX on 2080 Ti.
  6. Validate against eval harness. Commit only if quality gate passes.
- **Success metrics**:
  - nDCG@10 gain `>= +2.0` on semantic-hard ambiguous queries
  - No Recall@5 regression larger than `0.005`
  - p95 latency overhead `<= 30ms` when the reranker activates
  - Model runs on CPU in the production binary (ONNX). If only A100 can run it, it does not ship.

**Required tests**:

- `test_training_export_emits_real_idf_df_features`
- `test_training_export_preserves_content_type`
- `test_corrected_export_baseline_matches_or_beats_old_reranker`
- `test_adaptive_depth_halting_on_stable_distribution`
- `test_adaptive_depth_respects_max_iterations`
- `test_reranker_only_activates_on_ambiguous_candidate_sets`

**Commit criteria**:

- All tests pass
- Corrected-export baseline numbers are committed before adaptive-depth numbers
- Adaptive-depth benchmark includes nDCG@10, Recall@5, and p95 overhead
- Training artifacts (seed, dataset version, hyperparameters, Colab notebook snapshot) committed under `docs/training/batch5/`
- No top-level benchmark regression

**Kill criteria**:

- If the corrected-export baseline does not match or beat the old reranker, stop and fix data/features before any architecture work
- If adaptive-depth fails to beat the corrected-export baseline by at least `+1.0` nDCG@10 after two tuning cycles, archive it for post-v1
- If adaptive-depth regresses Recall@5 or exceeds the p95 overhead budget, do not ship it
- If the trained model cannot run on CPU within the latency budget, do not ship it

---

### Batch 6 — Sequential Carry Baseline → Memory Model (Conditional)

**Goal**: Handle cases where relevance depends on evidence spread across distant windows.

**Primary miss category**: **Cross-window dependency misses** — independent windows cannot see the evidence chain needed for correct ranking.

**Training compute**: Baseline work is CPU-only on 2080 Ti. If the neural memory model is justified, training happens on Colab Pro+ following the Section 4.5 workflow.

**Research framing**:

- ExactSDM-motivated primary
- Memory Caching / Sparse Selective Caching as the conditional neural architecture
- Not a committed production architecture unless it beats a much simpler baseline

**Tasks**:

1. **Build a long-document benchmark.**
   - Public, reproducible, committed under `docs/benchmarks/longdoc/`
   - Source:
     - selected public OSS files from the same Python and Java repositories already used for benchmark sourcing
     - files must be `> 8 KB` or span at least `20` code blocks
   - Construction:
     - 50 episodes minimum
     - query text from docstrings/comments or curated NL rewrites
     - judged relevant span must require evidence at least `3` window strides apart

2. **Implement the sequential carry-feature baseline first.**
   - Baseline features:
     ```rust
     pub struct SequentialCarryFeatures {
         pub anchor_hits_prefix: u16,
         pub unique_anchor_groups_seen: u8,
         pub prior_window_max_score_q8: u8,
         pub ordered_anchor_pairs_prefix: u16,
         pub bytes_since_last_anchor_bucket: u8,
     }
     ```

3. **Proceed to a neural memory model only if the baseline fails the target.**
   - If baseline misses target, build Sparse Selective Caching reranker
   - Cache hidden states at segment boundaries
   - Router selects top-k relevant prior memories for the current segment
   - Training workflow: same as Batch 5 (local data generation, Colab training, local validation)

**Success metrics**:

- Carry baseline or memory model improves long-document nDCG@10 by `>= +1.5`
- Added p95 latency overhead is `< 20%` on top-k `<= 20`

**Required tests**:

- `test_longdoc_benchmark_builds_and_replays`
- `test_sequential_carry_features_accumulate_monotonically`
- `test_sequential_carry_baseline_respects_latency_budget`
- `test_memory_router_selects_topk_prior_states`
- `test_memory_model_falls_back_when_no_prior_state_available`

**Commit criteria**:

- All tests pass
- Long-document benchmark artifacts are committed
- Baseline-versus-memory decision is documented explicitly
- Added latency is measured and reported
- If neural memory model is trained, training artifacts committed under `docs/training/batch6/`

**Kill criteria**:

- If the carry-feature baseline already meets the success metric, stop and do not build the neural memory model in this cycle
- If the neural memory model fails to beat the carry baseline by the target margin under the latency budget, archive it

---

### Batch 7 — `sieve train calibrate`

**Goal**: Let users adapt Sieve to their own corpora without expanding into full model training.

**Primary miss category**: **Corpus adaptation gap** — stock priors and surfaces underperform on unfamiliar private corpora.

**Training compute**: None. Calibration is deterministic corpus statistics, not model training. Runs on user's local hardware.

**Tasks**:

1. **Corpus scan and inventory.**
   - content-type estimation
   - corpus-size summary
   - top identifier and surface statistics

2. **Corpus-level calibration.**
   - DF prior estimation
   - code/prose prior separation
   - surface lexicon calibration for corpus-specific identifiers

3. **Packaging and versioning.**
   - write versioned calibration bundle
   - optional ONNX packaging of existing model config
   - no tokenizer replacement, DAPT, or hard-negative mining

**Success metrics**:

- On at least one third-party or held-out heterogeneous corpus:
  - scan-only Recall@5 improves by `>= +0.03`, **or**
  - events/query drops by `>= 20%` at equal Recall@5
- Calibration completes in:
  - `<= 30 minutes` on CPU-only commodity hardware for a `100k` chunk corpus, **or**
  - `<= 15 minutes` on a single consumer GPU if used
- Calibration bundle round-trips cleanly through export/import

**Required tests**:

- `test_sieve_train_calibrate_produces_df_priors`
- `test_sieve_train_calibrate_produces_surface_lexicon`
- `test_sieve_train_calibrate_writes_versioned_bundle`
- `test_calibration_bundle_round_trip`
- `test_calibrate_respects_phase_7_scope_limits`

**Commit criteria**:

- All tests pass
- Calibration improves a real held-out corpus or yields the target efficiency gain
- Bundle format is versioned and documented
- No training features beyond the declared Phase 7 scope are added

**Kill criteria**:

- If calibration fails to achieve either the Recall@5 or efficiency target on a real held-out corpus, stop at DF/surface calibration and do not expand into DAPT or fine-tuning
- If calibration runtime exceeds the budget on the target corpus size, treat it as non-release-ready and archive for post-v1

---

### Batch 8 — Release

**Goal**: Produce the external artifacts needed for v1.0: integration surface, docs, benchmark report, paper, demo, and publishable release.

**Primary miss category**: **External usability gap** — the engine may work, but third parties cannot install it, call it, or trust its claims.

**Training compute**: None. Release work.

**Tasks**:

1. **MCP server.**
   - stdio transport
   - tools:
     - `sieve.search(query, deadline_ms, min_semantic_coverage, provisional_ok)`
     - `sieve.index(path)`
     - `sieve.status()`
     - `sieve.watch(path)`

2. **Production README.**
   - replace stale Phase 0 scaffold text
   - include:
     - what Sieve does
     - install
     - usage
     - architecture overview
     - benchmark results
     - ablation-backed claims
     - link to this build spec

3. **Release benchmark report.**
   - rerun the full Batch 0 benchmark protocol on the final candidate
   - publish per-track tables, CIs, p-values, latency, and miss-category summaries

4. **Paper draft.**
   - working title:
     - *Query-Compiled Learned Sparse Retrieval over Raw Bytes: Late Interaction over Transient Match Events with Zero Document-Side Preprocessing*

5. **Demo recording.**
   - terminal-first
   - shows T+0 semantic lift on a just-written file
   - includes `--explain` provenance so the scan path is visible on screen

6. **crates publish.**
   - `cargo publish --dry-run` first
   - full crates publish only after all release checks pass
   - GitHub release with benchmark report, paper draft, demo recording, and versioned binaries

**Success metrics**:

- Third party can `cargo install sieve`, index a repo, run semantic search, and reproduce benchmark-backed claims from the README
- MCP integrates with at least one agent client using the provisional/final result contract
- Release benchmark report is committed and consistent with the shipped binary
- v1.0 release artifacts are complete

**Required tests**:

- `test_mcp_server_search_tool`
- `test_mcp_server_provisional_response`
- `test_readme_examples_smoke_test`
- `test_cargo_publish_dry_run`
- `test_release_benchmark_manifest_matches_binary_version`

**Commit criteria**:

- All tests pass
- README, benchmark report, paper draft, and demo are committed
- crates publish succeeds
- GitHub v1.0 release is created

-----

## 4. Cross-Cutting Architecture

### 4.1 Benchmark protocol (D0 / D_fresh formalization)

- D0 = base corpus, processed fully before evaluation (all systems get unlimited prep)
- B1…Bn = append-only fresh batches
- q_i = query whose judged relevant document first appears in B_i
- T+0 rule: append B_i, fsync, immediately issue q_i, forbid any additional preprocessing before the T+0 measurement
- T+steady = final deadline after all background jobs complete (`>= 30s`)
- Report `fresh_recall@5(T+0)`, `fresh_recall@5(T+500ms)`, `fresh_recall@5(T+steady)`, `p50/p95` latency separately

### 4.2 Eligibility rules

| System | T+0 eligible? |
| --- | --- |
| Sieve scan | Yes (WAL commit is commit point) |
| Sieve dense path | No (requires embedding) |
| Sieve SPLADE→Tantivy | No (requires index insertion) |
| ripgrep | Yes |
| Dense ANN | No |
| SPLADE + inverted index | No |
| BM25 (Tantivy/Lucene) | No |

### 4.3 Metrics hierarchy

**Primary (report every time)**:

- Recall@5 at T+0, T+500ms, T+steady
- p50/p95 query latency
- Semantic lift over ripgrep at T+0
- ZeroPrepRetention@5

**Secondary (report for deep dives)**:

- Recall@5 vs events/query Pareto frontier
- Recall@5 vs p95 latency Pareto frontier
- `scan_efficiency_ratio` (demoted from V2 primary to secondary)
- Miss categorization: `DLatent` vs `DHard` vs `Other`

### 4.4 Semantic-hard subset

Curated queries where ripgrep Recall@5 `<= 0.29`. Use the curated rewrite logic from commit `4ed6d6b` lineage. Do not use the earlier failed auto-filter. Batch 0 expands this subset across both primary tracks; the curation rule stays the same.

### 4.5 Training Infrastructure and Compute Strategy

**Principle**: Sieve is a local-first product. The trained model artifact ships with the binary. Training compute is rented from the cloud when needed. No cloud inference, no user data leaves the user's machine.

#### 4.5.1 Compute split

| Workload | Where | Hardware |
| --- | --- | --- |
| All Sieve code and testing | Local | RTX 2080 Ti, i9, 64GB DDR4 |
| Eval harness runs | Local | 2080 Ti |
| Teacher model inference for distillation | Local | 2080 Ti, overnight unattended |
| ONNX export and local benchmarking | Local | 2080 Ti |
| Adaptive-depth reranker training (Batch 5) | Rented | Colab Pro+ A100 |
| Sequential memory model training (Batch 6, if justified) | Rented | Colab Pro+ A100 |
| Lexicalizer training (Batch 2, if justified) | Rented | Colab Pro+ A100 |
| Production inference | User's machine | User's CPU / GPU |

#### 4.5.2 When training enters the build sequence

- **Batches 0, 1, 3, 4, 7, 8**: No GPU training. 2080 Ti handles everything. Do not start Colab subscription.
- **Batch 2**: Conditional. Only if the lexicalizer justification threshold is met.
- **Batch 5**: First required training run. Colab Pro+ starts here.
- **Batch 6**: Conditional. Only if the carry-feature baseline fails and neural memory model is needed.

#### 4.5.3 Standard training workflow

Every training run, without exception, follows this sequence:

1. **Local data generation (2080 Ti, overnight).**
   - Select teacher model for the task.
   - Run teacher over eval corpus against eval queries.
   - Save training triples to disk in JSONL format.
   - Expected time: 8-24 hours unattended.
   - Output: ~50K-200K labeled triples.

2. **Data upload.**
   - Push triples to Google Drive or S3 bucket.
   - Dataset size: typically ~500MB-2GB.

3. **Colab training (rented A100).**
   - Clone Sieve training code from GitHub inside Colab notebook.
   - Mount Drive, load dataset.
   - Train with checkpointing every 30 minutes to Drive.
   - Typical run: 4-8 hours on A100.
   - Budget 5-15 training runs for hyperparameter search per batch.

4. **Local validation (2080 Ti).**
   - Download final weights.
   - Export to ONNX.
   - Run eval harness against prior batch baseline.
   - Must hit the batch quality gate before merge.

5. **Commit training artifacts.**
   - Seed, dataset version, hyperparameters, Colab notebook snapshot under `docs/training/batch{N}/`.

#### 4.5.4 Training schedule

- Overnight jobs preferred. Local teacher inference and Colab training both run unattended 10pm-6am.
- Colab A100 availability is best during off-peak US hours (late night / early morning). Queues clear fast.
- Synchronous "babysit the training run" is not required. Checkpoint-and-resume handles disconnects.

#### 4.5.5 Cost budget

| Item | Cost |
| --- | --- |
| Colab Pro+ | $50/month, starting at Batch 5 |
| Estimated duration | 2-3 months through Batch 6 |
| Subtotal Colab | $100-150 |
| Vast.ai fallback (optional) | $50-100 |
| Third-party API costs for distillation (optional) | $0-200 |
| **Total training spend through v1.0** | **~$150-450** |

#### 4.5.6 Fallback compute path

If Colab Pro+ becomes unavailable or unreliable:

- Vast.ai on-demand A100: $1-1.50/hour, no queue.
- RunPod Community Cloud: similar pricing, slightly more polished.
- Same workflow applies — upload data, train, download weights.

Hermes must record which provider was used for each training run in `docs/training/batch{N}/`.

#### 4.5.7 Hard rules

- Never train on user data without explicit user opt-in.
- Training artifacts (weights, logs) must be reproducible from a seed and dataset version.
- All trained models export to ONNX for local CPU inference.
- No training run longer than 24 hours on a single session. Checkpoint and resume.
- Eval gate must pass on local 2080 Ti hardware — if the model only works on A100, it does not ship.
- No cloud inference path. Sieve is local-first.

#### 4.5.8 What NOT to do

- Do not train on the 2080 Ti for Batches 5 or 6 neural work. 10-20x slower, memory-limited, hurts final quality.
- Do not rent cloud GPUs for development or inference testing. Waste of money.
- Do not build any cloud inference path. Sieve is local-first; cloud would be architectural drift.
- Do not train Batches 0, 1, 3, 4, 7, or 8 on Colab. Those batches do not need training.
- Do not start Colab subscription before Batch 5 (or conditional Batch 2 if triggered).

#### 4.5.9 When to stop and confirm with Carson

Hermes must stop and ask Carson to confirm Colab setup is ready before starting the first training run in Batch 5 (or conditional Batch 2). Specifically:

- Colab Pro+ subscription active
- Google Drive mounted with adequate space
- Local teacher inference data generated and uploaded
- Training script committed and tested on a small subset locally

Do not begin the full training run unattended without this confirmation.

-----

## 5. Living Checklist

Hermes updates this after every commit. Format: `- [x] task (commit hash)` for done, `- [ ] task` for pending, `- [!] task (reason)` for blocked or reverted.

### Batch 0 — Benchmark Expansion

- [ ] Expand benchmark to `n_stable=1000`, `n_fresh=200`, `5` runs
- [ ] Add `codesearchnet-java` track
- [ ] Emit Wilson CIs for single-system Recall@5
- [ ] Emit paired-bootstrap CIs and exact McNemar for primary comparisons
- [ ] Add miss-schema and evidence plumbing
- [ ] Standardize p50/p95 latency reporting
- [ ] Batch 0 benchmark artifacts committed
- [ ] Batch 0 commit + push

### Batch 1 — Foundation Quality

- [ ] SPLADE-Code real ONNX integration
- [ ] Code-like query routing to SPLADE-Code
- [ ] Generic SPLADE fallback with warning
- [ ] Corpus-aware DF from Tantivy into scan scorer
- [ ] `CODE_DF_PRIOR` table
- [ ] Indexed sparse-path anchor repair
- [ ] Batch 1 benchmark rerun with Task A/B/C metrics
- [ ] Batch 1 commit + push

### Batch 2 — Taxonomy → Rescan → Lexicalizer (Conditional)

- [ ] Categorize semantic-hard misses as `DLatent` / `DHard` / `Other`
- [ ] Add bounded one-iteration feedback rescan
- [ ] Measure post-rescan `DLatent` share
- [ ] Decision: lexicalizer justified or not
- [ ] (If justified) Colab Pro+ subscription active, confirmed with Carson
- [ ] (If justified) local data generation complete, uploaded to Drive
- [ ] (If justified) train learned lexicalizer on Colab A100
- [ ] (If justified) validate on 2080 Ti eval harness
- [ ] (If justified) training artifacts committed under `docs/training/batch2/`
- [ ] Batch 2 benchmark rerun with pre/post-rescan and lexicalizer metrics
- [ ] Batch 2 commit + push

### Batch 3 — Fusion Refinement

- [ ] Deadline-aware fusion
- [ ] File-level fusion before window selection
- [ ] Explicit fusion dedup semantics
- [ ] Calibrate planner heuristics and dense-budget policies
- [ ] Provisional/final result contract
- [ ] Revisit semantic-hard T+steady regression
- [ ] Batch 3 benchmark rerun
- [ ] Batch 3 commit + push

### Batch 4 — Ops Hardening

- [ ] Semantic debt scheduler (three-queue)
- [ ] Content-hash embedding cache
- [ ] Versioned cancellation for stale jobs
- [ ] `sieve watch` command
- [ ] WAL graduation with `indexed_generation`
- [ ] WAL compaction policy
- [ ] SIMD backend evaluation and decision record
- [ ] Bulk-write stress test
- [ ] Batch 4 commit + push

### Batch 5 — Reranker Data Hygiene + Adaptive-Depth Reranker

- [ ] Fix `training_export.rs` to emit real IDF/DF features
- [ ] Preserve content type in reranker export
- [ ] Retrain corrected-export baseline reranker
- [ ] Decision: corrected baseline acceptable
- [ ] Colab Pro+ subscription active, confirmed with Carson
- [ ] Local distillation data generation complete (overnight on 2080 Ti)
- [ ] Data uploaded to Drive
- [ ] Shared-block adaptive-depth reranker trained on Colab A100
- [ ] ONNX export validated on 2080 Ti
- [ ] Candidate-set-level halting rule
- [ ] Adaptive-depth benchmark on ambiguous queries
- [ ] Training artifacts committed under `docs/training/batch5/`
- [ ] Batch 5 commit + push

### Batch 6 — Sequential Carry Baseline → Memory Model (Conditional)

- [ ] Build long-document benchmark
- [ ] Implement sequential carry-feature baseline
- [ ] Evaluate baseline against exit criteria
- [ ] Decision: proceed to neural memory or stop
- [ ] (If proceed) local data generation complete, uploaded to Drive
- [ ] (If proceed) Sparse Selective Caching reranker trained on Colab A100
- [ ] (If proceed) ONNX export validated on 2080 Ti
- [ ] (If proceed) training artifacts committed under `docs/training/batch6/`
- [ ] Batch 6 commit + push

### Batch 7 — `sieve train calibrate`

- [ ] Corpus scan and inventory
- [ ] DF prior estimation
- [ ] Surface lexicon calibration
- [ ] Versioned calibration bundle
- [ ] Held-out corpus evaluation
- [ ] Batch 7 commit + push

### Batch 8 — Release

- [ ] MCP server with four tools
- [ ] Production README
- [ ] Final benchmark report
- [ ] Paper draft
- [ ] Demo recording
- [ ] crates publish dry-run
- [ ] crates publish
- [ ] GitHub v1.0 release

-----

## 6. Hermes Directive

For all future work on Sieve, Hermes uses only:

- `SIEVE-BUILD-SPEC-V3.1.md` for scope, build order, and success criteria
- Repo code on `main` as ground truth for what currently exists
- Benchmark JSON artifacts in `docs/` as ground truth for quality numbers
- Git log for provenance and decision timing

Hermes ignores as scope sources:

- `SIEVE-BUILD-SPEC-V3.md`
- `RESEARCH-CONTRACT-V2.pdf`
- `SIEVE-FUTURE-SPEC.md`
- `SIEVE-FUTURE-SPEC-REVISION-NOTES.md`
- `README.md` on `main` for architecture state
- `spec-v0.3.md`
- Any other prior spec document

Hermes maintains the checklist in Section 5 after every commit. Completed items get `[x]` with commit hash. Blocked or reverted items get `[!]` with a brief reason.

Hermes does not add tasks to this spec without explicit approval. If Hermes identifies missing work, Hermes raises a proposal first. Scope creep kills projects.

Hermes follows Section 4.5 (training infrastructure) strictly. No GPU training before Batch 5 (or conditional Batch 2). No cloud inference path. All trained models validate on 2080 Ti before commit.

-----

## 7. What's Different

### 7.1 vs `RESEARCH-CONTRACT-V2.pdf`

V2 was a proof contract: establish whether the T+0 claim is real. That objective is complete. V3.1 starts from a proven primitive and scopes the path to the best pre-release system.

### 7.2 vs `SIEVE-FUTURE-SPEC.md` and revision notes

V3.1 keeps only the future-spec items that are on the measured bottleneck path. Anything not tied to the current miss categories is out of active scope.

### 7.3 vs `SIEVE-BUILD-SPEC-V3.md`

V3.1 changes the build order materially:

- adds **Batch 0** for benchmark power and breadth
- splits **Batch 1** into task-specific failure modes and metrics
- adds **indexed sparse-path repair** to the foundation batch
- reorders **Batch 2** so miss taxonomy comes before lexicalizer work
- moves **fusion refinement ahead of ops hardening**
- adds **reranker data hygiene** as a blocking prerequisite before adaptive-depth
- splits **local adaptation** from **release/distribution**
- adds **Section 4.5 training infrastructure policy** so Hermes knows when and how to use Colab for GPU training

### 7.4 Scope philosophy

V3.1 is stricter than V3. It treats benchmark breadth, batch-local success metrics, and kill criteria as first-class controls on scope. Learned components are conditional on measured remaining miss mass, not on intuition.

### 7.5 Why V3 was superseded within 24 hours

V3 got the broad direction right but had seven concrete planning errors:

1. the benchmark was still too small and too single-track to justify heavy model work
2. Batch 1 mixed candidate-generation and scoring tasks under one miss category
3. indexed sparse-path repair was missing from the early quality path
4. Batch 2 tried to build the lexicalizer before measuring whether `DLatent` misses justified it
5. ops hardening was placed ahead of fusion quality work
6. reranker architecture work was planned before fixing dirty training export
7. adaptation and distribution work were still coupled

V3.1 fixes those directly. It does not change the thesis. It removes planning error from the path to the best possible system.

### 7.6 What Section 4.5 adds

V3.1 formally separates local development compute (2080 Ti) from training compute (Colab Pro+ A100). This matters because:

- Training a transformer-based reranker on an 11GB consumer GPU is 10-20x slower than on A100
- Batch size compromises on 2080 Ti hurt final model quality
- Colab Pro+ at $50/month closes the gap without owning hardware
- Sieve's local-first promise is preserved: training happens in the cloud, inference stays local

Hermes now knows exactly when to trigger Colab work (Batch 5, conditional Batch 2, conditional Batch 6) and never before.

-----

## 8. Decision Log

Major architectural and planning decisions preserved for context:

- **Coverage-aware fusion shipped**: weighted RRF scales by each layer's coverage fraction. Complete / Partial / Unavailable coverage states remain the active fusion base.
- **Query-promoted dense shipped**: Layer 1b is now first-class freshness completion, not a hidden fallback.
- **Weights do not matter; term selection does**: the random-weight ablation held. This locks focus on term emission and lexicalization, not score tinkering.
- **Semantic-hard T+steady regression deferred in V3**: V3.1 moves fusion ahead of ops to fix this on the correct path.
- **Benchmark underpower critique accepted**: resolved by adding Batch 0 with `n_stable=1000`, `n_fresh=200`, `5` runs, Wilson + paired-bootstrap + exact McNemar, and one additional Java track.
- **Batch 1 mixed-failure critique accepted**: resolved by splitting Batch 1 into Task A term emission, Task B scoring/efficiency, and Task C indexed sparse repair, each with its own success metric.
- **Indexed sparse-path repair critique accepted**: resolved by adding anchor-gate softening to Batch 1 with a direct `splade-bm25` target.
- **Batch 2 ordering critique accepted**: resolved by requiring miss categorization first, then bounded rescan, then a conditional lexicalizer only if `DLatent > 30%` of remaining misses.
- **Fusion-before-ops critique accepted**: resolved by swapping the V3 Batch 3 and Batch 4 ordering.
- **Reranker data hygiene critique accepted**: resolved by making export cleanup and corrected-baseline retraining a blocking prerequisite in Batch 5.
- **Adaptation-vs-distribution split critique accepted**: resolved by isolating `sieve train calibrate` in Batch 7 and moving MCP / README / crates publish into Batch 8.
- **`sieve watch` placement decision**: kept in ops hardening, not release, because it is the primary real-world churn surface for validating scheduler, cancellation, and WAL correctness.
- **Training compute policy added (Section 4.5)**: resolved by adopting Colab Pro+ for Batch 5+ neural training, preserving local-first production inference. 2080 Ti remains development and eval hardware; A100 is rented only when required.
- **Training schedule preference**: overnight jobs preferred. Carson works at night, so 2080 Ti teacher inference runs late evening and Colab training runs overnight.
- **No cloud inference path**: production inference stays local on the user's machine. Trained models ship as ONNX artifacts in or alongside the binary.
- **V3 archived**: Hermes uses V3.1 only. Prior specs remain historical reference, not scope authority.
