# SIEVE Future Spec — Revision Notes After External Critique

## Overall judgment

The critique is mostly correct and worth acting on. The biggest improvements are:

1. Reframe Area 4 as a research branch motivated primarily by ExactSDM-style long-document positional dependence, with MEMENTO retained only as architectural inspiration.
2. Replace hard-coded routing thresholds with a calibrated gating policy.
3. Narrow `sieve train` v1 to calibration-only plus optional small adaptation; defer full DAPT/tokenizer work.
4. Formalize the freshness benchmark as a partitioned `D0` / `D_fresh` protocol with strict `T+0` rules.
5. Move the scan-efficiency ratio to a secondary metric and report Pareto frontiers instead.
6. Add missing operational sections for SIMD scan options, WAL compaction / graduation, and fusion deduplication semantics.

---

## 1. Area 4 should be reframed

### Change
Rename:

- **From:** `Sequential window processing with residual state (MEMENTO-inspired)`
- **To:** `Sequential window processing over ordered match-event windows (ExactSDM-motivated, MEMENTO-inspired)`

### New positioning

- **Primary research basis:** long-document sparse retrieval and ExactSDM / local dependence.
- **Secondary inspiration:** MEMENTO-style compact carry-forward state.
- **Status:** research branch, not committed production architecture.

### Add null hypothesis

> H0: independent window scoring plus simple cross-window aggregate features is sufficient; learned residual state does not improve Recall@10 / nDCG@10 enough to justify sequential cost.

### Add exit criteria

Proceed with learned sequential reranker only if it beats both baselines below on held-out long-document/code tasks:

1. independent window scorer + document coverage features
2. independent window scorer + ordered-anchor carry features

Target gain to continue:

- `+1.5 nDCG@10` or more on long-document/code benchmark, **and**
- `<20%` added p95 rerank latency on top-`k<=20` candidate docs.

### New baseline before neural memory

```rust
pub struct SequentialCarryFeatures {
    pub anchor_hits_prefix: u16,
    pub unique_anchor_groups_seen: u8,
    pub prior_window_max_score_q8: u8,
    pub ordered_anchor_pairs_prefix: u16,
    pub bytes_since_last_anchor_bucket: u8,
}
```

Score each window with the existing model plus these accumulated features before attempting a transformer memory model.

---

## 2. Replace hard-coded routing thresholds with calibration

### Problem
The current example policy uses `0.7 / 0.8 / 0.3` without calibration provenance.

### Revision
State explicitly that these are placeholders and must not ship as literals.

### Replacement plan

1. Train or choose a lightweight query-type classifier returning `P(code_query)`.
2. Calibrate probabilities with temperature scaling or isotonic regression on a held-out query set.
3. Learn routing thresholds from grid search on real workloads.
4. Persist thresholds in model metadata rather than source code.

```rust
pub struct RoutingCalibration {
    pub classifier_version: String,
    pub calibration_method: CalibrationMethod,
    pub threshold_code_only: f32,
    pub threshold_prose_only: f32,
    pub threshold_dual: (f32, f32),
}
```

### Deployment rule

- ship with data-driven thresholds derived from a held-out mixed corpus
- expose override flags for operators
- log routing decisions for later recalibration

---

## 3. Narrow `sieve train` v1

### Revised scope

### Phase 5.0 / v1
Ship only:

- corpus scan
- content-type estimation
- DF prior estimation
- surface lexicon calibration
- optional ONNX packaging of the existing model config

No tokenizer replacement. No full DAPT. No hard-negative mining pipeline.

### Phase 5.1
Optional small adaptation:

- small-model MLM/DAPT
- small sparse fine-tune
- ONNX export

### Phase 5.2+
Stretch:

- tokenizer retraining
- full corpus-adaptive backbone replacement
- large hard-negative refresh pipelines

### CLI framing

```bash
sieve train calibrate   # phase 5.0
sieve train adapt-small # phase 5.1
sieve train adapt-full  # research / advanced
```

---

## 4. Tighten the freshness benchmark protocol

### Add formal definition

Let:

- `D0` = base corpus available before evaluation
- `B1...Bn` = append-only fresh batches
- `q_i` = query whose judged relevant document first appears in batch `B_i`

### `T+0` evaluation rule

At batch commit time `t_i`:

1. append `B_i`
2. fsync / commit
3. immediately issue only queries mapped to `B_i`
4. forbid any additional background preprocessing before the `T+0` measurement
5. record search quality and latency

### Baseline fairness clause

A system is eligible to answer at `T+0` only if every artifact required for semantic retrieval of `B_i` already exists at query time.

- Sieve: eligible at WAL commit
- ripgrep: eligible at file write
- dense ANN: only eligible after embedding + ANN insertion of `B_i`
- Vespa vector path: only eligible after vector field population / tensor indexing

### Report both

- `fresh_recall@k(T+0)`
- `steady_state_recall@k(T+N)` after all background jobs finish

---

## 5. Demote scan-efficiency ratio to a secondary metric

### Keep

```text
scan_efficiency_ratio = Recall@10 / log2(1 + events_per_query)
```

### But report primarily

1. Recall@10 vs events/query Pareto frontier
2. Recall@10 vs p95 latency Pareto frontier
3. events/query vs p95 latency scatter

Use the ratio only as a secondary tie-breaker inside an operating region.

---

## 6. Add a SIMD scan roadmap section

### New section title
`4.1A Optional SIMD literal scan backend`

### Scope

- evaluate Rust `aho-corasick` packed mode for small pattern sets
- evaluate Hyperscan / VectorScan backend for dedicated x86 deployments
- retain classic AC path as portable default

### Success criteria

- no regression in match semantics
- `>=1.3x` speedup for small/medium literal sets on x86
- portable fallback kept as default path

---

## 7. Add WAL graduation / compaction policy

### New section title
`7g. WAL lifecycle, segment graduation, and compaction`

### Required decisions

- when WAL entries become eligible for Tantivy segment build
- how fresh scan excludes graduated entries
- compaction trigger by bytes, count, or age
- crash-safe marker for `graduated_at`

### Suggested metadata

```rust
pub struct WalEntryMeta {
    pub source_path: String,
    pub content_hash: [u8; 32],
    pub byte_offset: u64,
    pub byte_length: u32,
    pub committed_at_unix_ms: u64,
    pub content_type: ContentType,
    pub indexed_generation: Option<u64>,
}
```

### Invariant

Fresh semantic scan covers only entries with `indexed_generation == None`.

---

## 8. Add explicit fusion dedup semantics

### New section title
`Fusion semantics and duplicate suppression`

### Required statement

Before weighted RRF or any fusion score is applied, candidates from all retrieval sources must be deduplicated by stable `result_id` (document/file identity), with per-source ranks retained as evidence.

```rust
pub struct FusedCandidate {
    pub result_id: ResultId,
    pub source_ranks: SmallVec<[(SourceKind, u16); 4]>,
    pub source_scores: SmallVec<[(SourceKind, f32); 4]>,
}
```

This makes overlap between semantic scan and Tantivy lexical retrieval an intended behavior, not an ambiguity.

---

## 9. Recommended wording changes

### Area 4 opening sentence

> This section specifies a research-track reranker, not a committed core-path component. The baseline system remains independent window scoring.

### Area 2 routing note

> Numeric routing thresholds shown below are illustrative placeholders and must be calibrated on a held-out workload before production use.

### Area 3 opening sentence

> `sieve train` v1 is a calibration tool first; full corpus-adaptive model training is a later phase.

### Area 6 freshness note

> The benchmark’s novel contribution is not merely a corpus choice, but a strict `T+0` eligibility protocol that forbids semantic systems from counting documents as searchable until every required retrieval artifact exists.

---

## 10. Area 4 architectural reference swap: MEMENTO → Memory Caching (SSC)

### Change
Replace MEMENTO as the primary architectural reference for Area 4's sequential window reranker. MEMENTO is about KV cache compression during LLM generation — not a retrieval architecture.

### New primary reference
Memory Caching: RNNs with Growing Memory (arxiv.org/abs/2602.24281, Google Research / Cornell / USC). Introduces Sparse Selective Caching (SSC): cache hidden state checkpoints at segment boundaries, use a router to selectively retrieve the most relevant past states at near-constant cost per token.

### Retained secondary reference
ExactSDM remains the retrieval-theory basis for why cross-window dependence matters in long-document sparse retrieval.

### What changes architecturally

Instead of carrying a single residual state forward window by window, the reranker would:

1. Emit a checkpoint at each window boundary: compact key, compact value/state, metadata (byte range, anchor-group counts, content type)
2. When scoring window t, a router conditioned on query + current window state retrieves only top-k prior checkpoints
3. Fuse retrieved checkpoints with local window representation before producing the score

The independent local event encoder stays. The sequential part becomes: checkpoint cache + top-k retrieval + gated fusion — not one persistent hidden state.

### Training pipeline change

Train on ordered window sequences from long documents/code files with:
- Standard ranking loss on final window score
- Auxiliary memory-selection loss teaching the router to retrieve useful prior windows
- Teacher-student: let a full-history model identify which prior windows mattered, train the router to recover them

### Updated success metrics

Three gates:
1. Quality on long-gap cases: +3-5% nDCG@10 on queries whose evidence spans distant windows
2. Memory retrieval quality: top-k hit rate against oracle prior windows
3. Bounded overhead: <25-30% p95 latency increase vs independent scorer

Exit criterion: if these gates are not met, keep the simpler independent scorer plus handcrafted cross-window features.

### MEMENTO disposition
Demote to footnote or remove entirely. It remains an interesting observation about residual information in compressed representations but is not an architecture template for retrieval.

