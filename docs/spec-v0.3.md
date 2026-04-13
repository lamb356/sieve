# SIEVE — Semantic Grep Engine

## System Design Specification v0.3 (FROZEN)

One-liner: ripgrep meets vector search. Local-first, single-binary, zero-wait hybrid retrieval.

Product thesis: Every existing solution forces a choice: grep (instant, no index, no semantics) or search engine (semantic, requires indexing pipeline, slow to set up). Sieve eliminates the choice. Drop files in, search immediately — exact match AND meaning — with background optimization that makes repeated queries faster over time.

---

## 1. Problem Statement

The tweet that started this: *“whoever invents realtime vector/fulltextsearch without needing a pre-processing step, just like grep, is going to become a billionaire”*

### What exists today

|Tool           |Instant?|Semantic?|Scales?|Local?   |
|---------------|--------|---------|-------|---------|
|ripgrep        |✓       |✗        |~10GB  |✓        |
|Zoekt          |minutes |✗        |✓      |✓        |
|Turbopuffer    |~1s WAL |✓        |✓      |✗ (cloud)|
|Qdrant/Pinecone|seconds |✓        |✓      |✗        |
|Tantivy        |seconds |✗        |✓      |✓        |
|Vespa streaming|✓ (scan)|✓ (exact)|bounded|✗        |

Gap: No single-binary, local-first tool gives you both grep-speed freshness AND semantic retrieval.

### Why this is a real product

1. Cursor uses turbopuffer for exactly this — each codebase = namespace, semantic search on code. But it’s a cloud dependency.
2. Every RAG pipeline needs ingest → chunk → embed → index → query. That pipeline is the bottleneck Sieve eliminates.
3. Code search (Sourcegraph/Zoekt) proves the market. Adding semantic on top of lexical is the obvious next step.
4. Personal knowledge bases (Obsidian, logseq, notes) — users want “find the thing I wrote about X” without setting up Elasticsearch.

---

## 2. Architecture

### Core Invariant

Every document is immediately lexically searchable on write. Semantic retrieval is immediately available through hot embeddings when present, with bounded fallback scoring over the small unembedded delta.

No user-visible indexing step. No “wait for processing.” You write a file, you search it. Lexical results are instant. Semantic results are instant in the common case (eager embedding at ~200ms/file). For the narrow transient window where content exists but hasn’t been embedded yet, Sieve performs query-time semantic scoring over only the unembedded delta — strictly budgeted by chunk count and latency target.

This is achieved by a 4-layer tiered architecture (plus a bounded fallback path) where fresh data is always scanned directly, and background processes build optimized indexes for repeated query performance.

### Layer 0: Hot Raw Scan
Purpose: Grep-like instant visibility for all fresh data.

Mechanism:
- New files/content appended to an append-only WAL (write-ahead log)
- On query: mmap the WAL segments, scan with SIMD-accelerated matchers
- Supports: exact match, substring, regex, glob patterns
- This is essentially ripgrep over the hot buffer

Implementation:
- Use memchr crate (same as ripgrep) for SIMD byte scanning
- memmap2 for zero-copy file access
- Aho-Corasick for multi-pattern matching
- regex-automata for regex

Performance target: 1GB/s scan rate on modern hardware (ripgrep baseline)

When data leaves this layer: After Layer 2 lexical fragments are built for the same content, raw scan is no longer needed for lexical queries. But raw scan remains available as fallback.

### Layer 1: Hot Exact Vector Tier

Purpose: Immediate semantic search on fresh data without waiting for ANN index construction.

Mechanism:
- On write (or on first query if lazy): compute embeddings for new content
- Append embedding vectors to a flat, append-only vector store
- On query: embed the query, brute-force exact kNN over all hot vectors
- No HNSW, no IVF, no clustering — just dot product / cosine over flat array

Implementation:
- Embedding model: ONNX Runtime with a small, fast model
  - Primary: bge-small-en-v1.5 (384-dim, ~30ms/chunk on CPU)
  - Future: nomic-embed-text-v1.5 (768-dim, better quality)
  - Stretch: SPLADE sparse expansion for hybrid sparse+dense
- Vector storage: flat f32 array, mmap’d
- Distance computation: SIMD-accelerated dot product
  - simsimd crate or hand-rolled AVX2/NEON kernels
- Chunking: pluggable chunker interface (see below)

Chunking architecture (pluggable from day one):
```rust
pub trait Chunker: Send + Sync {
    fn chunk(&self, content: &str, path: &Path) -> Vec<Chunk>;
}

pub struct Chunk {
    pub text: String,
    pub byte_offset: usize,
    pub line_range: (usize, usize),
    pub metadata: ChunkMetadata,
}
```

Three implementations ship:

|Chunker         |Trigger                 |Strategy                                                                       |
|----------------|------------------------|-------------------------------------------------------------------------------|
|`ProseChunker`  |.md, .txt, .rst, .tex   |Paragraph boundaries, 512-token max, sentence-aware splits                     |
|`CodeChunker`   |.rs, .py, .ts, .go, etc.|Function/class/module boundaries via tree-sitter, fallback to blank-line splits|
|`SlidingChunker`|fallback / unknown      |Fixed 512-token window, 128-token overlap                                      |

Auto-detection by file extension. Override via config. CodeChunker uses tree-sitter grammars to find natural boundaries — functions, classes, impl blocks. This produces much better retrieval quality for code than naive sliding windows.

Performance target:
- Embedding: <100ms per chunk (CPU), <10ms (GPU)
- kNN over 10K vectors (384-dim): <1ms with SIMD
- kNN over 100K vectors: <10ms
- kNN over 1M vectors: ~100ms (this is where you want Layer 3)

Critical design decision: Embeddings are computed eagerly on write by default (configurable to lazy). This is the “unavoidable preprocessing” — but it happens inline, not as a batch job. The user never waits.

### Layer 1b: Unembedded Delta Fallback

Purpose: Semantic coverage for the transient window where content is committed but not yet embedded.

When this activates:
- Lazy embedding mode (bulk ingestion of thousands of files)
- System under load (embedding backpressure)
- Model not yet downloaded (first run, before model cache is populated)

Mechanism:
- Track which WAL entries have been embedded and which haven’t
- On semantic query: identify the unembedded delta set
- If delta is small (< MAX_FALLBACK_CHUNKS, default 50):
  - Embed query via ONNX
  - Embed delta chunks on the fly
  - Score via exact cosine similarity
  - Merge results into main RRF pipeline
- If delta is large (> `MAX_FALLBACK_CHUNKS`):
  - Return lexical-only results for unembedded content
  - Stream semantic results as background embedding catches up
  - Show progress indicator: “semantic coverage: 847/1000 chunks”

Latency budget (adaptive, not fixed):
The fallback stops when ANY of these conditions is hit:
- Chunk count exceeds MAX_FALLBACK_CHUNKS (default 50)
- Estimated fallback latency exceeds MAX_FALLBACK_MS (default 1500ms)
- System CPU load exceeds threshold (avoid starving other layers)

Beyond the budget, degrade to lexical + streaming semantic results.

Implementation:
- Bitmap tracking: `embedded_set: RoaringBitmap` over WAL entry IDs
- Delta = WAL entries NOT IN embedded_set
- Query-time embedding uses same ONNX session (already loaded for query embedding)
- Results tagged with source: `delta_fallback` for transparency

Key rule: Never make the main query path pay corpus-scale semantic scan costs. The delta fallback is strictly bounded.

### Layer 2: Incremental Lexical Fragments

Purpose: Move from O(n) raw scan to O(log n) lexical lookup for repeated queries on stable content.

Mechanism:
- Background thread builds small, immutable lexical index segments from WAL content
- Each segment is self-contained: posting lists + term dictionary + stored content hashes
- Segments are published atomically (rename into active set)
- Query planner unions results across all active segments + hot scan layer

Implementation options (pick one for MVP):

Option A: Tantivy-backed lexical shards (recommended for MVP)
- Sieve manages immutable lexical shards, each backed by a Tantivy index
- Sieve owns the shard lifecycle (create, publish, merge, prune)
- Tantivy provides: BM25 scoring, phrase queries, concurrent readers
- We do NOT use Tantivy’s built-in IndexWriter merge policy — we manage merges ourselves
- This avoids fighting Tantivy’s internal segment model

Option B: Custom trigram index (higher ceiling, more work)
- Build trigram posting lists from content (like Zoekt)
- Verify trigram candidates against actual content
- Excellent for code search (handles CamelCase, snake_case, partial matches)
- More control over the format, but 2–3 weeks more work

Recommendation: Start with Tantivy. Switch to custom trigrams later if code search becomes the primary use case.

### Layer 3: Warm Compacted Index

Purpose: Low-latency search at scale for stable data.

Mechanism:
- Periodically, background compactor merges:
  - Multiple lexical segments → fewer, larger segments
  - Hot flat vectors → ANN structure (HNSW or centroid-based)
- ANN index provides sub-linear vector search for large corpora

Implementation:
- Lexical: Sieve-managed shard merge using Tantivy’s low-level segment facilities under our own orchestration
- Vector ANN: `hnsw` crate, or `hnswlib-rs`, or custom centroid index
  - Centroid-based (like turbopuffer/SPFresh) is better for append-heavy workloads
  - HNSW is better for query-heavy workloads with rare updates
- For MVP: skip Layer 3 entirely. Flat exact kNN is fine up to ~100K–500K chunks.

When to build Layer 3: Only when the hot vector tier exceeds the scan budget (>100ms). For a personal knowledge base of 10K files, this may never be needed.

### Query Planner

Purpose: Route queries, fan out to layers, fuse results.

Query flow:
1. Parse query
   - Detect query type: exact match? regex? natural language?
   - If regex/exact: skip semantic layers, use Layer 0 + Layer 2
   - If natural language: use all layers

2. Execute in parallel:
   a. Layer 0: raw scan for exact/substring matches
   b. Layer 1: embed query → flat kNN over hot vectors
   c. Layer 1b: if unembedded delta exists and is within budget, embed delta chunks → score against query
   d. Layer 2: BM25/trigram lookup in lexical segments
   e. Layer 3: ANN lookup in warm index (if exists)

3. Fuse results:
   - Weighted Reciprocal Rank Fusion (RRF) across all result sets
   - RRF_K = 20
   - Deduplicate results within the same WAL entry when byte-range IoU > 0.5
   - If delta was too large for 1b: annotate results with coverage %

4. Return top-K results with:
   - Source file path
   - Line range / byte offset
   - Match snippet with context
   - Score breakdown (lexical score, semantic score, raw match)
   - Semantic coverage indicator (if < 100%)

### Operational Semantics

Write lifecycle:
1. File content read + chunked
2. WAL append (this is the durability/visibility commit point)
3. Lexical visibility: GUARANTEED from this moment
4. Eager embedding: scheduled immediately post-commit (best-effort, not prerequisite)
5. If embedding completes before next query → semantic via Layer 1
6. If not → semantic via Layer 1b (bounded delta fallback)
7. Lexical fragments + warm index structures are background optimizations only

Key rule: WAL append is the commit point. Eager embedding is best-effort immediate post-commit work, not a prerequisite for visibility. The write call returns after WAL append, not after embedding.

Query lifecycle:
1. Detect query type (exact/regex/natural language)
2. Lexical coverage: ALWAYS 100% of committed content
3. Semantic coverage: 100% when all content is embedded, otherwise partial
4. If partial: surface coverage indicator (`semantic: 847/1000 chunks`)
5. Optionally stream improving results as background embedding progresses

Semantic freshness target: Sub-250ms additional semantic availability latency for typical small-file writes under eager mode.

---

## 3. Embedding Strategy

This is the crux. The tweet says “no preprocessing” but embeddings ARE preprocessing. Here’s how we make it invisible:

### Eager inline embedding (default mode)

User adds file to index
    → sieve reads file
    → chunks content via selected Chunker
    → appends raw text + chunk metadata to WAL
    → file is now DURABLE and lexically searchable (commit point)
    → eager embedding starts immediately post-commit
    → embeds each chunk via ONNX model (in-process, no network call)
    → appends vectors to hot.vecs
    → file is now semantically searchable via Layer 1
    → total additional semantic availability latency target: <250ms for a typical 10KB file

The key insight: a 10KB file produces ~2–4 chunks. At 30–50ms per chunk on CPU, semantic availability usually arrives within ~200ms after commit. There is no separate user-visible indexing step: lexical visibility is immediate at WAL append, and semantic visibility follows almost immediately in eager mode.

### Lazy embedding (opt-in for large ingestion)
User adds 10,000 files to index
    → sieve reads all files
    → appends raw text to WAL (instant, ~5s for 10K files)
    → ALL files are immediately searchable via Layer 0 (raw scan)
    → ALL files are immediately searchable via Layer 2 (lexical, after segment build)
    → background thread starts embedding
    → as each file is embedded, it becomes searchable via Layer 1 (semantic)
    → progress bar shows embedding progress
    → user can query at any point — gets best available results

---

## 4. User Interface

### CLI (primary interface)
```bash
sieve index ./my-project
sieve search "authentication middleware"
sieve search "/auth.*middleware/"
sieve search '"exact phrase match"'
sieve watch ./my-project
sieve status
```

---

## 10. Success Metrics

|Metric                           |Target                     |
|---------------------------------|---------------------------|
|Cold index + search on 1000 files|< 30 seconds               |
|Query latency (warm, 1000 files) |< 50ms                     |
|Scan throughput                  |> 500MB/s                  |
|Embedding throughput (CPU)       |> 20 chunks/sec            |
|Binary size (CLI, no model)      |< 50MB                     |
|Default model download           |< 150MB (bge-small-en-v1.5)|
|Memory usage (1000 files indexed)|< 200MB                    |

---

*Spec version: 0.3 (FROZEN — build from here)*
*Author: Carson + Claude + Hermes review*
*Date: 2026-04-12*
