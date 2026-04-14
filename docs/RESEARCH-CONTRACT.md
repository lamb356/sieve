# RESEARCH CONTRACT V2

## Page 1

SIEVE Research Contract V2
Version: 2026-04-14-v2 Status: Active — all work must serve this document until the core
chart exists Rule: Before starting any task, ask: “Does this make the T+0 claim more true,
more measurable, or more defensible?” If not, it waits.
Status labels used throughout:
 VERIFIED — exists in current repo on main
 IN PROGRESS — Hermes is actively working on this
 SMALL MISSING WORK — not in repo yet, bounded implementation
 FUTURE RESEARCH — out of scope until core chart exists
1. Novelty Claim
Stated narrowly:
Sieve introduces query-compiled learned sparse retrieval over raw bytes: a retrieval
primitive that makes fresh local files semantically searchable at T+0 by compiling learned
sparse query expansions into a multi-pattern byte automaton, scanning untouched file
content, and scoring local match-event windows — without document embeddings,
document sparse vectors, or any document-side preprocessing on the freshness path.
What this is NOT claiming:
Not claiming “first semantic search tool” (many exist)
Not claiming “first hybrid retrieval” (standard technique)
Not claiming “first local search CLI” (ripgrep, Zoekt, etc.)
Not claiming “no preprocessing exists” (embeddings are preprocessing; we do them
optionally in background)
What this IS claiming:
First system where learned sparse expansion is compiled into a raw-byte scan path
rather than matched against stored document representations
First system where semantic retrieval quality is available at T+0 on files that have

## Page 2

undergone zero document-side processing
The retrieval primitive is “late interaction over transient match events” — the events exist
only during the query, are never stored, and require no pre-built index
Why this matters:
Every existing semantic search system requires document-side work before semantic
retrieval is possible: embedding (dense), sparse encoding (SPLADE standard), tokenization
+ index insertion (BM25), or token-level encoding (ColBERT). Sieve’s scan path skips all of
it. The document is raw bytes. The intelligence is entirely query-side.
Repo status: 
 The core primitive exists. The repo has a SPLADE encoder, Aho-Corasick
automaton compilation from expansion terms, post-expansion DF filtering, fresh/indexed
content partitioning, raw-byte semantic scan of fresh WAL entries, and structured window
scoring with coverage, proximity, and ordered-pair features.
2. Benchmark Protocol
2.1 Freshness Protocol (T+0)
Definition of T+0: The moment a file’s bytes are durable on disk (fsync complete). No
additional processing of any kind may occur before the T+0 query.
Formal protocol:
Let D0 = base corpus (may be pre-indexed, pre-embedded, fully processed)
Let B_i = fresh batch i (one or more new files)
Let q_i = query whose relevant answer exists only in B_i
For each episode:
  1. Process D0 fully (all systems get unlimited prep time)
  2. Write B_i to disk, fsync
  3. Start timer
  4. Issue q_i immediately (T+0 measurement)
  5. Record: latency, recall@k, which retrieval path produced the hit
  6. Allow background processing to complete
  7. Issue q_i again (T+steady measurement)
Repo status: 
 The bench harness (sieve-bench) already preprocesses the stable
corpus, appends a fresh file through the WAL, withholds fresh vectors from the embedding
baseline at the zero-deadline query, and clears hits when latency exceeds deadline.
Deadlines at 0ms, 100ms, 500ms, 1s, 5s are implemented.

## Page 3

The harness does not currently have a 30s or explicit “T+steady” pass. Add this as a
final deadline after all background jobs complete.
2.2 System Eligibility Rules
A system may count a document as “searchable” at T+0 ONLY if every artifact required for
retrieval already exists at query time.
System T+0
eligible? Why
Sieve (scan path) Yes Query-compiled, scans raw bytes
Sieve (dense path)No Requires embedding computation
Sieve
(SPLADE→Tantivy) No Requires Tantivy index insertion
ripgrep Yes Scans raw bytes (but no semantics)
Dense ANN (any) No Requires embed + index insert
SPLADE + inverted
index No Requires document sparse encoding + index
insert
BM25 (Tantivy/Lucene)No Requires tokenization + index insert
Repo status: 
 The bench harness enforces this correctly for Sieve and ripgrep. 
 An
embed-knn runner exists. 
 A standalone SPLADE→Tantivy-only benchmark runner does
not exist yet — needed for ablation A3.
2.3 Baselines
Baseline Purpose Repo status
ripgrep Lexical T+0 ceiling
  Runner exists in sieve-bench
SPLADE→Tantivy
(indexed only)
Semantic quality with
preprocessing
  No standalone runner yet
bge-small + flat
kNN
Dense retrieval quality
with preprocessing
  embed-knn runner exists
BM25 via TantivyLexical quality with
preprocessing
 No standalone runner yet (could
extract from Sieve’s Layer 2)

## Page 4

Sieve scan (no
expansion)
Ablation: is SPLADE
helping?
  Needs --no-expand flag or config
2.4 Metrics
Primary (must appear in every report):
Metric Definition Repo status
Fresh Recall@5 (T+0)Recall on fresh files at T+0
 Reported as
fresh_recall_at_5 by
deadline
Fresh Hit@5 (T+0)Whether the fresh file
appears in top-5 at T+0
  Reported
ZeroPrepRetention@5Fresh Recall@5(T+0) / Fresh
Recall@5(T+steady)
 Reported (uses best-deadline
as proxy for steady)
Query latency
p50/p95 (T+0) End-to-end latency at T+0
 Reported
Semantic lift over
ripgrep
Sieve T+0 recall minus ripgrep
T+0 recall
  Reported
Secondary:
Metric Definition Repo status
Time-to-searchableTime until first semantic hit is
possible
  Reported
Scan efficiency ratioRecall@10 / log2(1 +
events_per_query)
 Not yet reported by
harness
Semantic-hard subset
scores
Scores on queries where ripgrep
gets zero hits
  Subset computed
3. Ablations
Each ablation removes one component to test whether it contributes to the core claim. All
ablations run on the same episodes with the same T+0 protocol.

## Page 5

3.1 Required Ablations
IDAblation What changes Repo status
A1No SPLADE
expansion
Query terms only, no learned expansion.
Compile literal query words into Aho-
Corasick.
 Needs
flag/config to
disable
expansion
A2No window
scoring
Count raw hit events per file, rank by total
hits. No proximity, ordered pairs, coverage.
 Needs
simplified scorer
path
A3SPLADE→Tantivy
only
Skip byte scan. Use SPLADE expansion
through Tantivy inverted index only. Not T+0
eligible for fresh.
 Needs
standalone
bench runner
A4Dense kNN only
Skip scan and lexical. Use bge-small
embeddings + flat kNN. Not T+0 eligible for
fresh.
 embed-knn
runner exists
A5No DF filteringRemove post-expansion DF filter. Let all
SPLADE terms into the automaton.
 Needs
config flag to
disable DF filter
A6Random
expansion
Replace SPLADE weights with random
weights on the same vocabulary. Same
number of expansion terms.
 Needs mock
expander
All ablations are 
 small missing work — each is a config flag or a small alternate code
path, not new architecture.
3.2 What Each Ablation Proves
If A1 (no expansion) significantly hurts recall: SPLADE expansion is doing real work, not
just pattern matching.
If A2 (no window scoring) significantly hurts recall: Structured scoring (proximity,
ordered pairs, coverage) matters. Evidence for “late interaction over match events” as a real
primitive.
If A3 (SPLADE→Tantivy) matches scan quality at T+steady but not T+0: The scan
path’s value is freshness, not quality. Still a valid claim but a different one.
If A3 matches scan quality at T+0: The claim is false. SPLADE→Tantivy with fast index

## Page 6

insertion is sufficient. The byte scan is unnecessary.
If A4 (dense only) dominates at T+steady: Dense is better quality but can’t do T+0. Claim
becomes “Sieve trades quality for freshness.” Need to quantify the tradeoff.
If A6 (random expansion) matches SPLADE: The learned model isn’t helping. Serious
problem.
3.3 Kill Criteria
Stop and reassess if ANY of these are true:
A1 (no expansion) Recall@5 is within 5% of full system → SPLADE isn’t helping
A3 (SPLADE→Tantivy) T+0 Recall@5 is within 5% of scan path → indexed path is fast
enough, byte scan is unnecessary
A6 (random expansion) Recall@5 is within 10% of SPLADE → learned weights don’t
matter
These would mean the core novelty claim doesn’t hold. Better to know now.
4. Scope Boundary
In scope (required for the core chart):
Item Status Notes
Score
coherence
fix
 IN
PROGRESS
Separate each retrieval source into its own fusion input; stop
collapsing semantic indexed + fresh before outer RRF
SPLADE
single-
encode
latency fix
 IN
PROGRESSOne ONNX call per query, not N+1
--no-
embed flag
on sieve
index
 SMALL
WORK
Skip embed_pending() so scan path can be isolated. ~10
lines.
--explain
flag on
  SMALL
Promote existing --debug / source_layer plumbing to
first-class per-result provenance output. Partial plumbing

## Page 7

sieve
search
WORK exists via search –debug and per-result source_layer tags.
Ablation
config flags
(A1, A2, A5)
 SMALL
WORK Config flags to disable expansion, window scoring, DF filter
Ablation
bench
runners
(A3, A6)
 SMALL
WORK SPLADE→Tantivy-only runner; random expansion mock
T+steady
benchmark
pass
 SMALL
WORK Add a final deadline after all background jobs complete
Core chart
production
 SMALL
WORK Run all ablations, produce the chart from Section 5
In scope (strengthens the claim, do after core chart):
Item Status Notes
Corpus-level DF
from Tantivy into
scan scorer
 SMALL
WORK
Tantivy exposes doc_freq and total_num_docs. Pass
into scan IDF.
Content-type in
WAL
 SMALL
WORK
Add ContentType enum to WAL metadata. Future spec
section 7c.
Adaptive
windows per
content type
 SMALL
WORK Code: 256B, Prose: 512B. Future spec section 7d.
Anchor-preferred
group centers
 SMALL
WORK
Prefer anchor positions over high-weight expansions.
Future spec section 7e.
Compound-
identifier
realization
 SMALL
WORK
surface.rs already emits title-case variants;
finish/validate code-aware initial-cap forms for
compounds like ErrorHandler. Future spec section 7f.
SPLADE-Code
model swap
 MODERATE
WORK
Replace generic SPLADE with code-specialized model.
Future spec Area 2.

## Page 8

Explicitly out of scope (until core chart exists):
Item Status
ColBERT / late interaction token store
  FUTURE
Unified multi-representation model
  FUTURE
sieve train (any version)
  FUTURE
DF-FLOPS fine-tuning
  FUTURE
Sequential / Memory Caching reranker
  FUTURE
Watch mode polish
  FUTURE
crates.io publish
  FUTURE
IDE integrations
  FUTURE
MCP server
  FUTURE
Business development
  FUTURE
5. The Core Chart
When this document is fulfilled, the deliverable is one chart and one table.
Chart: Fresh Recall@5 vs time-after-write, showing:
Sieve (full system)
Sieve (scan path only — no dense, no indexed SPLADE)
ripgrep (lexical baseline)
Dense kNN (available only after embedding delay)
SPLADE→Tantivy (available only after index insertion)
Each ablation variant (A1-A6)
The x-axis is time after file write (0ms, 100ms, 500ms, 1s, 5s, T+steady). The y-axis is
Recall@5.
The claim is proven if:

## Page 9

Sieve scan path Recall@5 at T+0 is significantly above ripgrep
Sieve scan path Recall@5 at T+0 is not significantly below dense/indexed at T+steady
The scan path advantage at T+0 disappears only after other systems finish
preprocessing
Table: Ablation results showing each component’s contribution.
6. Repo Hygiene Note
 The following repo docs are stale as of this writing and should NOT be treated as ground
truth by Hermes or any agent:
README.md on main still says “Phase 0 scaffold in progress”
Some doc references label Area 4 as “MEMENTO-inspired” while revision notes rename
it toward “Memory Caching”
spec-v0.3 advertises sieve watch but CLI only exposes index, search, status,
export-training, download-model
This research contract supersedes all of those for scope decisions. The future spec (SIEVE-
FUTURE-SPEC.md) remains the long-term R&D plan but is explicitly downstream of this
contract — nothing in the future spec is pursued until the core chart exists.
7. Timeline
WeekDeliverable
1 Score coherence fix + SPLADE latency fix committed
1 --no-embed and --explain flags
2 Ablation flags + runners (A1-A6)
2 Core chart produced on CodeSearchNet
3 Decision point: does the claim hold?
3+ If yes: demo, paper draft, Phase 4.1 quality upgrades from future spec
3+ If no: reassess architecture, decide whether the primitive is worth saving

## Page 10

This document is the contract. No work proceeds that is not in service of Section 4 “in
scope” until the core chart in Section 5 exists.
