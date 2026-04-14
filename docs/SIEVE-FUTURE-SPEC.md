
# SIEVE-FUTURE-SPEC.md

Version: 2026-04-14  
Status: R&D / implementation spec  
Audience: Sieve engineers and research collaborators

## 0. Purpose

This document specifies the next major R&D arc for Sieve after late Phase 4: a query-compiled semantic scan engine that already performs SPLADE-style query expansion, compiles realized terms into an Aho-Corasick byte automaton, scans raw bytes with zero document preprocessing, and scores local windows immediately after ingest.

The goal of the next phase is not to turn Sieve into “yet another sparse index.” The goal is to preserve Sieve’s defining property:

> **Fresh local data becomes semantically searchable immediately, without document-side preprocessing.**

Everything below is judged against that invariant.

---

## 1. Current operating assumptions

The spec assumes the following current baseline in Sieve:

- query-side semantic expansion already exists;
- scan happens over raw bytes via Aho-Corasick;
- ranking is based on windowed match events rather than stored document embeddings;
- fresh content is visible via WAL-backed raw scan immediately;
- background lexical shards exist in Tantivy;
- current code still uses fixed windows (`512/256`), fresh-scan-only DF in `compute_idf`, no WAL `content_type`, no dedicated `CODE_DF_PRIOR`, group-center selection based on highest-weight event, and limited single-term case realization.

These assumptions are consistent with Sieve’s public spec and current code layout, where the architecture is local-first, zero-wait, and uses Aho-Corasick / raw byte scanning with Tantivy as the lexical layer, while current implementation details still show fixed windows and WAL metadata without `content_type`. The public spec also frames Sieve as a zero-wait hybrid retrieval engine with WAL append as the commit point and background lexical segment builds. Relevant external support for these design points comes from the current Sieve spec and code, plus Tantivy’s current ability to expose term DF / corpus statistics via `doc_freq` and total document counts. (See the current Sieve spec/code and current Tantivy statistics APIs.)

### Design constraints

1. **No document-side semantic preprocessing for freshness-critical search.**
2. **CPU-first retrieval path.**
3. **Background indexing remains optional optimization, not a correctness dependency.**
4. **Any learned component must improve scan selectivity, not merely offline benchmark scores.**
5. **Code and prose must become first-class content types.**

### Engineering note on numbers

All compute, latency, and training-time numbers below are **engineering targets or estimates** unless explicitly attributed to a paper.

---

## 2. Roadmap summary

### Release order

| Phase | Scope | Outcome |
|---|---|---|
| 4.1 Stabilization | Technical debt 7c, 7a, 7b, 7d, 7e, 7f + benchmark harness skeleton | Correct DF, content-aware routing, adaptive windows, cleaner scoring |
| 4.2 Code specialization | SPLADE-Code integration, code DF priors, code-aware surface realization | Stronger NL→code retrieval with lower scan waste |
| 5.0 Local training | `sieve train`, corpus adaptation, ONNX export, model registry | User-specific expansion models |
| 5.1 DF-aware sparse learning | DF-FLOPS fine-tuning on mixed corpora | Better expansion patterns, fewer useless scan hits |
| 5.2 Sequential reranking | MEMENTO-inspired residual-state window scorer | Better long-file / long-doc relevance |
| 5.3 Publication track | Freshness benchmark + event late interaction paper | Publishable core primitive |

### Recommended sequence

1. Fix the schema and scoring debt first.
2. Ship the **post-expansion DF filter** before attempting DF-FLOPS fine-tuning.
3. Add `content_type` before WAL format hardens further.
4. Integrate SPLADE-Code on top of content-aware routing.
5. Build `sieve train` only after content-aware model selection and corpus DF plumbing exist.
6. Treat sequential window processing as a final-stage reranker only.

---

## 3. Cross-cutting architecture additions

The following types should be introduced before the area-specific work begins.

```rust
#[derive(Clone, Copy, Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub enum ContentType {
    Code,
    Prose,
    Config,
    Log,
    Mixed,
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExpansionModelKind {
    SpladeDfBase,     // ~109M params, INT8 ONNX target: 110-140 MB
    SpladeDfSmall,    // ~67M params, INT8 ONNX target: 60-90 MB
    SpladeCode06B,    // ~0.6B params, GPU-friendly inference path
    Hybrid,           // prose + code fusion
}

#[derive(Clone, Copy, Debug)]
pub enum TokenClass {
    NaturalWord,
    IdentifierPart,
    TypeName,
    LibraryName,
    Symbol,
    PathFragment,
    Unknown,
}

#[derive(Clone, Debug)]
pub struct ExpansionToken {
    pub term: String,
    pub weight: f32,
    pub group_id: u16,
    pub is_anchor: bool,
    pub token_class: TokenClass,
    pub source_model: ExpansionModelKind,
}

#[derive(Clone, Copy, Debug)]
pub struct DfStats {
    pub indexed_df: u32,
    pub indexed_docs: u32,
    pub wal_df: u32,
    pub wal_docs: u32,
    pub prior_df_frac: f32,
}

pub trait CorpusDfProvider {
    fn stats(&self, term: &str, content_type: ContentType) -> DfStats;
}

#[derive(Clone, Debug)]
pub struct CompiledPattern {
    pub bytes: Vec<u8>,
    pub term_id: u32,
    pub group_id: u16,
    pub weight: f32,
    pub df_frac: f32,
    pub is_anchor: bool,
    pub token_class: TokenClass,
    pub content_mask: u8, // bitmask over ContentType
}

#[derive(Clone, Copy, Debug)]
pub struct WindowProfile {
    pub bytes: u32,
    pub stride: u32,
    pub max_events: u16,
}

#[derive(Clone, Copy, Debug)]
pub struct MatchEvent {
    pub entry_id: u64,
    pub content_type: ContentType,
    pub byte_pos: u32,
    pub window_idx: u32,
    pub term_id: u32,
    pub group_id: u16,
    pub weight: f32,
    pub df_frac: f32,
    pub is_anchor: bool,
    pub pattern_len: u16,
}

#[derive(Clone, Debug)]
pub struct GroupSummary {
    pub group_id: u16,
    pub anchor_center: Option<u32>,
    pub fallback_center: u32,
    pub best_anchor_weight: f32,
    pub best_any_weight: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelManifest {
    pub model_id: String,
    pub base_model: String,
    pub model_kind: ExpansionModelKind,
    pub corpus_fingerprint: String,
    pub tokenizer_hash: String,
    pub content_mix: std::collections::BTreeMap<ContentType, f32>,
    pub onnx_path: String,
    pub quantization: String,
    pub created_at_unix: u64,
    pub metrics: std::collections::BTreeMap<String, f32>,
    pub compatible_sieve_semver: String,
}
```

### Core shared policies

```rust
pub fn default_window_profile(content_type: ContentType) -> WindowProfile {
    match content_type {
        ContentType::Code   => WindowProfile { bytes: 256, stride: 128, max_events: 64 },
        ContentType::Prose  => WindowProfile { bytes: 512, stride: 256, max_events: 64 },
        ContentType::Config => WindowProfile { bytes: 256, stride: 128, max_events: 48 },
        ContentType::Log    => WindowProfile { bytes: 384, stride: 192, max_events: 64 },
        ContentType::Mixed | ContentType::Unknown =>
            WindowProfile { bytes: 384, stride: 192, max_events: 64 },
    }
}
```

---

## Area 1. DF-FLOPS Expansion Model

## 1.1 Problem statement

Sieve’s current SPLADE-style expansion model emits many high-document-frequency terms such as `the`, `is`, `data`, `error`, `result`, or `config`. In an inverted index this is expensive because high-DF postings are long. In Sieve it is worse in a different way: these terms turn into byte patterns that match nearly everywhere, causing:

- explosion in Aho-Corasick event counts;
- more candidate windows per file;
- weaker local discrimination because every window gets “semantic dust”;
- larger CPU cost exactly on the fresh path that should be cheapest.

This is now the central bottleneck in query-compiled semantic scan.

## 1.2 Research basis

SPLADE introduced learned sparse lexical/query expansion using the MLM head and sparsity regularization, and SPLADE v2 improved effectiveness and efficiency over the original model. A unified analysis of learned sparse retrieval later found that query expansion is often not worth its latency unless it improves the actual retrieval primitive. (Formal et al., 2021; Formal et al., 2021b; Nguyen et al., 2023).

DF-FLOPS is directly relevant. The SIGIR 2025 paper shows that standard FLOPS regularization does not sufficiently suppress high-DF terms, and that **DF-aware regularization reduces latency by about 10x** while preserving much of the effectiveness, with much lower top-token DF and better out-of-domain results on most BEIR tasks. The same paper also reports that **static precomputed DF is materially worse than periodically refreshed DF estimates**, which matters for Sieve because corpus composition changes over time. (DF-FLOPS, SIGIR 2025).

## 1.3 Decision

Sieve should do **both**, in this order:

1. **Immediate tactical fix**: post-expansion DF filter at query time.
2. **Strategic model fix**: DF-FLOPS fine-tuning on a mixed code/docs corpus.

The query-time filter is cheap, local, and removes current pain. The fine-tuned model is the long-term answer because it can learn **alternative low-DF expansions** instead of just deleting bad ones.

## 1.4 Approach A — Fine-tune SPLADE with DF-FLOPS regularization

### Recommended model variants

| Variant | Backbone | Params | Intended use |
|---|---|---:|---|
| `sieve-splade-df-small` | SPLADE distil-sized encoder | ~67M | laptops, Colab, local adaptation |
| `sieve-splade-df-base` | coCondenser/BERT-base SPLADE | ~109M | default production-quality query expander |
| `sieve-splade-df-codehybrid` | prose backbone with content-conditioned DF priors | ~109M | bridge before full SPLADE-Code rollout |

The current public SPLADE ecosystem still uses common 67M and 109M parameter classes for practical models such as `splade_v2_distil`, `splade-v3-distilbert`, and `splade-cocondenser-ensembledistil`. (see current practical SPLADE model families).

### Training objective

Use a standard SPLADE ranking loss plus FLOPS and DF-FLOPS:

```python
# train_df_splade.py
import torch
import torch.nn.functional as F

def splade_sparse(logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # logits: [B, T, V]
    act = torch.log1p(torch.relu(logits))
    act = act * attention_mask.unsqueeze(-1)
    # max pooling over sequence -> sparse vocabulary logits
    return act.max(dim=1).values  # [B, V]

def ranking_loss(q_vec, pos_vec, neg_vec, temperature=0.02):
    pos = (q_vec * pos_vec).sum(-1) / temperature
    neg = (q_vec * neg_vec).sum(-1) / temperature
    return F.softplus(neg - pos).mean()

def flops_loss(vec: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(vec).mean(dim=0) ** 2)

def df_flops_loss(q_vec: torch.Tensor, df_frac: torch.Tensor,
                  df_mid: float = 0.05, df_slope: float = 40.0) -> torch.Tensor:
    # df_frac: [V] corpus-level document-frequency fraction in [0, 1]
    penalty = torch.sigmoid((df_frac - df_mid) * df_slope)
    return (q_vec.mean(dim=0) * penalty).pow(2).mean()

def total_loss(q_vec, pos_vec, neg_vec, df_frac):
    return (
        ranking_loss(q_vec, pos_vec, neg_vec)
        + 2e-5 * flops_loss(q_vec)
        + 2e-5 * flops_loss(pos_vec)
        + 5e-5 * df_flops_loss(q_vec, df_frac)
    )
```

### Key adaptation for Sieve

The original DF-FLOPS work was aimed at sparse retrieval over inverted indexes. Sieve’s runtime pain is query-side expansion that becomes byte patterns. Therefore the DF-aware penalty should be applied **more strongly to query activations than document activations**.

Use:

- `λ_q_df = 5e-5`
- `λ_d_df = 1e-5` or `0`
- `df_mid_prose = 0.05`
- `df_mid_code = 0.10`
- `df_slope = 40`

This says: in prose, penalize terms once they appear in more than ~5% of docs; in code, tolerate slightly higher DF because some API vocabulary is shared.

### Dataset requirements

Use a mixed corpus because Sieve is not just a prose retriever.

#### Minimum viable training set

- **Prose**: 500k–1M query-document pairs
  - MS MARCO passage/dev style pairs
  - BEIR subsets for validation
- **Code**: 200k–500k NL→code pairs
  - CodeSearchNet
  - CoIR subsets
- **Held-out DF corpus sample**:
  - 1M documents minimum
  - stratified by `content_type`

#### Good production set

- **Prose**: 2M–5M pairs
- **Code**: 1M–2M pairs
- **DF estimation corpus**: 5M–20M documents or passages
- **Hard negatives**: 7–8 per query
  - BM25/Tantivy negatives
  - current Sieve top false positives
  - one dense baseline negative

The DF-FLOPS paper itself trained on MS MARCO scale (500k queries, 8.8M passages), used in-batch negatives plus 7 hard negatives, and refreshed DF estimates periodically; that is the right order of magnitude for a “serious” run, but Sieve should start smaller. (DF-FLOPS, SIGIR 2025).

### Training pipeline

1. **Build a stratified corpus sample**
   - 50% prose/docs
   - 40% code
   - 10% config/logs if available
2. **Normalize text for DF accounting**
   - lowercase prose words
   - preserve code identifier casing separately
3. **Precompute rolling DF tables**
   - `df_prose[token]`
   - `df_code[token]`
4. **Train with periodic DF refresh**
   - refresh every 500 steps on local runs
   - every 100 steps on bigger GPU runs
5. **Validate on both retrieval and scan metrics**
6. **Export ONNX INT8 for query inference**
   - small model target: 60–90 MB
   - base model target: 110–140 MB

### Hardware targets

| Run | Hardware | Effective batch | Time target |
|---|---|---:|---:|
| local dev small | 1× RTX 4090 / L4 24GB | 64–128 | 8–16 h |
| local base | 1× 24GB GPU + grad accumulation | 64 | 12–24 h |
| heavier production | 2× A100 40GB | 128–256 | 6–12 h |
| large paper-quality | 4× H100 | 128+ | 3–6 h |

These are engineering targets; the paper’s own reported setup used 4×H100 and 50k steps. (DF-FLOPS, SIGIR 2025).

### How to tell whether the fine-tuned model is actually better for Sieve

Do **not** evaluate only `MRR@10`. Evaluate “scan pattern quality.”

#### Retrieval metrics

- Recall@5
- Recall@10
- nDCG@10
- MRR@10

#### Scan-shape metrics

- mean expansions kept/query
- mean retained term DF
- top-1 retained term DF
- total match events/query
- candidate windows/query
- p50 / p95 raw scan latency
- p50 / p95 total semantic query latency
- fraction of events falling into top-5 final windows
- **scan efficiency ratio** = `Recall@10 / log2(1 + events_per_query)`

#### Acceptance target

A fine-tuned model should count as successful only if it achieves **one** of:

- same Recall@10 ±1 point with **≥50% fewer events/query**, or
- better Recall@10 by ≥2 points with **no increase** in p95 latency, or
- same retrieval quality with **≥30% lower p95 latency**.

### Rust-side DF-aware query compilation

```rust
pub struct DfFilterConfig {
    pub df_threshold_prose: f32, // 0.05
    pub df_threshold_code: f32,  // 0.10
    pub max_terms: usize,        // 24
    pub min_weight: f32,         // 0.05
    pub soft_penalty_lambda: f32,
}

pub fn blended_df_frac(stats: DfStats) -> f32 {
    let total_docs = (stats.indexed_docs + stats.wal_docs).max(1) as f32;
    let observed = (stats.indexed_df + stats.wal_df) as f32 / total_docs;
    // prior is only a backoff, not a replacement
    let prior_weight = 16.0 / (16.0 + total_docs);
    observed * (1.0 - prior_weight) + stats.prior_df_frac * prior_weight
}

pub fn filter_and_reweight(
    toks: &[ExpansionToken],
    df: &dyn CorpusDfProvider,
    ct: ContentType,
    cfg: &DfFilterConfig,
) -> Vec<ExpansionToken> {
    let hard = match ct {
        ContentType::Code => cfg.df_threshold_code,
        _ => cfg.df_threshold_prose,
    };

    let mut out = Vec::new();
    for tok in toks {
        let stats = df.stats(&tok.term, ct);
        let frac = blended_df_frac(stats);

        if tok.is_anchor {
            out.push(tok.clone());
            continue;
        }
        if tok.weight < cfg.min_weight {
            continue;
        }
        if frac > hard {
            continue;
        }

        let mut kept = tok.clone();
        let soft_penalty = 1.0 + cfg.soft_penalty_lambda * frac;
        kept.weight /= soft_penalty;
        out.push(kept);
    }

    out.sort_by(|a, b| b.weight.total_cmp(&a.weight));
    out.truncate(cfg.max_terms);
    out
}
```

## 1.5 Approach B — Post-expansion DF filter

### Why ship this first

This is the fastest path to stabilizing the current scan engine.

It requires:

- corpus-level DF statistics from Tantivy;
- WAL DF counts for unindexed data;
- priors only as backoff.

It does **not** require any retraining.

### Concrete behavior

1. Generate base SPLADE expansions.
2. Preserve all anchors.
3. For non-anchor expansions:
   - drop if `df_frac > τ(content_type)`
   - drop if `weight < min_weight`
   - reweight remaining terms by inverse DF penalty
4. Keep top `K` terms after filtering.
5. Compile only those into Aho-Corasick.

### Recommended starting config

| Content type | Hard DF threshold | Max terms | Comment |
|---|---:|---:|---|
| Prose | 0.05 | 24 | suppress generic language quickly |
| Code | 0.10 | 24 | tolerate some shared API vocabulary |
| Config | 0.08 | 20 | keys repeat often |
| Log | 0.12 | 20 | recurring log words are common |

### Expected outcome

- 3x–10x fewer events/query on bad queries
- much smaller candidate-window sets
- lower p95 latency
- small recall loss if thresholds too aggressive

### Risk

The filter can only remove terms. It cannot invent better low-DF expansions. Therefore it is a **governor**, not a model improvement.

## 1.6 Comparison

| Criterion | DF-FLOPS fine-tune | Post-expansion DF filter |
|---|---|---|
| Time to ship | slow | fast |
| New data needed | yes | no |
| Learns better alternatives | yes | no |
| Immediate latency relief | medium | high |
| Long-term scan quality ceiling | high | medium |
| Failure mode | undertrained model | over-aggressive pruning |
| Recommendation | strategic track | immediate default |

## 1.7 Success metrics

### Ship criteria for the filter

- ≥60% reduction in mean events/query on code benchmark
- ≥40% reduction in p95 semantic latency
- Recall@10 drop ≤1.5 points versus current baseline

### Ship criteria for DF-FLOPS fine-tuned model

- same or better Recall@10 than base SPLADE
- ≥50% reduction in total events/query
- top-1 retained term DF reduced by ≥5×
- p95 latency reduced by ≥30%

---

## Area 2. SPLADE-Code for code search

## 2.1 Problem statement

General SPLADE models are good at prose semantics but weak at NL→code retrieval because code retrieval has three distinct problems:

1. natural language and code use different vocabularies;
2. identifiers are fragmented by subword tokenization;
3. file/function structure creates longer local dependencies than ordinary prose snippets.

Sieve already compensates partially with surface realization for `camelCase`, `snake_case`, `PascalCase`, and related variants. That helps lexical reach, but it is still not the same as a model that actually understands code-centric expansion terms.

## 2.2 Research basis

SPLADE-Code is the first large-scale learned sparse retrieval family specialized for code retrieval. The March 2026 paper introduces models from **0.6B to 8B**, reports **75.4 on MTEB Code under 1B parameters**, **79.0 with 8B**, and shows that learned expansion tokens are critical for bridging the NL→code gap while still enabling **sub-millisecond retrieval on a 1M-passage collection** with modest effectiveness loss. The paper also emphasizes code-specific challenges: subword fragmentation, semantic mismatch, long code contexts, and multilingual / multi-language diversity. (Lupart et al., 2026; SPLADE-Code model card).

## 2.3 Decision

Integrate SPLADE-Code as a **parallel expansion model family** rather than replacing the prose model.

### Recommended policy

- prose-heavy corpus/query → use prose DF-aware SPLADE
- code-heavy corpus/query → use SPLADE-Code
- mixed corpus or ambiguous query → run both and fuse

## 2.4 Detecting code vs prose

This must happen at **two levels**:

1. **document level** on ingest
2. **query level** at search time

### Document-level detector

Use a two-stage classifier:

#### Stage A: cheap heuristics

- file extension (`.rs`, `.py`, `.ts`, `.go`, `.java`, `.cpp`, etc.)
- shebangs
- path fragments (`src/`, `lib/`, `tests/`, `docs/`, `config/`)
- punctuation ratios
- keyword counts (`fn`, `class`, `async`, `impl`, `def`, `return`, `{`, `};`)
- timestamp / log-pattern regexes
- YAML/JSON/TOML markers for config

#### Stage B: tiny content classifier

Use a char-ngram linear classifier or fastText-style model:

- classes: `code`, `prose`, `config`, `log`, `mixed`
- model size target: `<2 MB`
- CPU inference target: `<0.2 ms` per file header/sample

```rust
pub struct ContentTypeDetector {
    pub ext_rules: std::collections::HashMap<String, ContentType>,
    pub linear_model_bytes: Vec<u8>,
}

impl ContentTypeDetector {
    pub fn classify(&self, path: &str, sample: &[u8]) -> ContentType {
        // 1. extension/path fast path
        // 2. fallback to linear classifier over char n-grams + feature counts
        ContentType::Unknown
    }
}
```

### Query-level detector

Use the same idea but over the query text:

Signals:

- identifier-like tokens (`ErrorHandler`, `snake_case`, `::`, `/api`, `.from`, `TypeError`)
- programming language names
- stack traces
- file paths
- API-like method names

Return `P(code_query)`.

## 2.5 Model selection policy

```rust
pub struct ExpansionPlan {
    pub primary: ExpansionModelKind,
    pub secondary: Option<ExpansionModelKind>,
    pub fuse_weights: (f32, f32),
}

pub fn choose_expansion_plan(
    query_code_prob: f32,
    corpus_mix_code: f32,
) -> ExpansionPlan {
    if query_code_prob > 0.7 || corpus_mix_code > 0.8 {
        ExpansionPlan {
            primary: ExpansionModelKind::SpladeCode06B,
            secondary: None,
            fuse_weights: (1.0, 0.0),
        }
    } else if query_code_prob < 0.3 && corpus_mix_code < 0.3 {
        ExpansionPlan {
            primary: ExpansionModelKind::SpladeDfBase,
            secondary: None,
            fuse_weights: (1.0, 0.0),
        }
    } else {
        ExpansionPlan {
            primary: ExpansionModelKind::SpladeCode06B,
            secondary: Some(ExpansionModelKind::SpladeDfBase),
            fuse_weights: (0.65, 0.35),
        }
    }
}
```

### Deployment recommendation

- Default laptop mode: single-model routing only
- Workstation / server mode: hybrid fusion for ambiguous queries

## 2.6 Whether to run both models and fuse

### Option 1 — one model only

Pros:
- cheapest latency
- simpler automaton
- fewer events

Cons:
- ambiguous queries lose recall

### Option 2 — dual expansion + term fusion (recommended default for mixed corpora)

Expand with both models, normalize weights, merge by `max` or weighted sum, then compile **one** automaton.

Pros:
- one scan
- best fit for Sieve’s engine
- preserves zero-preprocessing path

Cons:
- more compiled terms if not pruned

### Option 3 — dual scan + result fusion

Run separate automata over content-specific subsets and fuse results with weighted RRF.

Pros:
- clean isolation
- easier debugging

Cons:
- double scan cost

### Recommendation

Use **Option 2** by default, but cap total fused terms to 32 after DF filtering.

## 2.7 Interaction with Sieve’s surface realization layer

SPLADE-Code already learns code-specific expansion tokens, including identifier fragments and type-like vocabulary. Sieve should therefore stop treating all expansion outputs as ordinary words.

### New rule

Every expansion token gets a `TokenClass`.

- `NaturalWord` → normal prose realization
- `IdentifierPart` → code realization
- `TypeName` → preserve case, add initial-cap / Pascal variants
- `Symbol` → preserve exact surface
- `PathFragment` → allow slash/dot variants

### Code-specific realization policy

```rust
pub fn realize_code_token(term: &str, class: TokenClass) -> Vec<Vec<u8>> {
    let mut out = vec![term.as_bytes().to_vec()];

    // initial-cap form for inside compound identifiers
    if let Some(first) = term.chars().next() {
        if first.is_ascii_lowercase() {
            let mut chars = term.chars();
            let head = chars.next().unwrap().to_ascii_uppercase();
            let tail: String = chars.collect();
            out.push(format!("{}{}", head, tail).into_bytes()); // error -> Error
        }
    }

    // existing surface rules still apply selectively
    // snake_case, camelCase, PascalCase, screaming snake, etc.
    // but only when class warrants it
    out
}
```

### Important constraint

Do **not** let SPLADE-Code and the surface layer create a combinatorial explosion. Use this rule:

- if the model emits an exact multi-token identifier or type-like token, preserve it;
- if it emits fragments, realize only:
  - exact lowercase
  - initial-cap
  - one delimiter-aware form
  - one camel/pascal form if supported by the query context

Maximum realized surfaces per code token: **4**

## 2.8 Training / inference pipeline for code path

### Inference

- preferred code model: `naver/splade-code-06B`
- quantized GPU path for power users
- fallback if no GPU: keep prose model + code priors + code surface realization

The 0.6B SPLADE-Code model is already available as a released model and is the natural first integration target. (see the released `splade-code-06B` model card).

### Fine-tuning

If Sieve later fine-tunes the code model:

- use LoRA rank 64
- max length 512
- 7–8 hard negatives/query
- mined from dense retriever + BM25 + current Sieve
- learning rate target: `1e-4`
- QLoRA if running on 24GB cards

This follows the broad training pattern described in the SPLADE-Code work. (following the SPLADE-Code training recipe).

## 2.9 Success metrics

- +5 to +10 Recall@10 on code benchmark vs prose SPLADE
- ≥25% reduction in wasted code match events/query
- ≥15% increase in “no lexical overlap” code query success
- no more than 20% added median query latency in hybrid mode

---

## Area 3. Corpus-specific vocabulary fine-tuning

## 3.1 Problem statement

Generic vocabularies underserve local corpora. Repositories, internal docs, configs, stack traces, and product names all produce domain-specific lexical spaces. A general SPLADE model often expands to the wrong nearby concepts because it has never seen the user’s corpus-specific terminology enough times.

Sieve’s retrieval primitive is query-side and lexicalized. That means vocabulary fit matters even more than it does in a standard embed-then-search stack.

## 3.2 Research basis

The ECIR 2024 paper **Improved Learned Sparse Retrieval with Corpus-Specific Vocabularies** shows that pretraining the underlying BERT on the target corpus and adapting vocabulary can improve retrieval quality by **up to 12%** and reduce latency by **up to 50%**, with benefits transferring across SPLADE and related sparse methods. (Yu et al., ECIR 2024).

## 3.3 Product decision

Add a first-class command:

```bash
sieve train \
  --corpus /path/to/repo-or-docs \
  --base splade-df-base \
  --content auto \
  --output ~/.local/share/sieve/models/acme-2026-04 \
  --onnx-int8
```

### What it should do

1. collect corpus data
2. classify content types
3. optionally adapt vocabulary / tokenizer
4. run domain-adaptive pretraining
5. fine-tune sparse expansion model
6. validate on synthetic or logged queries
7. export ONNX + metadata + DF tables

## 3.4 Modes

### Mode A — calibrate only

For tiny corpora, do not train.

Use when:

- `<20k documents` or
- `<100 MB` of normalized text

Action:

- build corpus fingerprint
- build local DF tables
- build local lexicon for surface realization
- reuse global model

### Mode B — small adaptation

Use when:

- `20k–100k documents`
- `100 MB–1 GB`

Action:

- sample up to 50M tokens
- run 1 epoch MLM / DAPT
- fine-tune for 5k–10k IR steps
- export INT8 ONNX

### Mode C — full adaptation

Use when:

- `>100k documents`
- `>1 GB`
- repeated use on same corpus is expected

Action:

- 100M–300M pretraining tokens
- 10k–50k IR steps
- optional DF-FLOPS
- optional code-specialized path

## 3.5 Compute requirements

### Consumer feasibility

Yes, for BERT-sized models.

| Scenario | Hardware | Feasible? | Notes |
|---|---|---|---|
| `splade-df-small` adaptation | 12–16GB GPU | yes | local laptop / desktop |
| `splade-df-base` adaptation | 24GB GPU or Colab L4 | yes | recommended |
| SPLADE-Code 0.6B LoRA | 24GB GPU / Colab Pro L4/A100 | yes | slower, use QLoRA |
| SPLADE-Code 8B | consumer GPU | no | server/research only |

### Time estimates

| Task | Hardware | Time target |
|---|---|---:|
| collect + classify + fingerprint | CPU | 10–60 min |
| 50M-token MLM | RTX 4090 / L4 | 2–6 h |
| 100M-token MLM | RTX 4090 / L4 | 4–10 h |
| 10k-step sparse fine-tune | 24GB GPU | 4–8 h |
| 0.6B code LoRA fine-tune | 24GB GPU | 8–20 h |
| ONNX export + INT8 quantization | CPU/GPU | 10–30 min |

### Colab guidance

- Colab T4: only `small` mode
- Colab L4: good for BERT base and light 0.6B LoRA
- Colab A100: good for `full adaptation`

## 3.6 Data pipeline

### Corpus collection

```python
# prepare_corpus.py
from pathlib import Path
import hashlib, json

def fingerprint_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()

def collect_corpus(root: Path):
    rows = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            raw = path.read_bytes()
        except Exception:
            continue
        rows.append({
            "path": str(path),
            "sha256": fingerprint_file(path),
            "bytes": len(raw),
        })
    return rows

if __name__ == "__main__":
    rows = collect_corpus(Path("/corpus"))
    Path("manifest.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows)
    )
```

### Query supervision sources

Use them in this order:

1. real query logs if available
2. README / heading / section titles → file targets
3. docstrings/comments → function/file targets
4. symbol names → file/function targets
5. issue titles / commit messages → touched files
6. LLM paraphrases of 2–5 if the user enables synthetic augmentation

### Hard negatives

Mine from:

- Tantivy/BM25 top-k false positives
- current Sieve false positives
- dense baseline false positives

This matters because Sieve’s failure modes are not the same as standard sparse retrieval failure modes.

## 3.7 Vocabulary adaptation strategy

Do not always replace the tokenizer. Tokenizer replacement is powerful but operationally expensive.

### Recommended two-tier approach

#### Tier 1 — lexicon adaptation (default)

- keep base tokenizer
- add local lexicon file:
  - project names
  - symbols
  - APIs
  - type names
  - recurring config keys
- feed lexicon into surface realization and DF priors

Use this for most users.

#### Tier 2 — tokenizer/vocabulary adaptation (advanced)

- train SentencePiece / WordPiece on corpus
- target vocab sizes:
  - 32k base-compatible
  - 40k / 48k extended for larger corpora
- initialize new embeddings from old where possible
- DAPT on new vocabulary
- fine-tune sparse retriever

Use this only when the corpus is large and stable.

## 3.8 Sparse fine-tuning script skeleton

```python
# train_sieve_local.py
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, Trainer
import torch

BASE_MODEL = "naver/splade-cocondenser-ensembledistil"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForMaskedLM.from_pretrained(BASE_MODEL)

def make_pair(example):
    return {
        "query": example["query"],
        "positive": example["positive"],
        "negative": example["negative"],
    }

# Replace with real dataset builder:
train_ds = load_dataset("json", data_files="train_pairs.jsonl")["train"].map(make_pair)

class SparseTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # tokenize query / pos / neg
        # build SPLADE sparse vectors
        # add optional DF-FLOPS from local corpus stats
        loss = torch.tensor(0.0, device=model.device)
        return (loss, None) if return_outputs else loss

args = TrainingArguments(
    output_dir="out",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    save_steps=1000,
    logging_steps=50,
    bf16=True,
)

trainer = SparseTrainer(model=model, args=args, train_dataset=train_ds)
trainer.train()
```

## 3.9 ONNX export

### Requirements

- export query encoder only
- include tokenizer assets
- include model manifest
- optionally quantize to INT8

### Manifest schema

```rust
pub struct TrainArtifacts {
    pub manifest: ModelManifest,
    pub df_tables_path: String,
    pub surface_lexicon_path: String,
    pub content_detector_path: String,
}
```

### Versioning policy

Model IDs should encode:

- base model
- content regime
- corpus fingerprint
- training config hash
- Sieve-compatible version

Example:

```text
sieve-splade-df-base@1.1.0+corp.7f3a1c2e.code0.62.prose0.31
```

### Storage layout

```text
~/.local/share/sieve/models/
  sieve-splade-df-base@1.1.0+corp.7f3a1c2e/
    manifest.json
    model.onnx
    tokenizer.json
    tokenizer_config.json
    special_tokens_map.json
    df_stats.zst
    surface_lexicon.txt
    eval.json
```

### Model selection at runtime

Prefer in order:

1. exact corpus fingerprint match
2. same content mix + nearest vocabulary hash
3. global code model / global prose model
4. global base fallback

## 3.10 Success metrics

- local held-out Recall@10 improves by 5–12%
- p95 query latency improves by 15–50%
- number of retained expansions/query decreases by 20–40%
- model export remains under:
  - 90 MB INT8 for small
  - 140 MB INT8 for base

---

## Area 4. Sequential window processing with residual state (MEMENTO-inspired)

## 4.1 Problem statement

Sieve’s current event reranker scores each window independently. That misses cross-window structure such as:

- a function signature in one window and its body two windows later;
- an error declaration early in a file and its retry logic later;
- a README section that defines terminology before the detailed subsection;
- log context preceding the actual fault.

Independent scoring is efficient, but it assumes local relevance is self-contained.

## 4.2 Research basis

The MEMENTO work shows that transformer computation can preserve useful information from previous blocks through compressed state and residual pathways even when earlier context is no longer directly present. It reports major memory savings and throughput improvements, but more importantly for Sieve it shows that **compressed carry-forward state can retain more signal than expected**. Removing the implicit carry-forward path substantially reduced reasoning performance. (MEMENTO, 2026).

For retrieval, the relevant analogy is not “use hidden leakage as-is.” The relevant lesson is: **local block scoring can be improved by forwarding compact state from earlier blocks instead of re-encoding the entire document each time**.

Long-document sparse retrieval work also supports the importance of proximity and segment interactions. ExactSDM outperformed simpler aggregation and even SoftSDM did not beat ExactSDM, suggesting that positional dependence matters, but complex soft expansion dependence may not be necessary. (Nguyen et al., SIGIR 2023).

## 4.3 Decision

Add a **final-stage sequential reranker** that is used only on a small candidate set.

Do **not** replace the current independent window scorer globally.

## 4.4 Architecture

### Input

For each candidate document, take the top `W` windows from the current independent scorer, ordered by document position, not by score.

Recommended default:
- `W = 8` for code
- `W = 6` for prose

### Event tokenization

Each window becomes up to 64 event tokens:

```rust
#[derive(Clone, Copy, Debug)]
pub struct EventToken {
    pub term_id: u32,
    pub group_id: u16,
    pub rel_pos_bucket: u8, // 0..31 within window
    pub weight_q8: u8,
    pub df_bucket: u8,
    pub is_anchor: bool,
}
```

### Memory state

```rust
#[derive(Clone, Debug)]
pub struct MemoryState {
    pub slots: [[f32; 128]; 4],  // 4 memory tokens, 128 dims each
    pub residual: [f32; 128],
    pub prev_window_score: f32,
}
```

### Model

Recommended first implementation:

- 4-layer transformer
- hidden size 256
- 4 heads
- 4 learned memory slots
- 1 scalar score head
- 1 memory compression head

At step `t`:

1. encode current window event tokens
2. prepend memory slots from step `t-1`
3. self-attend within `[memory | current window]`
4. output:
   - `score_t`
   - `memory_t = compress(hidden_t)`

### Compression module

```python
class MemoryCompressor(torch.nn.Module):
    def __init__(self, d_model=256, n_slots=4):
        super().__init__()
        self.slot_queries = torch.nn.Parameter(torch.randn(n_slots, d_model))
        self.proj = torch.nn.Linear(d_model, 128)

    def forward(self, hidden):  # [B, T, d_model]
        # learned slot queries attend over current hidden states
        attn = torch.softmax(self.slot_queries @ hidden.transpose(-1, -2), dim=-1)
        slots = attn @ hidden
        return self.proj(slots)  # [B, n_slots, 128]
```

### Explicit rule

Make the carry-forward state **explicit and trainable**. Do not rely on incidental KV leakage.

## 4.5 Training data requirements

This model needs ordered multi-window supervision.

### Minimum viable data

- 100k query-document-span triples
- documents must be full files / full docs, not pre-chunked passages
- positive span offsets or positive windows

### How to get it

1. **Code corpora**
   - map docstrings/comments/issues to full source files
   - positive span = function/class region
2. **Docs/prose**
   - heading/title query to section span
   - QA datasets with answer spans if available
3. **Pseudo-labeling**
   - use current Sieve top window in relevant docs as seed positive
   - include adjacent windows with weaker labels
4. **User feedback**
   - file opened / snippet copied / accepted answer as weak signal

## 4.6 Training objective

Use a multi-task loss:

- window-level relevance BCE
- pairwise ranking between positive and hard-negative windows
- document-level max/softmax aggregated score
- auxiliary future-prediction loss:
  - from `memory_t`, predict whether a high-anchor window appears within next 2 windows

That last auxiliary task encourages memory to preserve useful deferred context.

## 4.7 Runtime policy

Sequential reranking is worth the cost only when the document exhibits distributed evidence.

### Trigger conditions

Run it only when **all** are true:

- document has at least 4 candidate windows
- top 2 windows are separated by at least 2 strides
- query has at least 2 semantic groups or anchor+expansion evidence
- content type is `Code`, `Prose`, or `Mixed` (not logs unless explicitly enabled)

### Cost control

- max documents reranked sequentially: 20
- max windows/doc: 8
- max events/window: 64

### Parallelism tradeoff

Independent scoring is embarrassingly parallel. Sequential scoring is not.

Mitigations:

- bucket docs by `W`
- batch across documents for each timestep
- run on GPU if available, CPU fallback otherwise

## 4.8 When this is worth it

Use it for:

- large source files
- long markdown docs
- error/debug flows
- cross-window semantic dependencies

Do **not** use it for:

- tiny functions
- short prose snippets
- pure lexical anchor hits with one dominant window

## 4.9 Success metrics

- +2 to +5 nDCG@10 on long-file subset
- +3 to +8 Recall@5 on distributed-evidence subset
- ≤25% added rerank-stage latency
- no change to first-stage freshness path

---

## Area 5. Late interaction over match events (publishable research)

## 5.1 Formal problem statement

Let a query `q` be transformed by a learned expansion model into a sparse weighted set of lexicalized terms:

`E(q) = {(t_i, w_i, g_i, a_i)}`

where:

- `t_i` is a lexical term or surface form seed
- `w_i` is a learned weight
- `g_i` is a semantic group id
- `a_i` indicates anchor/seed status

Let a surface realization function map those terms to byte patterns:

`R(E(q)) = {p_j}`

Given a document `d` represented only as a byte string, let a byte-scan operator produce transient match events:

`M(q, d) = {(p_j, b_k)}`

where `b_k` is a byte position of a realized pattern.

Sieve’s retrieval problem is:

> **Retrieve top-k documents or windows using only raw bytes, lightweight metadata, and query-side learned compilation, without stored document-side embeddings or sparse token vectors.**

The ranking function is a late interaction over transient events, not over stored document representations:

`score(q, d) = Agg_{windows ω in d} F(q, M(q, d, ω))`

This is the core primitive.

## 5.2 Positioning in the IR literature

### Where Sieve fits

- **Lexical retrieval**: BM25, grep, inverted indexes
- **Dense retrieval**: DPR / embedding + ANN
- **Learned sparse retrieval**: SPLADE and successors
- **Late interaction**: ColBERT
- **Routed token interaction**: COIL, CITADEL
- **Sparse late interaction**: SLIM
- **Long-doc sparse proximity**: ExactSDM / SoftSDM

### What is novel

Sieve combines two ideas that are individually known but, as far as the current literature shows, not combined in this specific form:

1. **routing / lexicalized matching** from learned sparse retrieval and token routing;
2. **late interaction-like scoring** over local positions;
3. **without stored document-side embeddings or sparse vectors**;
4. **and with zero document preprocessing on the freshness path**.

ColBERT performs late interaction over stored contextualized token embeddings. COIL stores contextualized token representations in inverted lists. CITADEL routes token vectors to lexical keys. SLIM turns token vectors into sparse token-space representations and still stores those representations for inverted-index retrieval. Sieve instead performs interaction over **transient byte-level match events produced at query time**. (Khattab and Zaharia, 2020; Gao et al., 2021; Li et al., 2022; Li et al., 2023).

The closest conceptual support for Sieve’s proximity scoring comes from long-document learned sparse retrieval, where proximal term dependence is crucial and ExactSDM outperforms simpler aggregation. (Nguyen et al., SIGIR 2023).

## 5.3 Suggested paper title

**Query-Compiled Semantic Scan: Late Interaction over Match Events without Document-Side Semantic Indexing**

Alternative:
**Event Late Interaction for Zero-Preprocessing Retrieval**

## 5.4 Core contribution claim

A careful, defensible contribution statement:

> To our knowledge, this is the first retrieval architecture that performs learned semantic retrieval by compiling query expansions into a byte-level scan and ranking documents via late interaction over transient match events, without storing document-side embeddings or sparse token vectors.

Do not overclaim “first semantic grep ever.” Claim the exact architectural combination.

## 5.5 Paper structure

### 1. Introduction
- freshness problem in local search
- dense embeddings require ingest-time preprocessing
- sparse retrieval usually still assumes document-side sparse indexing
- Sieve’s query-compiled scan primitive

### 2. Related work
- SPLADE / learned sparse retrieval
- ColBERT / late interaction
- COIL / CITADEL / SLIM
- SDM for long sparse retrieval
- streaming search systems

### 3. Method
- query expansion
- DF-aware pruning
- surface realization
- Aho-Corasick scan over raw bytes
- event-based window ranking
- optional sequential reranker

### 4. Benchmark
- fresh append-only evaluation
- code and prose subsets

### 5. Results
- retrieval quality
- zero-preprocessing freshness
- latency/storage tradeoffs
- ablations

### 6. Discussion
- when Sieve wins
- when embed+ANN still wins
- limitations

## 5.6 Experiments to run

### Primary experiments

1. **Zero-preprocessing freshness benchmark**
2. **Code retrieval benchmark**
3. **Long-document benchmark**
4. **Storage cost benchmark**
5. **Latency-vs-quality benchmark**

### Benchmarks

Minimum set:

- CodeSearchNet Challenge / Corpus
- BEIR subset for prose
- Robust04 or MS MARCO doc-style long contexts
- custom append-only freshness protocol

### Baselines

#### Lexical baselines
- ripgrep
- Tantivy/BM25

#### Dense baselines
- `bge-code-v1 + FAISS HNSW` for code
- `Qwen3-Embedding-0.6B + FAISS` as lighter semantic baseline

The BGE code model and Qwen3 embedding series are current code/text embedding baselines available as public models. (current public model cards for BGE-Code-v1 and Qwen3-Embedding-0.6B).

#### Sparse baselines
- SPLADE v2 / v3 over inverted index
- SPLADE-Code over inverted index

#### Late interaction baselines
- ColBERTv2 if feasible
- SLIM if CPU-oriented comparison is desired

#### Production-like competitor
- Vespa streaming search
- Vespa vector search with embedding field

Vespa’s streaming search is the closest production analogue on the “search raw/stored fields without a conventional index” axis, but semantic vector search in Vespa still depends on embedding fields and schema/index setup. (see current Vespa streaming and vector-search documentation).

### Ablations

1. no DF filter
2. DF filter only
3. DF-FLOPS model
4. fixed vs adaptive windows
5. anchor-preferred group centers vs highest-weight centers
6. prose model vs SPLADE-Code
7. surface realization on/off / code-only variants
8. sequential reranker on/off

## 5.7 Main plots and tables

### Must-have figures

- Recall@5 vs time-since-ingest
- p95 latency vs query difficulty
- events/query vs Recall@10
- storage footprint per million docs
- code query success on no-overlap subset

### Must-have tables

- overall benchmark table
- freshness-at-T+0 table
- ablation table
- failure-analysis table

## 5.8 Venue recommendation

### Best target: SIGIR full paper
Choose this if:
- benchmark is strong
- experiments cover both freshness and standard retrieval
- ablations are clean
- code/artifact is reproducible

### Good fallback: CIKM
Choose this if:
- system contribution is stronger than theory
- evaluation is broad but not fully polished for SIGIR

### Good first publication: ECIR
Choose this if:
- novelty is real but experiments are still narrower
- paper is a strong “systems + method” package

### Workshop fallback
Only if the benchmark is not ready yet.

### Recommendation

Aim for **SIGIR 2027 full paper**, with **CIKM 2026/2027** as fallback depending on readiness.

## 5.9 Success metrics

A publishable Area 5 outcome should satisfy all of the following:

- a clear formalization of query-compiled semantic scan and event late interaction;
- at least one benchmark showing a statistically meaningful gain over ripgrep/BM25 on semantic recall at `T+0`;
- a steady-state result showing Sieve reaches at least **85–95%** of the dense baseline’s Recall@5 on the primary code track;
- a storage comparison showing Sieve avoids document-side embedding or sparse-vector state on the freshness path;
- ablations showing that event interaction, adaptive windows, and anchor-preferred ordering each matter;
- a fully reproducible artifact with scripts, configs, and evaluation harness.

---

## Area 6. Benchmark and evaluation framework

## 6.1 Goal

Design a benchmark that proves Sieve’s central claim:

> **Fresh local data becomes semantically searchable immediately with no document preprocessing.**

The benchmark must reward systems that can retrieve relevant results at `T+0` after ingest, not merely after an offline embedding/indexing job finishes.

## 6.2 Public corpus to use

### Primary corpus: CodeSearchNet Challenge + Corpus

Why this corpus:

- public and well known
- large enough to matter
- natural language → code retrieval is semantically hard
- includes **99 expert queries with ~4k relevance annotations**
- corpus contains about **6 million functions across six languages**

That makes it a strong primary corpus for Sieve because it naturally stresses semantic search over raw local code. (Husain et al., 2019).

### Optional extension track

Add a prose track later with BEIR, but do not block the first benchmark on that.

## 6.3 Query set with known relevant results

Use two sets:

### Set A — expert-labeled evaluation
- CodeSearchNet Challenge 99 queries
- ~4k expert relevance annotations
- primary reported quality numbers

### Set B — large-scale streaming evaluation
- auto-generated docstring/comment queries from CodeSearchNet pairs
- use for latency/freshness stress
- sample 10k–100k queries

## 6.4 Benchmark protocol

### Freshness protocol

1. Start with empty corpus.
2. Ingest documents in append-only batches.
3. After each batch fsync completes:
   - issue queries whose relevant docs first appear in that batch
   - require systems to answer **immediately**
4. Record:
   - whether the system can answer at all
   - time to searchability
   - Recall@5 / Recall@10 / nDCG@10
   - end-to-end latency

### Fairness rule

A system is considered “searchable” for a newly ingested document only when all artifacts required for its semantic retrieval are available.

That means:

- Sieve: searchable at WAL commit
- ripgrep: searchable at file write
- dense ANN: searchable only after chunking, embedding, and ANN insertion
- Vespa semantic vector path: searchable only after embedding field is populated
- Vespa streaming lexical: searchable immediately, but not semantically

This is the key test. It makes the claim falsifiable.

## 6.5 Metrics

### Quality

- Recall@5
- Recall@10
- nDCG@10
- MRR@10

### Latency

- p50 / p95 / p99 query latency
- p50 / p95 scan-only latency
- candidate windows/query
- events/query

### Freshness

- **time_to_searchable_ms**
- **fresh_recall@5(T+0)**
- **fresh_recall@5(T+1s)**
- **fresh_recall@5(T+10s)**

### Efficiency

- storage bytes / million docs
- CPU time/query
- GPU requirement at query time
- ingest work/doc

### Core derived metric

**Zero-Preprocessing Semantic Quality (ZPSQ)**

`ZPSQ = Recall@5 at T+0 immediately after ingest`

This is the centerpiece metric.

## 6.6 Comparisons

### System 1 — ripgrep
- lexical baseline
- immediate
- zero preprocessing
- expected low semantic recall

### System 2 — embed + kNN
Recommended code baseline:
- `bge-code-v1`
- FAISS HNSW or IVF-HNSW
- chunk/file embedding pipeline

### System 3 — Vespa streaming
Two modes:
- lexical streaming search
- vector semantic search with embedding field

### System 4 — Sieve
- immediate raw-byte semantic scan
- no document preprocessing
- DF filter on
- adaptive windows on
- code/prose routing on

## 6.7 Specific zero-preprocessing semantic test

### Test name

`fresh_code_t0_semantic`

### Procedure

For each batch:

1. append 1000 new functions/files
2. commit to WAL
3. immediately issue batch-relevant NL queries
4. **disallow** background jobs from completing before the `T+0` measurement
5. measure ZPSQ and latency
6. then allow background jobs to complete and measure steady-state quality

### Why this works

It cleanly separates:

- **semantic capability without preprocessing**
- **semantic capability after preprocessing**

Dense pipelines may still win on fully indexed steady-state quality, but they should lose on `T+0` searchability.

## 6.8 Benchmark harness pseudocode

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StreamBatch {
    pub batch_id: u32,
    pub doc_ids: Vec<String>,
    pub arrival_unix_ms: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QueryCase {
    pub qid: String,
    pub text: String,
    pub relevant_doc_ids: Vec<String>,
    pub first_arrival_batch: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunResult {
    pub system: String,
    pub batch_id: u32,
    pub qid: String,
    pub searchable_at_ms: u64,
    pub queried_at_ms: u64,
    pub latency_ms: f32,
    pub recall_at_5: f32,
    pub ndcg_at_10: f32,
    pub events_per_query: Option<u32>,
    pub windows_per_query: Option<u32>,
}
```

## 6.9 Target numbers

These are ship targets, not literature claims.

### `T+0` targets on the expert set

| System | ZPSQ Recall@5 target | Comment |
|---|---:|---|
| ripgrep | 0.10–0.20 | lexical only |
| dense+kNN | 0.00 before indexing completes | strict fairness rule |
| Vespa semantic vector | 0.00 before embedding completes | strict fairness rule |
| Sieve | 0.30–0.50 | immediate semantic retrieval |

### steady-state targets

| System | Recall@5 target |
|---|---:|
| dense+kNN | 0.45–0.60 |
| Sieve | within 85–95% of dense baseline |
| ripgrep | lower but faster/simple |
| Vespa streaming lexical | near lexical baseline |

### latency targets

- Sieve p95 semantic query latency on 1M-doc subset: `<100 ms`
- ripgrep p95 lexical: competitive but semantically weaker
- dense ANN retrieval latency: often lower after indexing, but not immediate

## 6.10 Reporting template

Every experiment report should include:

- hardware
- corpus size
- batch size
- whether `T+0` forbids preprocessing completion
- query model used
- number of realized patterns/query
- events/query
- windows/query
- p50/p95 latency
- quality metrics

## 6.11 Success metrics for the benchmark itself

The benchmark should count as complete only if it is:

- **reproducible**: one command per system, same hardware class, same corpus stream;
- **freshness-sensitive**: `time_to_searchable_ms` is recorded explicitly;
- **semantic**: the primary query set is natural-language-to-code, not lexical paraphrase only;
- **comparative**: includes ripgrep, dense+kNN, Vespa streaming, and Sieve;
- **diagnostic**: emits events/query and windows/query for Sieve so scan quality can be analyzed, not just end metrics.

---

## Area 7. Post-Batch 3 known technical debt

These six fixes are not optional cleanup. They are foundational for the roadmap above.

## 7.1 Priority order

### P0 — merge before major R&D branches split
1. **(c) Add `content_type` to WAL entries**
2. **(a) Use corpus-level DF from Tantivy + WAL**
3. **(b) Add `CODE_DF_PRIOR`**

### P1 — merge before benchmark freeze
4. **(d) Adaptive window sizes by content type**
5. **(e) Anchor-preferred group center ordering**

### P2 — merge before full SPLADE-Code release
6. **(f) Initial-cap single-term variants in code mode**

## 7.2 Debt item (a): IDF degrades as entries move to indexed

### Problem

Current `compute_idf` blends scan-only DF from fresh/unindexed WAL entries with a static prior. As the corpus grows and content migrates into Tantivy shards, the fresh-scan portion shrinks and the prior dominates. IDF therefore gets worse as the corpus gets larger.

Current code confirms that `compute_idf` uses only scan-time DF counts plus a prior blend rather than indexed-corpus DF. Tantivy’s current APIs expose document frequency and total document counts, so Sieve should consume those directly. (see current Sieve code and Tantivy statistics APIs).

### Fix plan

Introduce a corpus-aware DF provider.

```rust
pub struct TantivyDfProvider {
    // wraps Tantivy searcher / field mapping
}

impl CorpusDfProvider for TantivyDfProvider {
    fn stats(&self, term: &str, ct: ContentType) -> DfStats {
        // indexed_df from Tantivy searcher.doc_freq(term)
        // indexed_docs from searcher.total_num_docs()
        // wal_df / wal_docs from fresh scan metadata
        // prior_df_frac from ct-specific prior table
        DfStats {
            indexed_df: 0,
            indexed_docs: 0,
            wal_df: 0,
            wal_docs: 0,
            prior_df_frac: 0.0,
        }
    }
}

pub fn compute_idf_v2(stats: DfStats) -> f32 {
    let total_docs = (stats.indexed_docs + stats.wal_docs).max(1) as f32;
    let df = (stats.indexed_df + stats.wal_df).max(1) as f32;
    let prior = stats.prior_df_frac.max(1e-6) * total_docs;

    let backed_off_df = if df < 4.0 { df.max(prior) } else { df };
    ((total_docs + 1.0) / (backed_off_df + 0.5)).ln() + 1.0
}
```

### Notes

- since the fresh scan should only cover unindexed WAL entries, overlap with Tantivy stats should be zero;
- prior becomes a true backoff, not a dominant substitute.

### Interaction with Area 1

This is a prerequisite for:
- post-expansion DF filter
- DF-FLOPS evaluation
- realistic scan-cost metrics

### Success metric

As corpus size grows, mean IDF calibration error should **decrease**, not increase.

## 7.3 Debt item (b): missing `CODE_DF_PRIOR` table

### Problem

Current priors include English words plus some programming identifiers, but common code terms such as `error`, `handler`, `config`, `impl`, `async`, and `result` are not penalized nearly enough in code contexts. That makes them act like rare gold nuggets when they are actually high-DF scan bombs.

Current code shows identifier priors but no dedicated code-context prior table. (see current `df_prior.rs` state).

### Fix plan

Add parallel prior tables:

```rust
pub static ENGLISH_DF_PRIOR: phf::Map<&'static str, f32> = /* existing + expanded */;
pub static CODE_DF_PRIOR: phf::Map<&'static str, f32> = /* new */;
pub static CONFIG_DF_PRIOR: phf::Map<&'static str, f32> = /* optional later */;
pub static LOG_DF_PRIOR: phf::Map<&'static str, f32> = /* optional later */;
```

### Seed values

Build `CODE_DF_PRIOR` from:

- CodeSearchNet
- CoIR-supporting code corpora
- local sampled repos when user runs `sieve train`

High-priority initial tokens:
`error`, `errors`, `handler`, `config`, `impl`, `async`, `await`, `result`, `ctx`, `req`, `resp`, `args`, `value`, `data`, `test`, `mock`, `client`, `server`, `service`, `manager`

### Interaction with Area 2

This is required for:
- code-aware DF filtering
- SPLADE-Code fusion
- accurate routing

### Success metric

On code queries, terms like `error` or `result` should no longer dominate retained non-anchor expansions.

## 7.4 Debt item (c): no `content_type` on WAL entries

### Problem

Current WAL metadata stores path, hash, byte offsets, lengths, timestamps, and line ranges, but no `content_type`. That blocks:

- code/prose model routing
- adaptive windows
- code-specific priors
- code-specific surface realization
- later train-on-corpus workflows

Current WAL metadata in code indeed lacks this field. (see current WAL metadata struct).

### Fix plan

Add the field now.

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WalMetaRecord {
    pub wal_entry_id: u64,
    pub source_path: String,
    pub content_hash: [u8; 32],
    pub byte_offset: u64,
    pub byte_length: u32,
    pub committed_at: u64,
    pub line_range_start: Option<u32>,
    pub line_range_end: Option<u32>,
    pub content_type: ContentType, // NEW
}
```

### Migration policy

- schema version bump
- backward-compatible deserialization:
  - missing field -> `Unknown`
- add offline migration command:
  - `sieve wal migrate --fill-content-type`

### Why P0

Retrofitting later will require WAL migration anyway, but with much more existing data and more downstream assumptions.

### Interaction

Required for Areas 2, 3, 4, 7d, and 7f.

### Success metric

After migration, ≥95% of newly ingested entries should have non-`Unknown` `content_type`.

## 7.5 Debt item (d): fixed window sizes regardless of content type

### Problem

Current code uses `WINDOW_BYTES = 512` and `WINDOW_STRIDE = 256` globally. That is wrong for mixed corpora:

- code windows are often too wide in bytes but too narrow in logical structure;
- prose windows are often larger than needed;
- config/logs have their own rhythms.

Current code confirms fixed 512/256 defaults. (see current fixed-window chunking implementation).

### Fix plan

Make windows content-aware.

```rust
pub fn profile_for_entry(ct: ContentType) -> WindowProfile {
    match ct {
        ContentType::Code   => WindowProfile { bytes: 256, stride: 128, max_events: 64 },
        ContentType::Prose  => WindowProfile { bytes: 512, stride: 256, max_events: 64 },
        ContentType::Config => WindowProfile { bytes: 256, stride: 128, max_events: 48 },
        ContentType::Log    => WindowProfile { bytes: 384, stride: 192, max_events: 64 },
        ContentType::Mixed | ContentType::Unknown =>
            WindowProfile { bytes: 384, stride: 192, max_events: 64 },
    }
}
```

### Optional next step

Introduce structure-aware code chunking later:
- function boundary hints
- brace-balanced windows
- comment/docstring boundary hints

But do **not** block the first fix on AST parsing.

### Interaction with Area 2

This directly improves code retrieval quality and is required before evaluating SPLADE-Code fairly.

### Success metric

- +2 to +5 Recall@10 on code
- fewer split-function misses
- no p95 latency regression >10%

## 7.6 Debt item (e): group center ordering prefers expansion positions over anchors

### Problem

Current ordered-pair bonus computes group centers from the highest-weight event in each group. If a high-weight expansion lands at byte 800 and the true anchor is at byte 200, the ordering score can fire incorrectly.

The current code path indeed selects group centers from highest-weight events, while only separately checking anchor flags for the pair bonus. (see current `group_centers` / ordered-bonus path).

### Fix plan

Prefer anchor centers when available.

```rust
pub fn summarize_group(events: &[MatchEvent]) -> GroupSummary {
    let mut anchor_center = None;
    let mut best_anchor_weight = f32::MIN;
    let mut fallback_center = 0u32;
    let mut best_any_weight = f32::MIN;

    for ev in events {
        if ev.weight > best_any_weight {
            best_any_weight = ev.weight;
            fallback_center = ev.byte_pos;
        }
        if ev.is_anchor && ev.weight > best_anchor_weight {
            best_anchor_weight = ev.weight;
            anchor_center = Some(ev.byte_pos);
        }
    }

    GroupSummary {
        group_id: events[0].group_id,
        anchor_center,
        fallback_center,
        best_anchor_weight,
        best_any_weight,
    }
}

pub fn group_center(summary: &GroupSummary) -> u32 {
    summary.anchor_center.unwrap_or(summary.fallback_center)
}
```

### Paper interaction

This design choice should be explicitly documented in Area 5’s paper. It is not just a bug fix; it is part of the formal definition of event interaction.

### Success metric

Ordered-pair false positives should drop noticeably on adversarial expansion-heavy queries.

## 7.7 Debt item (f): single-term case variants miss compound positions

### Problem

Surface realization currently generates lowercase / Title / UPPER variants for single terms but misses the initial-cap form needed inside compound identifiers like `ErrorHandler` or `TypeError`. Because the Aho-Corasick automaton is case-sensitive, `error` will not match `ErrorHandler` unless `Error` is emitted.

Current surface realization code confirms limited single-term case handling. (see current surface realization logic).

### Fix plan

In code mode, add initial-cap variants for single terms.

```rust
pub fn realize_single_term(term: &str, is_anchor: bool, ct: ContentType) -> Vec<Vec<u8>> {
    let mut out = vec![term.as_bytes().to_vec()];

    // existing lower/title/upper logic ...

    if matches!(ct, ContentType::Code | ContentType::Mixed) {
        if let Some(first) = term.chars().next() {
            if first.is_ascii_lowercase() {
                let mut chars = term.chars();
                let head = chars.next().unwrap().to_ascii_uppercase();
                let tail: String = chars.collect();
                out.push(format!("{}{}", head, tail).into_bytes());
            }
        }
    }

    out.sort();
    out.dedup();
    out
}
```

### Constraint

Only enable this in code/mixed contexts. Do not inflate prose scans.

### Interaction

Depends on 7c (`content_type`) and strengthens Area 2 even when SPLADE-Code is not yet active.

### Success metric

Improved retrieval on queries whose relevant code contains the term only inside `PascalCase`/`camelCase` compounds.

---

## 8. Build plan by area

## 8.1 First 2 weeks
- add `content_type` to WAL
- add Tantivy-backed DF provider
- add `CODE_DF_PRIOR`
- add adaptive windows
- add anchor-preferred centers
- add code-only initial-cap realization
- land benchmark harness skeleton

## 8.2 Weeks 3–5
- implement post-expansion DF filter
- collect benchmark baselines
- integrate content-aware routing
- integrate SPLADE-Code single-model path

## 8.3 Weeks 6–8
- hybrid prose/code fusion
- `sieve train` calibrate + small adaptation mode
- ONNX export + model registry

## 8.4 Weeks 9–12
- DF-FLOPS fine-tuning
- benchmark full scan-pattern metrics
- prepare public FreshBench-style eval package

## 8.5 Research branch
- sequential residual-state reranker
- full paper experiments
- artifact + reproducibility release

---

## 9. Recommended defaults

### Query-time defaults
- prose model: `sieve-splade-df-base`
- code model: `splade-code-06B` when GPU available
- max retained terms/query: 24 prose, 24 code
- adaptive windows: on
- anchor-preferred group centers: on
- DF filter: on

### Local training defaults
- `<100 MB`: calibrate only
- `100 MB–1 GB`: small adaptation
- `>1 GB`: full adaptation if corpus is reused frequently

### Benchmark defaults
- primary corpus: CodeSearchNet
- freshness metric: ZPSQ / `Recall@5 @ T+0`
- dense baseline: `bge-code-v1 + FAISS HNSW`
- lexical baseline: ripgrep
- production-like comparator: Vespa streaming

---

## 10. Final recommendation

The highest-leverage path is:

1. fix the six debt items now;
2. ship the post-expansion DF filter immediately;
3. make content type a first-class routing primitive;
4. integrate SPLADE-Code for code paths;
5. add `sieve train` for local corpora;
6. only then invest in DF-FLOPS fine-tuning and sequential residual-state reranking.

That sequence preserves Sieve’s identity. It does not dilute the core insight into a conventional offline semantic index. It sharpens the distinctive primitive: **query-compiled semantic scan over raw bytes, with late interaction over transient match events and zero document preprocessing on the freshness path.**

---

## 11. Selected references

### Sieve / implementation context
- Sieve public spec (`spec-v0.3`) and current codebase.
- Tantivy documentation for corpus statistics APIs (`doc_freq`, `total_num_docs`).

### Sparse retrieval
- Thibault Formal, Benjamin Piwowarski, Stéphane Clinchant. **SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking.** SIGIR 2021.
- Thibault Formal, Carlos Lassance, Benjamin Piwowarski, Stéphane Clinchant. **SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval.** 2021.
- Thong Nguyen, Sean MacAvaney, Andrew Yates. **A Unified Framework for Learned Sparse Retrieval.** 2023.
- Thibault Formal et al. **Towards Effective and Efficient Sparse Neural Information Retrieval.** TOIS 2024.

### DF-aware sparse learning
- DF-FLOPS paper, accepted at **SIGIR 2025**.

### Code sparse retrieval
- Simon Lupart, Maxime Louis, Thibault Formal, Hervé Déjean, Stéphane Clinchant. **On the Challenges and Opportunities of Learned Sparse Retrieval for Code.** 2026 (introduces SPLADE-Code).
- CoIR benchmark paper. **CoIR: A Comprehensive Benchmark for Code Information Retrieval.** 2024.

### Corpus adaptation
- Puxuan Yu, Antonio Mallia, Matthias Petri. **Improved Learned Sparse Retrieval with Corpus-Specific Vocabularies.** ECIR 2024.

### Long-document sparse retrieval
- Thong Nguyen, Sean MacAvaney, Andrew Yates. **Adapting Learned Sparse Retrieval for Long Documents.** SIGIR 2023.
- Emmanouil Georgios Lionis, Jia-Huei Ju. **On the Reproducibility of Learned Sparse Retrieval Adaptations for Long Documents.** 2025.

### Late interaction / routing
- Omar Khattab, Matei Zaharia. **ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT.** 2020.
- Luyu Gao, Zhuyun Dai, Jamie Callan. **COIL: Revisit Exact Lexical Match in Information Retrieval with Contextualized Inverted List.** NAACL 2021.
- Minghan Li et al. **CITADEL: Conditional Token Interaction via Dynamic Lexical Routing for Efficient and Effective Multi-Vector Retrieval.** 2022.
- Minghan Li et al. **SLIM: Sparsified Late Interaction for Multi-Vector Retrieval with Inverted Indexes.** SIGIR 2023.

### Sequential residual-state inspiration
- Microsoft Research. **MEMENTO** and associated paper / code release, 2026.

### Benchmark corpus
- Hamel Husain et al. **CodeSearchNet Challenge: Evaluating the State of Semantic Code Search.** 2019.
