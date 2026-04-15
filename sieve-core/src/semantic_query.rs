use std::path::Path;

pub const MAX_SEED_GROUPS: usize = 6;
pub const MAX_TERMS: usize = 96;
pub const MAX_ANCHORS: usize = 12;
pub const MAX_PHRASES: usize = 8;
pub const MAX_VARIANTS_PER_TERM: usize = 6;
pub const ANCHOR_WEIGHT_RATIO: f32 = 0.55;
pub const MIN_TERM_WEIGHT_RATIO: f32 = 0.18;
pub const MIN_AUX_GROUP_RATIO: f32 = 0.35;
pub const SEED_CLAIM_RATIO: f32 = 0.20;

#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum ContentType {
    Code,
    Prose,
    Config,
    Log,
    Mixed,
    #[default]
    Unknown,
}

pub fn detect_content_type(path: &Path) -> ContentType {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some(
            "rs" | "py" | "ts" | "tsx" | "js" | "jsx" | "go" | "c" | "cpp" | "h" | "hpp" | "java"
            | "kt" | "swift" | "rb" | "php" | "cs" | "scala" | "zig" | "lua" | "sh" | "bash"
            | "zsh" | "pl" | "r",
        ) => ContentType::Code,
        Some("md" | "txt" | "rst" | "tex" | "adoc" | "org") => ContentType::Prose,
        Some("toml" | "yaml" | "yml" | "json" | "ini" | "cfg" | "conf" | "env" | "properties") => {
            ContentType::Config
        }
        Some("log") => ContentType::Log,
        _ => ContentType::Unknown,
    }
}

impl ContentType {
    pub fn from_path(path: &str) -> Self {
        detect_content_type(Path::new(path))
    }

    pub fn is_code_like(self) -> bool {
        matches!(self, Self::Code | Self::Mixed)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SemanticCompileOptions {
    pub no_expand: bool,
    pub random_expansion: bool,
}

pub type TermId = u16;
pub type GroupId = u16;
pub type PhraseId = u16;

#[derive(Debug, Clone, PartialEq)]
pub struct SemanticQuery {
    pub raw_query: String,
    pub normalized_query: String,
    pub content_type: ContentType,
    pub tokens: Vec<QueryToken>,
    pub seeds: Vec<SeedToken>,
    pub groups: Vec<SemanticGroup>,
    pub terms: Vec<SemanticTerm>,
    pub phrases: Vec<PhrasePattern>,
    pub query_order: Vec<GroupId>,
    pub total_group_importance: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SeedToken {
    pub text: String,
    pub ordinal: u8,
    pub group_id: GroupId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TokenClass {
    Word,
    Namespace,
    Dotted,
    Numeric,
    Compound,
    Language,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryToken {
    pub text: String,
    pub is_anchor: bool,
    pub token_class: TokenClass,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SemanticGroup {
    pub group_id: GroupId,
    pub canonical: String,
    pub query_ordinal: u8,
    pub is_seed: bool,
    pub importance: f32,
    pub member_terms: Vec<TermId>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SemanticTerm {
    pub term_id: TermId,
    pub vocab_id: u32,
    pub vocab_piece: String,
    pub canonical: String,
    pub raw_weight: f32,
    pub norm_weight: f32,
    pub group_id: GroupId,
    pub is_anchor: bool,
    pub source: TermSource,
    pub surface_variants: Vec<crate::surface::SurfaceVariant>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TermSource {
    OriginalToken,
    SparseAnchor,
    SparseExpansion,
    AliasExpansion,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PhrasePattern {
    pub phrase_id: PhraseId,
    pub canonical: String,
    pub component_groups: Vec<GroupId>,
    pub raw_weight: f32,
    pub norm_weight: f32,
    pub is_anchor: bool,
    pub surface_variants: Vec<crate::surface::SurfaceVariant>,
}

pub fn compile_semantic_query(
    raw_query: &str,
    sparse: &crate::sparse::SpladeEncoder,
    aliases: &crate::aliases::AliasLexicon,
    content_type: ContentType,
) -> crate::Result<SemanticQuery> {
    compile_semantic_query_with_options(
        raw_query,
        sparse,
        aliases,
        SemanticCompileOptions::default(),
        content_type,
    )
}

pub fn compile_semantic_query_with_options(
    raw_query: &str,
    sparse: &crate::sparse::SpladeEncoder,
    aliases: &crate::aliases::AliasLexicon,
    options: SemanticCompileOptions,
    content_type: ContentType,
) -> crate::Result<SemanticQuery> {
    compile_semantic_query_with(
        raw_query,
        &|text| sparse.encode(text),
        &|vocab_id| sparse.vocab_piece(vocab_id).map(str::to_string),
        aliases,
        options,
        content_type,
    )
}

pub fn compile_semantic_query_with<E, V>(
    raw_query: &str,
    encode: &E,
    vocab_piece: &V,
    aliases: &crate::aliases::AliasLexicon,
    options: SemanticCompileOptions,
    content_type: ContentType,
) -> crate::Result<SemanticQuery>
where
    E: Fn(&str) -> crate::Result<Vec<(u32, f32)>>,
    V: Fn(u32) -> Option<String>,
{
    let normalized_query = normalize_query(raw_query);
    let tokens = tokenize_query(&normalized_query, content_type);
    let seed_texts = tokens
        .iter()
        .map(|token| token.text.clone())
        .take(MAX_SEED_GROUPS)
        .collect::<Vec<_>>();
    if seed_texts.is_empty() {
        return Ok(SemanticQuery {
            raw_query: raw_query.to_string(),
            normalized_query,
            content_type,
            tokens,
            seeds: Vec::new(),
            groups: Vec::new(),
            terms: Vec::new(),
            phrases: Vec::new(),
            query_order: Vec::new(),
            total_group_importance: 0.0,
        });
    }
    if options.no_expand {
        return Ok(seed_only_query(
            raw_query,
            normalized_query,
            &tokens,
            &seed_texts,
            content_type,
        ));
    }

    let full_weights = encode(&normalized_query)?;
    let full_max_weight = full_weights
        .iter()
        .map(|(_, weight)| *weight)
        .fold(0.0f32, f32::max);
    if full_max_weight <= 0.0 {
        return Ok(seed_only_query(
            raw_query,
            normalized_query,
            &tokens,
            &seed_texts,
            content_type,
        ));
    }

    let mut seeds = Vec::new();
    let mut groups = Vec::new();
    for (index, seed) in seed_texts.iter().enumerate() {
        let group_id = index as GroupId;
        seeds.push(SeedToken {
            text: seed.clone(),
            ordinal: index as u8,
            group_id,
        });
        groups.push(SemanticGroup {
            group_id,
            canonical: seed.clone(),
            query_ordinal: index as u8,
            is_seed: true,
            importance: 0.0,
            member_terms: Vec::new(),
        });
    }

    let mut aux_group_map: std::collections::HashMap<String, GroupId> =
        std::collections::HashMap::new();
    let mut terms = Vec::new();
    let mut seen_groups_for_piece: std::collections::HashMap<u32, GroupId> =
        std::collections::HashMap::new();
    let mut seen_seed_terms = std::collections::HashSet::new();

    for (seed_index, seed) in seed_texts.iter().enumerate() {
        let seed_group_id = seed_index as GroupId;
        let seed_best = best_matching_seed_piece(&full_weights, seed, vocab_piece);
        if let Some((vocab_id, raw_weight, piece)) = seed_best {
            let norm_weight = if full_max_weight > 0.0 {
                (raw_weight / full_max_weight).clamp(0.0, 1.0)
            } else {
                1.0
            };
            push_term(
                &mut terms,
                &mut groups,
                &mut seen_seed_terms,
                *seen_groups_for_piece
                    .entry(vocab_id)
                    .or_insert(seed_group_id),
                vocab_id,
                piece,
                seed.clone(),
                raw_weight,
                norm_weight.max(ANCHOR_WEIGHT_RATIO),
                true,
                TermSource::OriginalToken,
            );
        } else {
            push_term(
                &mut terms,
                &mut groups,
                &mut seen_seed_terms,
                seed_group_id,
                u32::MAX - seed_index as u32,
                seed.clone(),
                seed.clone(),
                full_max_weight,
                1.0,
                true,
                TermSource::OriginalToken,
            );
        }
    }

    let mut retained = full_weights
        .into_iter()
        .filter_map(|(vocab_id, raw_weight)| {
            let piece = vocab_piece(vocab_id)?;
            if is_special_token(&piece) || raw_weight <= 0.0 {
                return None;
            }
            if raw_weight < MIN_TERM_WEIGHT_RATIO * full_max_weight {
                return None;
            }
            Some((vocab_id, raw_weight, piece))
        })
        .collect::<Vec<_>>();
    if options.random_expansion {
        for (_, raw_weight, piece) in &mut retained {
            let floor = MIN_TERM_WEIGHT_RATIO * full_max_weight;
            let rand = deterministic_unit_interval(raw_query, piece);
            *raw_weight = floor + (full_max_weight - floor).max(0.0) * rand;
        }
    }
    retained.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    retained.truncate(MAX_TERMS);

    let mut anchor_count = terms.iter().filter(|term| term.is_anchor).count();

    for (vocab_id, raw_weight, piece) in retained {
        let canonical = normalize_piece(&piece);
        if canonical.is_empty() {
            continue;
        }
        let norm_weight = (raw_weight / full_max_weight).clamp(0.0, 1.0);
        let maybe_claim = claim_group_for_term(&canonical, &seed_texts, aliases);
        let Some((group_id, source)) = maybe_claim.or_else(|| {
            if norm_weight >= MIN_AUX_GROUP_RATIO {
                let group_id = *aux_group_map.entry(canonical.clone()).or_insert_with(|| {
                    let group_id = groups.len() as GroupId;
                    groups.push(SemanticGroup {
                        group_id,
                        canonical: canonical.clone(),
                        query_ordinal: u8::MAX,
                        is_seed: false,
                        importance: 0.0,
                        member_terms: Vec::new(),
                    });
                    group_id
                });
                Some((group_id, TermSource::SparseExpansion))
            } else {
                None
            }
        }) else {
            continue;
        };

        if seen_groups_for_piece
            .get(&vocab_id)
            .is_some_and(|existing| *existing != group_id)
        {
            continue;
        }
        seen_groups_for_piece.insert(vocab_id, group_id);

        let is_anchor = if anchor_count < MAX_ANCHORS {
            norm_weight >= ANCHOR_WEIGHT_RATIO
        } else {
            false
        };
        if is_anchor {
            anchor_count += 1;
        }
        push_term(
            &mut terms,
            &mut groups,
            &mut seen_seed_terms,
            group_id,
            vocab_id,
            piece,
            canonical,
            raw_weight,
            norm_weight,
            is_anchor,
            source,
        );
    }

    for group in &mut groups {
        group.importance = group
            .member_terms
            .iter()
            .filter_map(|term_id| terms.get(*term_id as usize))
            .map(|term| term.norm_weight)
            .fold(0.0f32, f32::max);
    }

    let query_order = seeds.iter().map(|seed| seed.group_id).collect::<Vec<_>>();
    let phrases = build_phrases(&seed_texts, &groups, &terms, &query_order);
    let total_group_importance = groups.iter().map(|group| group.importance).sum();
    Ok(SemanticQuery {
        raw_query: raw_query.to_string(),
        normalized_query,
        content_type,
        tokens,
        seeds,
        groups,
        terms,
        phrases,
        query_order,
        total_group_importance,
    })
}

fn seed_only_query(
    raw_query: &str,
    normalized_query: String,
    tokens: &[QueryToken],
    seed_texts: &[String],
    content_type: ContentType,
) -> SemanticQuery {
    let mut seeds = Vec::new();
    let mut groups = Vec::new();
    let mut terms = Vec::new();
    for (index, seed) in seed_texts.iter().enumerate() {
        let group_id = index as GroupId;
        let term_id = index as TermId;
        seeds.push(SeedToken {
            text: seed.clone(),
            ordinal: index as u8,
            group_id,
        });
        groups.push(SemanticGroup {
            group_id,
            canonical: seed.clone(),
            query_ordinal: index as u8,
            is_seed: true,
            importance: 1.0,
            member_terms: vec![term_id],
        });
        terms.push(SemanticTerm {
            term_id,
            vocab_id: u32::MAX - index as u32,
            vocab_piece: seed.clone(),
            canonical: seed.clone(),
            raw_weight: 1.0,
            norm_weight: 1.0,
            group_id,
            is_anchor: true,
            source: TermSource::OriginalToken,
            surface_variants: Vec::new(),
        });
    }
    let query_order = seeds.iter().map(|seed| seed.group_id).collect::<Vec<_>>();
    let phrases = build_phrases(seed_texts, &groups, &terms, &query_order);
    let total_group_importance = groups.iter().map(|group| group.importance).sum();
    SemanticQuery {
        raw_query: raw_query.to_string(),
        normalized_query,
        content_type,
        tokens: tokens.to_vec(),
        seeds,
        groups,
        terms,
        phrases,
        query_order,
        total_group_importance,
    }
}

impl SemanticQuery {
    pub fn seed_only(raw_query: &str, content_type: ContentType) -> Self {
        let normalized_query = normalize_query(raw_query);
        let tokens = tokenize_query(&normalized_query, content_type);
        let seed_texts = tokens
            .iter()
            .map(|token| token.text.clone())
            .take(MAX_SEED_GROUPS)
            .collect::<Vec<_>>();
        seed_only_query(
            raw_query,
            normalized_query,
            &tokens,
            &seed_texts,
            content_type,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn push_term(
    terms: &mut Vec<SemanticTerm>,
    groups: &mut [SemanticGroup],
    seen_seed_terms: &mut std::collections::HashSet<(GroupId, String)>,
    group_id: GroupId,
    vocab_id: u32,
    vocab_piece: String,
    canonical: String,
    raw_weight: f32,
    norm_weight: f32,
    is_anchor: bool,
    source: TermSource,
) {
    let key = (group_id, canonical.clone());
    if !seen_seed_terms.insert(key) {
        return;
    }
    let term_id = terms.len() as TermId;
    terms.push(SemanticTerm {
        term_id,
        vocab_id,
        vocab_piece,
        canonical,
        raw_weight,
        norm_weight,
        group_id,
        is_anchor,
        source,
        surface_variants: Vec::new(),
    });
    if let Some(group) = groups.iter_mut().find(|group| group.group_id == group_id) {
        group.member_terms.push(term_id);
    }
}

fn best_matching_seed_piece<V>(
    weights: &[(u32, f32)],
    seed: &str,
    vocab_piece: &V,
) -> Option<(u32, f32, String)>
where
    V: Fn(u32) -> Option<String>,
{
    weights.iter().find_map(|(id, weight)| {
        let piece = vocab_piece(*id)?;
        let canonical = normalize_piece(&piece);
        if canonical == seed {
            Some((*id, *weight, piece))
        } else {
            None
        }
    })
}

fn claim_group_for_term(
    canonical: &str,
    seed_texts: &[String],
    aliases: &crate::aliases::AliasLexicon,
) -> Option<(GroupId, TermSource)> {
    if let Some(index) = seed_texts.iter().position(|seed| seed == canonical) {
        return Some((index as GroupId, TermSource::OriginalToken));
    }
    for (seed_index, seed) in seed_texts.iter().enumerate() {
        if aliases.same_alias_family(seed, canonical) || aliases.same_alias_family(canonical, seed)
        {
            return Some((seed_index as GroupId, TermSource::AliasExpansion));
        }
    }
    None
}

fn build_phrases(
    seed_texts: &[String],
    groups: &[SemanticGroup],
    terms: &[SemanticTerm],
    query_order: &[GroupId],
) -> Vec<PhrasePattern> {
    let mut phrases = Vec::new();
    if (2..=5).contains(&seed_texts.len()) {
        phrases.push(PhrasePattern {
            phrase_id: phrases.len() as PhraseId,
            canonical: seed_texts.join(" "),
            component_groups: query_order.to_vec(),
            raw_weight: 1.0,
            norm_weight: 1.0,
            is_anchor: true,
            surface_variants: Vec::new(),
        });
    }
    for pair in query_order.windows(2) {
        if phrases.len() >= MAX_PHRASES {
            break;
        }
        let canonical = pair
            .iter()
            .filter_map(|group_id| groups.iter().find(|group| group.group_id == *group_id))
            .map(|group| group.canonical.clone())
            .collect::<Vec<_>>()
            .join(" ");
        if canonical.split_whitespace().count() == 2 {
            phrases.push(PhrasePattern {
                phrase_id: phrases.len() as PhraseId,
                canonical,
                component_groups: pair.to_vec(),
                raw_weight: 0.9,
                norm_weight: 0.9,
                is_anchor: true,
                surface_variants: Vec::new(),
            });
        }
    }
    let mut substitutions = 0usize;
    for (seed_pos, group_id) in query_order.iter().enumerate() {
        if phrases.len() >= MAX_PHRASES || substitutions >= 2 {
            break;
        }
        let Some(expansion) = terms
            .iter()
            .filter(|term| term.group_id == *group_id && term.is_anchor)
            .filter(|term| !matches!(term.source, TermSource::OriginalToken))
            .max_by(|a, b| {
                a.norm_weight
                    .partial_cmp(&b.norm_weight)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        else {
            continue;
        };
        let mut parts = seed_texts.to_vec();
        parts[seed_pos] = expansion.canonical.clone();
        phrases.push(PhrasePattern {
            phrase_id: phrases.len() as PhraseId,
            canonical: parts.join(" "),
            component_groups: query_order.to_vec(),
            raw_weight: expansion.raw_weight,
            norm_weight: expansion.norm_weight,
            is_anchor: expansion.is_anchor,
            surface_variants: Vec::new(),
        });
        substitutions += 1;
    }
    phrases.truncate(MAX_PHRASES);
    phrases
}

pub fn tokenize_query(query: &str, content_type: ContentType) -> Vec<QueryToken> {
    match content_type {
        ContentType::Code | ContentType::Mixed => tokenize_code_query(query),
        ContentType::Prose | ContentType::Config | ContentType::Log | ContentType::Unknown => {
            tokenize_prose_query(query)
        }
    }
}

fn tokenize_prose_query(query: &str) -> Vec<QueryToken> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for ch in query.chars() {
        if ch.is_alphanumeric() {
            current.push(ch);
        } else if !current.is_empty() {
            maybe_push_prose_token(&mut tokens, &mut current);
        }
    }
    if !current.is_empty() {
        maybe_push_prose_token(&mut tokens, &mut current);
    }
    dedup_query_tokens(tokens)
}

fn tokenize_code_query(query: &str) -> Vec<QueryToken> {
    let mut tokens = Vec::new();
    for raw in query.split_whitespace() {
        let Some(primary) = normalize_code_query_token(raw) else {
            continue;
        };
        push_query_token(
            &mut tokens,
            primary.clone(),
            true,
            classify_code_token(&primary),
        );
        for seed in secondary_code_seeds(&primary) {
            push_query_token(&mut tokens, seed.clone(), false, classify_code_token(&seed));
        }
    }
    dedup_query_tokens(tokens)
}

fn dedup_query_tokens(tokens: Vec<QueryToken>) -> Vec<QueryToken> {
    let mut seen = std::collections::HashMap::<String, usize>::new();
    let mut deduped: Vec<QueryToken> = Vec::new();
    for token in tokens {
        match seen.get(&token.text).copied() {
            Some(index) => {
                deduped[index].is_anchor |= token.is_anchor;
            }
            None => {
                seen.insert(token.text.clone(), deduped.len());
                deduped.push(token);
            }
        }
    }
    deduped
}

fn maybe_push_prose_token(tokens: &mut Vec<QueryToken>, current: &mut String) {
    let token = current.to_ascii_lowercase();
    current.clear();
    if token.len() >= 2 && !is_stopword(&token) {
        push_query_token(tokens, token, true, TokenClass::Word);
    }
}

fn push_query_token(
    tokens: &mut Vec<QueryToken>,
    text: String,
    is_anchor: bool,
    token_class: TokenClass,
) {
    if text.len() < 2 {
        return;
    }
    tokens.push(QueryToken {
        text,
        is_anchor,
        token_class,
    });
}

fn normalize_code_query_token(raw: &str) -> Option<String> {
    let trimmed = raw.trim_matches(|c: char| {
        !c.is_ascii_alphanumeric()
            && c != '_'
            && c != ':'
            && c != '.'
            && c != '#'
            && c != '+'
            && c != '-'
    });
    if trimmed.is_empty() {
        return None;
    }
    let lowered = trimmed.to_ascii_lowercase();
    match lowered.as_str() {
        "c++" => Some("cpp".to_string()),
        "c#" => Some("csharp".to_string()),
        _ => Some(lowered),
    }
}

fn classify_code_token(token: &str) -> TokenClass {
    if token.chars().all(|ch| ch.is_ascii_digit()) {
        TokenClass::Numeric
    } else if token.contains("::") {
        TokenClass::Namespace
    } else if token.contains('.') {
        TokenClass::Dotted
    } else if token.contains('_') || token.chars().any(|ch| ch.is_ascii_digit()) {
        TokenClass::Compound
    } else if matches!(token, "cpp" | "csharp") {
        TokenClass::Language
    } else {
        TokenClass::Word
    }
}

fn secondary_code_seeds(token: &str) -> Vec<String> {
    let mut seeds = Vec::new();
    if token.contains("::") {
        seeds.extend(token.split("::").map(str::to_string));
    }
    if token.contains('.') {
        seeds.extend(token.split('.').map(str::to_string));
    }
    if token.contains('_') {
        seeds.extend(token.split('_').map(str::to_string));
    }
    seeds.retain(|seed| seed.len() >= 2);
    seeds.sort();
    seeds.dedup();
    seeds
}

fn normalize_query(query: &str) -> String {
    let mut collected = Vec::new();
    let mut saw_content = false;
    for raw_line in query.lines() {
        let line = raw_line.trim();
        if line.is_empty() {
            if saw_content {
                break;
            }
            continue;
        }
        let lower = line.to_ascii_lowercase();
        if is_doc_section_header(&lower) {
            break;
        }
        saw_content = true;
        collected.push(line.to_string());
        if line.ends_with('.') || line.ends_with('?') || line.ends_with('!') {
            break;
        }
    }

    let base = if collected.is_empty() {
        query.trim().to_string()
    } else {
        collected.join(" ")
    };
    base.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

fn is_doc_section_header(line: &str) -> bool {
    matches!(
        line.trim_end_matches(':'),
        "args"
            | "arguments"
            | "params"
            | "parameters"
            | "returns"
            | "return"
            | "raises"
            | "raise"
            | "examples"
            | "example"
            | "notes"
            | "note"
    )
}

fn normalize_piece(piece: &str) -> String {
    piece
        .trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '#' && c != '_' && c != '-')
        .trim_start_matches("##")
        .to_ascii_lowercase()
}

fn deterministic_unit_interval(raw_query: &str, piece: &str) -> f32 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(raw_query.as_bytes());
    hasher.update(&[0]);
    hasher.update(piece.as_bytes());
    let bytes = hasher.finalize();
    let n = u32::from_le_bytes([
        bytes.as_bytes()[0],
        bytes.as_bytes()[1],
        bytes.as_bytes()[2],
        bytes.as_bytes()[3],
    ]);
    n as f32 / u32::MAX as f32
}

fn is_special_token(piece: &str) -> bool {
    matches!(piece, "[PAD]" | "[UNK]" | "[CLS]" | "[SEP]" | "[MASK]")
        || (piece.starts_with('[') && piece.ends_with(']'))
        || (piece.starts_with('<') && piece.ends_with('>'))
}

fn is_stopword(token: &str) -> bool {
    matches!(
        token,
        "a" | "an"
            | "and"
            | "are"
            | "as"
            | "at"
            | "be"
            | "by"
            | "for"
            | "from"
            | "if"
            | "in"
            | "is"
            | "it"
            | "of"
            | "on"
            | "or"
            | "the"
            | "to"
            | "with"
    )
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::{
        compile_semantic_query_with, normalize_query, ContentType, SemanticCompileOptions,
        TermSource,
    };

    fn mock_vocab(id: u32) -> Option<String> {
        Some(
            match id {
                1 => "failure",
                2 => "handling",
                3 => "error",
                4 => "retry",
                5 => "module",
                6 => "library",
                7 => "package",
                _ => return None,
            }
            .to_string(),
        )
    }

    #[test]
    fn test_semantic_query_compilation() {
        let aliases = crate::aliases::AliasLexicon::built_in();
        let query = compile_semantic_query_with(
            "Failure handling",
            &|text| match text {
                "failure handling" => Ok(vec![(1, 2.0), (2, 1.7), (3, 1.1), (4, 0.8)]),
                "failure" => Ok(vec![(1, 2.2), (3, 1.2)]),
                "handling" => Ok(vec![(2, 2.1), (4, 1.0)]),
                _ => Ok(Vec::new()),
            },
            &mock_vocab,
            &aliases,
            SemanticCompileOptions::default(),
            ContentType::Prose,
        )
        .unwrap();
        assert_eq!(query.seeds.len(), 2);
        assert!(query.groups.len() >= 2);
        assert!(query
            .terms
            .iter()
            .any(|term| term.canonical == "failure" && term.is_anchor));
        assert!(query
            .terms
            .iter()
            .any(|term| term.canonical == "handling" && term.is_anchor));
        assert!(query
            .terms
            .iter()
            .any(|term| term.canonical == "error"
                && matches!(term.source, TermSource::SparseExpansion)));
        assert!(!query.phrases.is_empty());
    }

    #[test]
    fn test_semantic_group_formation() {
        let aliases = crate::aliases::AliasLexicon::built_in();
        let query = compile_semantic_query_with(
            "pkg mod",
            &|text| match text {
                "pkg mod" => Ok(vec![(7, 2.0), (5, 1.6), (6, 1.5)]),
                "pkg" => Ok(vec![(7, 2.1), (6, 1.8)]),
                "mod" => Ok(vec![(5, 2.0)]),
                _ => Ok(Vec::new()),
            },
            &mock_vocab,
            &aliases,
            SemanticCompileOptions::default(),
            ContentType::Prose,
        )
        .unwrap();
        assert_eq!(query.seeds.len(), 2);
        assert!(query
            .terms
            .iter()
            .any(|term| term.canonical == "pkg" && term.group_id == 0 && term.is_anchor));
        assert!(query
            .terms
            .iter()
            .any(|term| term.canonical == "module" && term.group_id == 1));
        assert!(query.groups.iter().all(|group| group.importance > 0.0));
    }

    #[test]
    fn test_normalize_query_strips_doc_sections() {
        let normalized = normalize_query(
            "Calculate the batched KL divergence KL(a || b) with a and b Gumbel.\n\nArgs:\n  a: instance\nReturns:\n  Batchwise KL(a || b)",
        );
        assert_eq!(
            normalized,
            "calculate the batched kl divergence kl(a || b) with a and b gumbel."
        );
    }

    #[test]
    fn test_semantic_query_uses_single_encode_call() {
        let aliases = crate::aliases::AliasLexicon::built_in();
        let calls = AtomicUsize::new(0);
        let query = compile_semantic_query_with(
            "failure handling retry",
            &|text| {
                calls.fetch_add(1, Ordering::SeqCst);
                match text {
                    "failure handling retry" => Ok(vec![(1, 2.0), (2, 1.7), (3, 1.1), (4, 0.8)]),
                    _ => Ok(Vec::new()),
                }
            },
            &mock_vocab,
            &aliases,
            SemanticCompileOptions::default(),
            ContentType::Prose,
        )
        .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(!query.terms.is_empty());
    }
}
