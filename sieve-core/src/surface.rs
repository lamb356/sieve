use crate::semantic_query::{GroupId, PhraseId, TermId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryMode {
    None,
    Word,
    Identifier,
    CodeSubtoken,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VariantKind {
    Lower,
    Title,
    Upper,
    Snake,
    ScreamingSnake,
    Kebab,
    Camel,
    Pascal,
    Alias,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceVariant {
    pub text: String,
    pub bytes: Vec<u8>,
    pub kind: VariantKind,
    pub boundary: BoundaryMode,
    pub quality: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RealizedPattern {
    pub pattern_id: u32,
    pub term_id: Option<TermId>,
    pub phrase_id: Option<PhraseId>,
    pub primary_group_id: GroupId,
    pub bytes: Vec<u8>,
    pub weight: f32,
    pub is_anchor: bool,
    pub boundary: BoundaryMode,
}

pub fn realize_surfaces(
    query: &mut crate::semantic_query::SemanticQuery,
    df_prior: &dyn Fn(&str) -> f32,
) -> Vec<RealizedPattern> {
    let seed_terms: std::collections::HashSet<String> = query
        .groups
        .iter()
        .filter(|group| group.is_seed)
        .map(|group| group.canonical.clone())
        .collect();

    let alias_terms: std::collections::HashSet<String> = query
        .terms
        .iter()
        .filter(|term| {
            matches!(
                term.source,
                crate::semantic_query::TermSource::AliasExpansion
            )
        })
        .map(|term| normalize_text(&term.canonical))
        .collect();

    let mut realized = Vec::new();
    let mut seen: std::collections::HashMap<Vec<u8>, (usize, f32)> =
        std::collections::HashMap::new();

    for term in &mut query.terms {
        let variants = realize_term_variants(
            term,
            query.content_type,
            &seed_terms,
            &alias_terms,
            df_prior,
        );
        term.surface_variants = variants.clone();
        for variant in variants {
            push_pattern(
                &mut realized,
                &mut seen,
                term.group_id,
                Some(term.term_id),
                None,
                term.norm_weight,
                term.is_anchor,
                variant,
            );
        }
    }

    for phrase in &mut query.phrases {
        let primary_group_id = phrase.component_groups.first().copied().unwrap_or_default();
        let variants = realize_phrase_variants(phrase, df_prior);
        phrase.surface_variants = variants.clone();
        for variant in variants {
            push_pattern(
                &mut realized,
                &mut seen,
                primary_group_id,
                None,
                Some(phrase.phrase_id),
                phrase.norm_weight,
                phrase.is_anchor,
                variant,
            );
        }
    }

    for (index, pattern) in realized.iter_mut().enumerate() {
        pattern.pattern_id = index as u32;
    }

    realized
}

#[allow(clippy::too_many_arguments)]
fn push_pattern(
    realized: &mut Vec<RealizedPattern>,
    seen: &mut std::collections::HashMap<Vec<u8>, (usize, f32)>,
    primary_group_id: GroupId,
    term_id: Option<TermId>,
    phrase_id: Option<PhraseId>,
    weight: f32,
    is_anchor: bool,
    variant: SurfaceVariant,
) {
    let bytes = variant.bytes.clone();
    if let Some((index, existing_weight)) = seen.get(&bytes).copied() {
        if weight <= existing_weight {
            return;
        }
        realized[index] = RealizedPattern {
            pattern_id: index as u32,
            term_id,
            phrase_id,
            primary_group_id,
            bytes,
            weight,
            is_anchor,
            boundary: variant.boundary,
        };
        seen.insert(realized[index].bytes.clone(), (index, weight));
        return;
    }

    let index = realized.len();
    realized.push(RealizedPattern {
        pattern_id: index as u32,
        term_id,
        phrase_id,
        primary_group_id,
        bytes: bytes.clone(),
        weight,
        is_anchor,
        boundary: variant.boundary,
    });
    seen.insert(bytes, (index, weight));
}

fn realize_term_variants(
    term: &crate::semantic_query::SemanticTerm,
    content_type: crate::semantic_query::ContentType,
    seed_terms: &std::collections::HashSet<String>,
    alias_terms: &std::collections::HashSet<String>,
    df_prior: &dyn Fn(&str) -> f32,
) -> Vec<SurfaceVariant> {
    let raw = strip_vocab_piece(&term.vocab_piece);
    if raw.is_empty() {
        return Vec::new();
    }
    let norm = normalize_text(&raw);
    if reject_term_fragment(&term.vocab_piece, &norm, seed_terms, alias_terms) {
        return Vec::new();
    }
    let mut variants = Vec::new();
    add_variant(
        &mut variants,
        &norm,
        VariantKind::Lower,
        BoundaryMode::Identifier,
        quality_score(
            &norm,
            df_prior(&norm),
            term.norm_weight,
            term.is_anchor,
            false,
            term.vocab_piece.starts_with("##"),
        ),
    );
    if norm.len() >= 4 {
        let title = to_title_case(&norm);
        add_variant(
            &mut variants,
            &title,
            VariantKind::Title,
            BoundaryMode::Identifier,
            quality_score(
                &title,
                df_prior(&norm),
                term.norm_weight,
                term.is_anchor,
                false,
                false,
            ),
        );
    }
    if norm.len() <= 24 && !is_stopword(&norm) && term.is_anchor {
        let upper = norm.to_ascii_uppercase();
        add_variant(
            &mut variants,
            &upper,
            VariantKind::Upper,
            BoundaryMode::Identifier,
            quality_score(
                &upper,
                df_prior(&norm),
                term.norm_weight,
                term.is_anchor,
                false,
                false,
            ),
        );
    }
    if content_type.is_code_like() && !term.is_anchor {
        for variant in realize_code_subtoken(&norm) {
            let kind = if variant.chars().all(|ch| ch.is_ascii_uppercase()) {
                VariantKind::Upper
            } else if variant
                .chars()
                .next()
                .is_some_and(|ch| ch.is_ascii_uppercase())
            {
                VariantKind::Title
            } else {
                VariantKind::Alias
            };
            add_variant(
                &mut variants,
                &variant,
                kind,
                BoundaryMode::CodeSubtoken,
                quality_score(
                    &variant,
                    df_prior(&variant.to_ascii_lowercase()),
                    term.norm_weight,
                    false,
                    true,
                    false,
                ),
            );
        }
    }
    variants
}

fn realize_phrase_variants(
    phrase: &crate::semantic_query::PhrasePattern,
    df_prior: &dyn Fn(&str) -> f32,
) -> Vec<SurfaceVariant> {
    let tokens: Vec<String> = phrase
        .canonical
        .split_whitespace()
        .map(normalize_text)
        .filter(|token| !token.is_empty())
        .collect();
    if tokens.len() < 2 || tokens.len() > 4 {
        return Vec::new();
    }
    let joined = tokens.join(" ");
    let prior = df_prior(&joined);
    let mut variants = Vec::new();
    let lower = joined.clone();
    add_variant(
        &mut variants,
        &lower,
        VariantKind::Lower,
        BoundaryMode::Word,
        quality_score(
            &lower,
            prior,
            phrase.norm_weight,
            phrase.is_anchor,
            false,
            false,
        ),
    );
    let snake = tokens.join("_");
    add_variant(
        &mut variants,
        &snake,
        VariantKind::Snake,
        BoundaryMode::Identifier,
        quality_score(
            &snake,
            prior,
            phrase.norm_weight,
            phrase.is_anchor,
            true,
            false,
        ),
    );
    let screaming = snake.to_ascii_uppercase();
    add_variant(
        &mut variants,
        &screaming,
        VariantKind::ScreamingSnake,
        BoundaryMode::Identifier,
        quality_score(
            &screaming,
            prior,
            phrase.norm_weight,
            phrase.is_anchor,
            true,
            false,
        ),
    );
    let kebab = tokens.join("-");
    add_variant(
        &mut variants,
        &kebab,
        VariantKind::Kebab,
        BoundaryMode::Identifier,
        quality_score(
            &kebab,
            prior,
            phrase.norm_weight,
            phrase.is_anchor,
            true,
            false,
        ),
    );
    let camel = to_camel_case(&tokens);
    add_variant(
        &mut variants,
        &camel,
        VariantKind::Camel,
        BoundaryMode::Identifier,
        quality_score(
            &camel,
            prior,
            phrase.norm_weight,
            phrase.is_anchor,
            true,
            false,
        ),
    );
    let pascal = to_pascal_case(&tokens);
    add_variant(
        &mut variants,
        &pascal,
        VariantKind::Pascal,
        BoundaryMode::Identifier,
        quality_score(
            &pascal,
            prior,
            phrase.norm_weight,
            phrase.is_anchor,
            true,
            false,
        ),
    );
    variants
}

fn add_variant(
    variants: &mut Vec<SurfaceVariant>,
    text: &str,
    kind: VariantKind,
    boundary: BoundaryMode,
    quality: f32,
) {
    if quality < 0.55 || reject_surface_text(text) {
        return;
    }
    if variants.iter().any(|variant| variant.text == text) {
        return;
    }
    variants.push(SurfaceVariant {
        text: text.to_string(),
        bytes: text.as_bytes().to_vec(),
        kind,
        boundary,
        quality,
    });
}

fn strip_vocab_piece(piece: &str) -> String {
    piece
        .trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '#' && c != '_' && c != '-')
        .to_string()
}

fn reject_term_fragment(
    vocab_piece: &str,
    normalized: &str,
    seed_terms: &std::collections::HashSet<String>,
    alias_terms: &std::collections::HashSet<String>,
) -> bool {
    if !vocab_piece.starts_with("##") {
        return false;
    }
    let recovered = normalized.strip_prefix('#').unwrap_or(normalized);
    let is_seed_recovery = seed_terms.iter().any(|seed| seed.ends_with(recovered));
    let is_alias_recovery = alias_terms.contains(recovered);
    vocab_piece.starts_with("##") && !is_seed_recovery && !is_alias_recovery && recovered.len() < 4
}

fn reject_surface_text(text: &str) -> bool {
    if text.is_empty() {
        return true;
    }
    let stripped = text.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '_' && c != '-');
    stripped.is_empty()
        || (stripped.len() < 3 && !text.chars().all(|c| c.is_ascii_uppercase()))
        || stripped.chars().all(|c| c.is_ascii_digit())
        || stripped.chars().all(|c| !c.is_ascii_alphanumeric())
        || (is_stopword(stripped) && !text.chars().all(|c| c.is_ascii_uppercase()))
}

fn normalize_text(text: &str) -> String {
    text.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '_' && c != '-' && c != ' ')
        .to_ascii_lowercase()
}

fn quality_score(
    text: &str,
    prior_df_frac: f32,
    norm_weight: f32,
    is_anchor: bool,
    is_identifier_variant: bool,
    is_fragment: bool,
) -> f32 {
    let len = text.len();
    let is_full_lexeme = !is_fragment && !text.starts_with("##");
    let len_score = ((len as f32 - 3.0) / 8.0).clamp(0.0, 1.0);
    let common_penalty = (prior_df_frac / 0.10).clamp(0.0, 1.0);
    0.30 * (is_full_lexeme as u8 as f32)
        + 0.20 * len_score
        + 0.20 * norm_weight
        + 0.15 * (is_anchor as u8 as f32)
        + 0.10 * (is_identifier_variant as u8 as f32)
        - 0.35 * (is_fragment as u8 as f32)
        - 0.20 * common_penalty
}

fn to_title_case(text: &str) -> String {
    let mut chars = text.chars();
    match chars.next() {
        Some(first) => format!("{}{}", first.to_ascii_uppercase(), chars.as_str()),
        None => String::new(),
    }
}

fn to_camel_case(tokens: &[String]) -> String {
    let mut out = String::new();
    for (index, token) in tokens.iter().enumerate() {
        if index == 0 {
            out.push_str(token);
        } else {
            out.push_str(&to_title_case(token));
        }
    }
    out
}

fn to_pascal_case(tokens: &[String]) -> String {
    tokens
        .iter()
        .map(|token| to_title_case(token))
        .collect::<Vec<_>>()
        .join("")
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

pub fn realize_code_subtoken(term: &str) -> Vec<String> {
    let mut out = vec![term.to_string()];
    let lower = term.to_ascii_lowercase();
    out.push(lower.clone());
    if let Some(first) = term.chars().next() {
        if first.is_ascii_lowercase() {
            let cap: String = term
                .chars()
                .enumerate()
                .map(|(i, c)| if i == 0 { c.to_ascii_uppercase() } else { c })
                .collect();
            out.push(cap);
        }
    }
    out.push(term.to_ascii_uppercase());
    out.sort();
    out.dedup();
    out.retain(|variant| variant.len() >= 2);
    out
}

pub(crate) fn split_identifier_subtokens(text: &str) -> Vec<String> {
    let mut pieces = Vec::new();
    let mut current = String::new();
    let mut prev_is_lower = false;
    for ch in text.chars() {
        if matches!(ch, '_' | '-' | '.' | ':' | '/' | '\\') {
            if !current.is_empty() {
                pieces.push(current.to_ascii_lowercase());
                current.clear();
            }
            prev_is_lower = false;
            continue;
        }
        if ch.is_ascii_uppercase() && prev_is_lower && !current.is_empty() {
            pieces.push(current.to_ascii_lowercase());
            current.clear();
        }
        current.push(ch);
        prev_is_lower = ch.is_ascii_lowercase();
    }
    if !current.is_empty() {
        pieces.push(current.to_ascii_lowercase());
    }
    pieces
}
