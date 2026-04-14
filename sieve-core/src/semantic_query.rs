pub const MAX_SEED_GROUPS: usize = 6;
pub const MAX_TERMS: usize = 96;
pub const MAX_ANCHORS: usize = 12;
pub const MAX_PHRASES: usize = 8;
pub const MAX_VARIANTS_PER_TERM: usize = 6;
pub const ANCHOR_WEIGHT_RATIO: f32 = 0.55;
pub const MIN_TERM_WEIGHT_RATIO: f32 = 0.18;
pub const MIN_AUX_GROUP_RATIO: f32 = 0.35;
pub const SEED_CLAIM_RATIO: f32 = 0.20;

pub type TermId = u16;
pub type GroupId = u16;
pub type PhraseId = u16;

#[derive(Debug, Clone, PartialEq)]
pub struct SemanticQuery {
    pub raw_query: String,
    pub normalized_query: String,
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
) -> crate::Result<SemanticQuery> {
    compile_semantic_query_with(
        raw_query,
        &|text| sparse.encode(text),
        &|vocab_id| sparse.vocab_piece(vocab_id).map(str::to_string),
        aliases,
    )
}

fn compile_semantic_query_with<E, V>(
    raw_query: &str,
    encode: &E,
    vocab_piece: &V,
    aliases: &crate::aliases::AliasLexicon,
) -> crate::Result<SemanticQuery>
where
    E: Fn(&str) -> crate::Result<Vec<(u32, f32)>>,
    V: Fn(u32) -> Option<String>,
{
    let normalized_query = normalize_query(raw_query);
    let seed_texts = tokenize_content_tokens(&normalized_query);
    if seed_texts.is_empty() {
        return Ok(SemanticQuery {
            raw_query: raw_query.to_string(),
            normalized_query,
            seeds: Vec::new(),
            groups: Vec::new(),
            terms: Vec::new(),
            phrases: Vec::new(),
            query_order: Vec::new(),
            total_group_importance: 0.0,
        });
    }

    let full_weights = encode(&normalized_query)?;
    let per_seed_weights: Vec<Vec<(u32, f32)>> = seed_texts
        .iter()
        .map(|seed| encode(seed))
        .collect::<crate::Result<_>>()?;
    let full_max_weight = full_weights
        .iter()
        .map(|(_, weight)| *weight)
        .fold(0.0f32, f32::max);
    if full_max_weight <= 0.0 {
        return Ok(seed_only_query(raw_query, normalized_query, &seed_texts));
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
        let seed_best = per_seed_weights
            .get(seed_index)
            .and_then(|weights| best_matching_seed_piece(weights, seed, vocab_piece))
            .or_else(|| {
                per_seed_weights
                    .get(seed_index)
                    .and_then(|weights| weights.first().cloned())
                    .and_then(|(id, weight)| vocab_piece(id).map(|piece| (id, weight, piece)))
            });
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
    retained.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    retained.truncate(MAX_TERMS);

    let mut anchor_count = terms.iter().filter(|term| term.is_anchor).count();

    for (vocab_id, raw_weight, piece) in retained {
        let canonical = normalize_piece(&piece);
        if canonical.is_empty() {
            continue;
        }
        let norm_weight = (raw_weight / full_max_weight).clamp(0.0, 1.0);
        let maybe_claim = claim_group_for_term(
            vocab_id,
            &canonical,
            norm_weight,
            &seed_texts,
            &per_seed_weights,
            vocab_piece,
            aliases,
        );
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
    seed_texts: &[String],
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
        seeds,
        groups,
        terms,
        phrases,
        query_order,
        total_group_importance,
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

fn claim_group_for_term<V>(
    vocab_id: u32,
    canonical: &str,
    norm_weight: f32,
    seed_texts: &[String],
    per_seed_weights: &[Vec<(u32, f32)>],
    vocab_piece: &V,
    aliases: &crate::aliases::AliasLexicon,
) -> Option<(GroupId, TermSource)>
where
    V: Fn(u32) -> Option<String>,
{
    if let Some(index) = seed_texts.iter().position(|seed| seed == canonical) {
        return Some((index as GroupId, TermSource::OriginalToken));
    }
    let mut best_claim = None;
    let mut best_weight = 0.0f32;
    for (seed_index, weights) in per_seed_weights.iter().enumerate() {
        let seed_max = weights
            .iter()
            .map(|(_, weight)| *weight)
            .fold(0.0, f32::max);
        if seed_max <= 0.0 {
            continue;
        }
        let Some(seed_weight) = weights
            .iter()
            .find(|(candidate_id, _)| *candidate_id == vocab_id)
            .map(|(_, weight)| *weight)
        else {
            continue;
        };
        if seed_weight >= SEED_CLAIM_RATIO * seed_max && seed_weight > best_weight {
            best_weight = seed_weight;
            best_claim = Some((seed_index as GroupId, TermSource::SparseExpansion));
        }
    }
    if best_claim.is_some() {
        return best_claim;
    }
    for (seed_index, seed) in seed_texts.iter().enumerate() {
        if aliases.same_alias_family(seed, canonical) || aliases.same_alias_family(canonical, seed)
        {
            return Some((seed_index as GroupId, TermSource::AliasExpansion));
        }
    }
    if norm_weight >= MIN_AUX_GROUP_RATIO {
        let _ = vocab_piece(vocab_id)?;
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

fn tokenize_content_tokens(query: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for ch in query.chars() {
        if ch.is_alphanumeric() {
            current.push(ch);
        } else if !current.is_empty() {
            maybe_push_token(&mut tokens, &mut current);
        }
    }
    if !current.is_empty() {
        maybe_push_token(&mut tokens, &mut current);
    }
    tokens.truncate(MAX_SEED_GROUPS);
    tokens
}

fn maybe_push_token(tokens: &mut Vec<String>, current: &mut String) {
    let token = current.to_ascii_lowercase();
    current.clear();
    if token.len() >= 3 && !is_stopword(&token) {
        tokens.push(token);
    }
}

fn normalize_query(query: &str) -> String {
    query.trim().to_lowercase()
}

fn normalize_piece(piece: &str) -> String {
    piece
        .trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '#' && c != '_' && c != '-')
        .trim_start_matches("##")
        .to_ascii_lowercase()
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
    use super::{compile_semantic_query_with, TermSource};

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
        )
        .unwrap();
        assert_eq!(query.seeds.len(), 2);
        assert_eq!(query.groups.len(), 2);
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
}
