use crate::semantic_query::TermId;

pub fn compute_idf(
    _term_id: TermId,
    df_scan_count: u32,
    n_scan_entries: u32,
    prior_df_frac: f32,
) -> f32 {
    let alpha = n_scan_entries as f32 / (n_scan_entries as f32 + 32.0);
    let df_frac = alpha * (df_scan_count as f32 / n_scan_entries.max(1) as f32)
        + (1.0 - alpha) * prior_df_frac;
    ((-df_frac.max(1e-6).ln()) / 6.0).clamp(0.25, 1.0)
}

pub fn score_window(
    window: &crate::semantic_scan::WindowAccumulator,
    query: &crate::semantic_query::SemanticQuery,
    idf: &[f32],
) -> f32 {
    if !window.has_anchor {
        return 0.0;
    }

    let mut term_tf: std::collections::HashMap<TermId, u32> = std::collections::HashMap::new();
    let mut term_mass: std::collections::HashMap<TermId, f32> = std::collections::HashMap::new();
    let mut group_mass: std::collections::HashMap<crate::semantic_query::GroupId, f32> =
        std::collections::HashMap::new();
    let mut matched_groups = std::collections::HashSet::new();
    let mut penalty = 0.0f32;
    let mut group_centers: std::collections::HashMap<
        crate::semantic_query::GroupId,
        (f32, f32, bool),
    > = std::collections::HashMap::new();

    for event in &window.events {
        matched_groups.insert(event.primary_group_id);
        let center = (event.byte_start as f32 + event.byte_end as f32) * 0.5;
        group_centers
            .entry(event.primary_group_id)
            .and_modify(|entry| {
                if event.weight > entry.0 {
                    *entry = (event.weight, center, event.is_anchor);
                } else if event.is_anchor {
                    entry.2 = true;
                }
            })
            .or_insert((event.weight, center, event.is_anchor));

        if let Some(term_id) = event.term_id {
            *term_tf.entry(term_id).or_default() += 1;
        } else if event.phrase_id.is_some() {
            *group_mass.entry(event.primary_group_id).or_default() += event.weight;
        }
    }

    for (term_id, tf) in term_tf {
        let term = match query.terms.get(term_id as usize) {
            Some(term) => term,
            None => continue,
        };
        let idf_value = idf.get(term_id as usize).copied().unwrap_or(1.0);
        let tf_sat = 1.0 - (-(tf as f32) / 1.5).exp();
        let mass = term.norm_weight * idf_value * tf_sat;
        term_mass.insert(term_id, mass);
        *group_mass.entry(term.group_id).or_default() += mass;
        penalty += 0.25 * (mass * (1.0 - idf_value));
    }

    let mut base = 0.0f32;
    let mut matched_group_importance = 0.0f32;
    for group in &query.groups {
        if let Some(mass) = group_mass.get(&group.group_id).copied() {
            let group_sat = 1.0 - (-(mass) / 1.2).exp();
            base += group.importance * group_sat;
            matched_group_importance += group.importance;
        }
    }

    let coverage = if query.total_group_importance > 0.0 {
        0.55 * matched_group_importance / query.total_group_importance
    } else {
        0.0
    };

    let proximity = 0.35 * compute_proximity(&group_centers);
    let order = 0.22 * compute_ordered_bonus(query, &group_centers);
    let window_len = (window.window_end.saturating_sub(window.window_start)).max(1) as f32;
    let len_norm = (512.0 / window_len.max(128.0)).powf(0.15);
    (base + coverage + proximity + order - penalty).max(0.0) * len_norm
}

fn compute_proximity(
    group_centers: &std::collections::HashMap<crate::semantic_query::GroupId, (f32, f32, bool)>,
) -> f32 {
    let mut pairs = Vec::new();
    let groups: Vec<_> = group_centers.iter().collect();
    for i in 0..groups.len() {
        for j in (i + 1)..groups.len() {
            let (_, (v_i, c_i, _)) = groups[i];
            let (_, (v_j, c_j, _)) = groups[j];
            let gap = (c_i - c_j).abs();
            let pair_score = (v_i * v_j).sqrt() * (-(gap) / 96.0).exp();
            pairs.push(pair_score);
        }
    }
    pairs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    pairs.into_iter().take(6).sum()
}

fn compute_ordered_bonus(
    query: &crate::semantic_query::SemanticQuery,
    group_centers: &std::collections::HashMap<crate::semantic_query::GroupId, (f32, f32, bool)>,
) -> f32 {
    let mut total = 0.0f32;
    for pair in query.query_order.windows(2) {
        let Some((v_i, c_i, anchor_i)) = group_centers.get(&pair[0]).copied() else {
            continue;
        };
        let Some((v_j, c_j, anchor_j)) = group_centers.get(&pair[1]).copied() else {
            continue;
        };
        if !anchor_i || !anchor_j || c_j < c_i {
            continue;
        }
        let center_gap = c_j - c_i;
        total += (v_i * v_j).sqrt() * (-(center_gap) / 64.0).exp();
    }
    total
}
