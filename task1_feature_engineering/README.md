# Task 1: manual features vs automatic features (OpenFE) and a graph model, for a CatBoost reranker (Yambda)

Run: `uv run python openfe.py` (~1.5 min on a MacBook M2) → `out/openfe_report.json`; then `uv run python graph.py` (~6 min, CPU) → `out/graph_report.json`.

## Setup

- **Data / split:** Yambda-50M, 2K-user sample, the fixed global time split shared with the RL study (`../rl_experiments/common.py`): `train | 1d gap | val 7d | 1d gap | test 7d`. All features and candidates are computed from events before the train cutoff; labels come from the validation week.
- **Candidates:** per user, top-50 ALS (tuned config: 128 factors, reg 0.01, half-life 30 d) ∪ top-50 DecayPop (half-life 1 d), unseen items only → 152K (user, item) pairs, 3,100 positives.
- **Label:** the item is a *new* listen+ in the target week (also evaluated on *organic new*: found by the user, `is_organic = 1`).
- **Users split 50 / 25 / 25** into tr (fit), va (feature screening), te (final check), so no user is in two parts.
- **Manual base features (15):** ALS score and rank, 1-day popularity and rank, item count / skip rate / organic share / 30-day count, user events / organic share / skip rate / 30-day activity, artist popularity, user-artist count and listen+ rate.

## OpenFE

1. **Generation (480 candidates, 477 non-constant):** unary `log`, within-user rank `urank`, deviation from the user's mean `udiff`, artist mean `amean`; binary `+ − × ÷` over all pairs of base features.
2. **Screening:** `feature_gain` = logloss drop of a 50-tree, depth-3 CatBoost trained on ONE candidate with the base model's raw scores as `baseline`. Successive halving: 25% → 50% → 100% of the tr/va users, keep the better half by rank, require gain > 0 only in the last round.
3. **Stage 2:** full CatBoost on base + survivors, keep features with positive importance.

Two details that decide whether this works:
- The base model's scores on tr must be **out-of-fold** (3 folds by user). In-sample scores are overconfident and leave no residual signal.
- The va baseline must come from the **same fold models** (mean of their raw scores). With a full-model baseline on va (more confident than the OOF one on tr), every single-feature booster learned a calibration shift that was wrong on va, and all 477 features looked harmful.

## Results: OpenFE (held-out te users; NDCG over candidates, ideal DCG over all of the user's targets)

| Ranker | New @10 (361 users) | New @50 | Organic new @10 (266 users) | Organic new @50 |
|---|---|---|---|---|
| Popularity only | 0.0162 | 0.0232 | 0.0166 | 0.0229 |
| ALS only | 0.0525 | 0.0570 | 0.0249 | 0.0286 |
| CatBoost, 15 manual features | 0.0798 | 0.0727 | 0.0374 | 0.0402 |
| CatBoost, manual + 2 OpenFE features | **0.0826** | **0.0747** | **0.0392** | **0.0414** |
| Δ OpenFE vs manual, 95% paired bootstrap CI | +0.0028 [−0.0017, +0.0073] | +0.0020 [−0.0005, +0.0048] | +0.0018 [−0.0012, +0.0049] | +0.0012 [−0.0010, +0.0038] |

Selected: `urank(item_cnt)` (item popularity relative to the user's other candidates) and `amean(item_cnt)` (mean popularity of the artist's candidates).

**Reading:** the reranker over manual features is the big step (+52% over ALS alone at @10). OpenFE finds two sensible relative-popularity features and a small gain at both cutoffs that is not significant on 361 test users; single-feature gains are ~1e-4 nats, so the manual features already carry most of the signal.

## Research question 2: can a graph model learn what aggregate features capture?

**Setup** (`graph.py`): a heterogeneous LightGCN over users, items and artists (~200K nodes, d = 64, 2 layers, BPR, 150 steps on CPU).
- **Event types as relations**, each with a learned propagation weight: user-item listen+ (likes count double), user-item skip, user-item *organic* listen+, item-artist.
- **Time**: edge weights decay with age (half-life 30 d), and training is temporal: the graph is built from events before T0 = cutoff − 14 d and the model is trained to predict the *new* items users picked up in [T0, cutoff), the same task as the reranker's label. At inference the same embeddings are propagated over the full graph at the cutoff. This is the "temporal neighbour" rule (only edges before the prediction time) of Utushkin, Ovsiannikov, D'yakonov, *Scaling GNNs for Friend Recommendation: Multi-Hash User Embeddings and Temporal Neighbor Sampling*, CIKM 2026 ([arXiv 2608.27413](https://arxiv.org/abs/2608.27413)).
- **Ablations**: plain LightGCN (one binary user-item relation, no decay, no event types); **multi-hash item embeddings** from the same paper (two hashes into 2^15-row tables, summed) instead of a full item table.
- Same candidates, users and split as OpenFE.

**Ranking** (held-out users; CatBoost refit inside `graph.py`, so compare rows within this table):

| Ranker | New @10 | New @50 | Organic new @10 | Organic new @50 |
|---|---|---|---|---|
| ALS only | 0.0525 | 0.0570 | 0.0249 | 0.0286 |
| Graph, plain LightGCN | 0.0675 | 0.0667 | 0.0384 | 0.0396 |
| Graph, typed + temporal | 0.0659 | 0.0666 | 0.0357 | 0.0404 |
| Graph, typed + temporal, multi-hash items (−45% parameters) | 0.0634 | 0.0637 | 0.0319 | 0.0385 |
| CatBoost, manual features | 0.0755 | 0.0703 | 0.0376 | 0.0404 |
| CatBoost, manual + graph score | 0.0848 | 0.0755 | 0.0413 | 0.0432 |
| CatBoost, manual + OpenFE | 0.0800 | 0.0721 | 0.0345 | 0.0383 |
| **CatBoost, manual + OpenFE + graph score** | **0.0873** | **0.0765** | **0.0440** | **0.0453** |

| Contrast (paired 95% CI) | New @10 | New @50 | Organic new @10 | Organic new @50 |
|---|---|---|---|---|
| + graph score vs manual | **+0.0092 [+0.0005, +0.0175]** | +0.0051 [−0.0009, +0.0104] | +0.0037 [−0.0068, +0.0116] | +0.0029 [−0.0054, +0.0091] |
| + graph vs manual + OpenFE | +0.0074 [−0.0003, +0.0159] | +0.0043 [−0.0004, +0.0100] | **+0.0095 [+0.0018, +0.0196]** | **+0.0071 [+0.0018, +0.0149]** |
| typed + temporal vs plain graph | −0.0016 [−0.0107, +0.0072] | −0.0001 [−0.0043, +0.0040] | −0.0027 [−0.0114, +0.0046] | +0.0008 [−0.0029, +0.0043] |

**Does the embedding learn the manual aggregates?** Linear probe (ridge) from the (user, item, user × item) embeddings of the typed + temporal model to each manual feature, R² on held-out users:

| Item-level aggregate | R² | User-level aggregate | R² |
|---|---|---|---|
| item count (log) | **0.80** | user event count (log) | 0.02 |
| item organic share | **0.74** | user skip rate | 0.07 |
| artist popularity (log) | 0.54 | user organic share | −0.03 |
| item skip rate | 0.44 | user-artist count (log) | 0.27 |

Learned relation weights (typed + temporal): artist 1.77, listen+ 1.15, skip 1.13, organic 0.76.

**Answers.**
1. **Yes for item-level aggregates, no for user-level ones.** The embeddings encode item popularity, organic share and skip rate almost linearly (R² 0.44-0.80), but not user activity, skip or organic rates (R² ≈ 0): BPR ranks items *within* a user, so a user's overall level never needs to be represented. That is exactly what the user-level manual features add.
2. **Alone, the graph model beats ALS** (+25% at @10) and on organic new @50 matches CatBoost with manual features (0.0404 vs 0.0404), but it does not beat the boosted reranker overall.
3. **Together they are best.** The graph score as one extra feature improves CatBoost significantly (new @10 +0.009; organic new +0.010 at @10 and +0.007 at @50 on top of OpenFE features): the graph adds multi-hop collaborative signal that hand-made aggregates do not.
4. **Event types and time did not help at this budget**: typed + temporal ≈ plain LightGCN (CIs include 0). The learned weights do use the types (artist edges up-weighted, organic down-weighted), but 150 steps on 2K users are not enough for them to pay off; a GNN that conditions on edge time (temporal neighbour sampling per prediction time, as in the paper) is the next step.
5. **Multi-hash item embeddings** cut parameters by 45% (10.6M → 5.9M; the item table by 95%) for a ~4% relative loss at @10, the scaling trade-off the paper reports for billion-edge graphs.

## Limitations / next steps

- 2K users and one week of labels: the test CIs are wide. One seed. The graph model trains only 150 steps. Next: 10K users and several weekly snapshots as training groups.
- Logloss gain is a pointwise proxy; a ranking objective (YetiRank) for the final model is the natural next step.
- The operator set is small (no time-window aggregates or artist/album group-bys beyond the mean); OpenFE's original order-2 search would add more.
