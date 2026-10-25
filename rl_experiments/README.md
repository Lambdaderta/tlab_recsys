# RL vs standard recommenders on Yambda-50M: organic discovery and push-based RL

All numbers: 10K users (all of Yambda-50M), test week, GPU server. Configs were chosen on the validation week; test was run once. Every choice and why: `DECISIONS.md`. The previous version of this study (unequal information between models, listen-based reward) is kept as `report_v2.md`.

## 1. Results

### Main table: 10K users, test week

Users with targets: 5210 (organic new), 7075 (new), 7731 (all). Δ = paired bootstrap 95% CI vs `sasrec_s0`, * = excludes 0.

| Group | Model | NDCG@10 organic new | Δ organic new | NDCG@10 new | Δ new | NDCG@100 new | Recall@100 new | Cov@10 new | NDCG@10 all |
|---|---|---|---|---|---|---|---|---|---|
| Heuristics | DecayPop (tuned half-life) | 0.0287 | +0.0105 [+0.0071, +0.0139]* | 0.0343 | -0.0104 [-0.0136, -0.0073]* | 0.0388 | 0.060 | 0.0002 | 0.0573 |
| Heuristics | Recently played (meant for "all"; on new items it falls back to popularity) | 0.0198 | +0.0016 [-0.0014, +0.0048] | 0.0234 | -0.0213 [-0.0243, -0.0185]* | 0.0320 | 0.053 | 0.0003 | 0.3737 |
| Co-occurrence / CF | ItemKNN (tuned) | 0.0085 | -0.0097 [-0.0119, -0.0074]* | 0.0281 | -0.0166 [-0.0192, -0.0139]* | 0.0288 | 0.038 | 0.1122 | 0.1278 |
| Co-occurrence / CF | Co-visitation (tuned) | 0.0146 | -0.0036 [-0.0059, -0.0013]* | 0.0197 | -0.0250 [-0.0275, -0.0226]* | 0.0309 | 0.053 | 0.1428 | 0.1884 |
| Co-occurrence / CF | ALS (tuned) | 0.0297 | +0.0115 [+0.0088, +0.0143]* | 0.0689 | +0.0242 [+0.0210, +0.0272]* | 0.0853 | 0.129 | 0.0255 | 0.1990 |
| GPT-2 | GPT-2 SASRec, seed 0 | 0.0182 | — | 0.0447 | — | 0.0589 | 0.092 | 0.0661 | 0.1914 |
| Push-based RL | β̂: imitate pushes (logging policy) | 0.0052 | -0.0130 [-0.0152, -0.0106]* | 0.0152 | -0.0295 [-0.0319, -0.0272]* | 0.0217 | 0.035 | 0.0059 | 0.0548 |
| Push-based RL | AWR on pushes + KL | 0.0199 | +0.0017 [+0.0007, +0.0027]* | 0.0505 | +0.0058 [+0.0045, +0.0071]* | 0.0647 | 0.100 | 0.0578 | 0.1903 |
| Push-based RL | Off-policy REINFORCE (π/β̂, top-K) + KL | 0.0141 | -0.0041 [-0.0053, -0.0030]* | 0.0259 | -0.0188 [-0.0204, -0.0171]* | 0.0373 | 0.061 | 0.0574 | 0.1597 |
| Push-based RL | Exact tilted π_ref·exp(r̂/1) | 0.0185 | +0.0003 [+0.0000, +0.0006]* | 0.0450 | +0.0003 [-0.0000, +0.0007] | 0.0589 | 0.092 | 0.0656 | 0.1916 |
| Push-based RL | Exact tilted π_ref·exp(r̂/0.3) | 0.0184 | +0.0002 [-0.0003, +0.0007] | 0.0445 | -0.0002 [-0.0009, +0.0004] | 0.0581 | 0.090 | 0.0639 | 0.1913 |

### Off-policy value on held-out test-week pushes (SNIPS, weights clipped at 10)

| Policy | SNIPS reward / push | Logged mean | Effective sample share |
|---|---|---|---|
| π_ref (GPT-2 SASRec) | 0.0714 | 0.0661 | 0.61 |
| β̂ (imitate pushes) | 0.0661 | 0.0661 | 1.00 |
| Exact tilted, b = 0.3 | 0.0725 | 0.0661 | 0.61 |
| Exact tilted, b = 1 | 0.0717 | 0.0661 | 0.61 |
| AWR on pushes + KL | 0.0705 | 0.0661 | 0.65 |
| Off-policy REINFORCE + KL | 0.0866 | 0.0661 | 0.47 |

The reward is 1[like] + 1[organic listen of another track of the artist 1-3 days later] − 1[dislike]; the logged policy (β̂ itself) scores the logged mean by construction. Effective sample share = (Σw)² / (n Σw²).

### Reward model on held-out test-week pushes (`rm_push_quality.py`, 50K pushes, 6.4% with reward > 0)

| Predictor | AUC (reward > 0) | Within-user AUC | Spearman | MAE |
|---|---|---|---|---|
| Push reward model r̂ (GPT-2 + bilinear head, 3K steps) | 0.585 | 0.578 | 0.030 | 0.095 |
| Item mean of training pushes | **0.700** | **0.699** | 0.072 | 0.116 |
| User mean of training pushes | 0.743 | 0.496 | 0.085 | 0.118 |

The learned reward model is **worse than the item mean**: with a sparse reward (6% positive) and a short training budget it did not learn item-level signal. The user mean has the best overall AUC but cannot rank items for a user (within-user AUC 0.50). This is why the tilted policy barely moves (finding 6) and why the RL gain comes from AWR, which uses the *logged* rewards directly.

### Tuning of the base algorithms (10K val, selection on organic-new NDCG@10)
| Model | Config | organic new | new |
|---|---|---|---|
| decaypop | `{'hl': 1}` | 0.0285 | 0.0272 |
| decaypop | `{'hl': 7}` | 0.0212 | 0.0286 |
| decaypop | `{'hl': 30}` | 0.0168 | 0.0223 |
| decaypop | `{'hl': 0.5}` | 0.0254 | 0.0234 |
| decaypop | `{'hl': 2}` **(chosen)** | 0.0288 | 0.0300 |
| itemknn | `{'hl': 14}` **(chosen)** | 0.0091 | 0.0281 |
| itemknn | `{'hl': 60}` | 0.0076 | 0.0246 |
| itemknn | `{'hl': 3}` | 0.0063 | 0.0194 |
| itemknn | `{'hl': 7}` | 0.0082 | 0.0257 |
| covisit | `{'window': 3, 'last_n': 20, 'alpha': 0.5}` | 0.0091 | 0.0111 |
| covisit | `{'window': 10, 'last_n': 100, 'alpha': 0.5}` | 0.0114 | 0.0136 |
| covisit | `{'window': 10, 'last_n': 100, 'alpha': 0.3}` **(chosen)** | 0.0151 | 0.0207 |
| covisit | `{'window': 20, 'last_n': 200, 'alpha': 0.5}` | 0.0119 | 0.0143 |
| als | `{'factors': 128, 'reg': 0.01, 'hl': 30}` | 0.0237 | 0.0574 |
| als | `{'factors': 256, 'reg': 0.1, 'hl': 120}` | 0.0193 | 0.0457 |
| als | `{'factors': 256, 'reg': 0.01, 'hl': 60}` | 0.0213 | 0.0513 |
| als | `{'factors': 64, 'reg': 0.01, 'hl': 30}` | 0.0233 | 0.0548 |
| als | `{'factors': 192, 'reg': 0.01, 'hl': 30}` | 0.0247 | 0.0593 |
| als | `{'factors': 128, 'reg': 0.001, 'hl': 14}` **(chosen)** | 0.0287 | 0.0689 |

### GPT-2 SASRec: step count chosen on 10K val
| Steps | organic new | new |
|---|---|---|
| 3000 | 0.0169 | 0.0383 |
| 6000 | 0.0193 | 0.0446 |
| 9000 | 0.0200 | 0.0474 |
| 12000 | 0.0200 | 0.0472 |

## 2. Problem statement

**Task.** For each user at a fixed moment, recommend 10 tracks for the next 7 days. Data: Yambda-50M (10K users, 46.5M listens over 301 days) with played ratio, likes, dislikes and the `is_organic` flag: 0 = the recommender pushed the track (radio / recommended playlists), 1 = the user chose it (search, own library, artist page).

**Targets.** Two co-primary target sets, both on items the user never had before the test week (seen items are filtered from the lists):
- **organic new**: new items the user went and found themselves. This is user intent, least shaped by the old recommender. Model selection uses this target.
- **new**: any new item played to listen+ (≥ 50%).

"All" targets (including re-listens) are reported as a secondary column; there "recently played" wins by construction.

**Fair comparison.** The previous study gave models different information (ALS saw the full history, SASRec only the last 200 events; EASE only the 20K most popular items; nothing was tuned). Now:
- every model sees the **full fit history**: the GPT-2 sequence model gets a **prefix token** summarising up to 256 events sampled from the history before its 200-event window (point-in-time, no leakage);
- one shared item vocabulary (≥ 5 listens); EASE's restriction to its top 35K items is reported as coverage instead of hidden;
- every base algorithm gets a small tuning grid on the validation week, selected on organic-new NDCG@10;
- the sequence model is **GPT-2** (Hugging Face `GPT2Model`, as the task requires), with item ids as tokens.

**Why RL, and what changed.** The data analysis (section 4) showed that "listen+" on a pushed track is almost no evidence of interest: pushes play in the background. So the old reward (listen+ = +1) mostly rewarded imitating background play. The new RL setup uses what Yambda uniquely has:
- **actions** = items the recommender pushed (`is_organic = 0`), the only events where the system chose the action;
- **reward** = 1[like] + 1[the user later goes to *another* track of the same artist on their own, 1-3 days after the push] − 1[dislike];
- **logging policy** β̂ = a GPT-2 trained to predict pushed items, which gives the propensities the logs lack, so policies can be evaluated off-policy (SNIPS) on held-out test-week pushes.

**Constraint: offline only.** No online feedback; improvements must stay close to the logged policy and be checked by more than one offline signal (NDCG on organic and all new targets, SNIPS reward on held-out pushes).

**Protocol.** Global time split `train | 1d gap | val 7d | 1d gap | test 7d`; val for choices only; for test, every model is refit on everything before the second gap. NDCG@10/100, Recall@100, coverage; paired bootstrap CIs over users.

## 3. System schema

```mermaid
flowchart LR
  subgraph Data
    L[Yambda listens<br/>played_ratio, is_organic] --> E[Events + feedback class<br/>skip / listen+ / like / dislike]
    LK[likes / dislikes] --> E
    E --> S[Time split<br/>train - gap - val - gap - test]
    E --> R[Push reward<br/>like + later organic return<br/>to another track of the artist]
  end
  subgraph Retrieval baselines, tuned on val
    S --> CF[DecayPop, ItemKNN, Co-visit,<br/>EASE 35K, ALS]
  end
  subgraph GPT-2 sequence model
    S --> G[GPT2Model<br/>tokens = item + feedback + organic<br/>+ prefix = long-term history]
    G --> REF[pi_ref: next listen+ item]
    G --> BETA[beta-hat: next pushed item<br/>= logging policy]
    G --> RM[r-hat: push reward model]
  end
  subgraph Push-based RL
    REF --> AWR[AWR on pushes + KL]
    REF --> OPC[Off-policy REINFORCE<br/>pi / beta-hat, top-K + KL]
    BETA --> OPC
    REF --> TILT[Exact tilted policy<br/>pi_ref x exp r-hat / b]
    RM --> TILT
  end
  subgraph Evaluation
    CF --> EV[NDCG organic new / new / all<br/>Recall@100, coverage]
    REF --> EV
    AWR --> EV
    OPC --> EV
    TILT --> EV
    AWR --> OPE[SNIPS reward on held-out<br/>test-week pushes, weights pi / beta-hat]
    OPC --> OPE
    TILT --> OPE
    BETA --> OPE
  end
```

Production view: retrieval models (ALS, DecayPop, GPT-2) produce candidates; a policy trained on push outcomes decides what to push; β̂ and the SNIPS estimate are the offline gate before an A/B test.

## 4. What the data says about organic listening (10K users, all 46.5M listens)

| | Share | Skip | Listen+ | Like | Repeat share | Median popularity of first listens |
|---|---|---|---|---|---|---|
| Pushed | 48% | 31% | 68% | 0.95% | 62% | 755 |
| Organic | 52% | 41% | 59% | 0.47% | 81% | 377 |

- Pushes are skipped **less** than organic listens: pushes are mostly background radio, so listen+ measures the listening mode more than preference.
- 65% of first-ever item listens come from pushes: the recommender drives discovery. When users discover on their own, they find less popular items.
- Users differ a lot: the per-user organic share goes from 9% (10th percentile) to 97% (90th).

**Does a push create real interest?** First contact with an artist via a push, then an organic listen of **another** track of that artist 1-7 days later:

| Reaction to the push | Same track replayed | Another track of the artist |
|---|---|---|
| skip | 1.6% | 0.97% |
| listen+ | 2.3% | 1.04% |
| like | 50% | 6.9% |
| dislike | 0.7% | 0.72% |

Listen+ on a push is no better than a skip at predicting real artist discovery; a like is ~7×. Most of "like → return" is the liked track replayed from the collection, so the reward counts only *other* tracks. The signal also depends on the user type (radio-heavy users almost never return on their own), which is why every comparison is within the same users.

## 5. Main idea and what we found

**1. Tuned matrix factorisation is the model to beat.** ALS (reg 0.001, 14-day half-life, chosen on val) is best on both co-primary targets: NDCG@10 0.0689 on new items (+0.024 [+0.021, +0.027] vs GPT-2 SASRec) and 0.0297 on organic new (+0.012). Tuning mattered most for ALS: on val organic new its configs ranged from 0.0193 to 0.0287 (the first grid point, 128 factors / reg 0.01 / 30 d, scored 0.0237).

**2. What users find on their own is driven by what is trending.** DecayPop with a 2-day half-life nearly ties ALS on organic new (0.0287 vs 0.0297) and beats GPT-2 SASRec there by +0.0105 [+0.0071, +0.0139]; on all new items it is far behind (0.0343). Organic discoveries concentrate on what is new and popular this week; a sequence model of the user's own history, with no notion of global time, misses that. Item freshness / trend features are the obvious next input for the sequence model.

**3. GPT-2 SASRec with full history is a solid but not leading retriever** (0.0447 new, 0.0182 organic new; the step count plateaued at 9K steps on val). Giving it the full history (prefix token) and the GPT-2 architecture did not close the gap to tuned ALS on this data size.

**4. Push-based RL: reweighting pushes by real engagement helps relevance.** AWR on pushes (reward = like + later organic artist return), kept close to π_ref by KL, improves the GPT-2 on both targets: +0.0058 [+0.0045, +0.0071] on new and +0.0017 [+0.0007, +0.0027] on organic new. The policy learns which pushed items people actually keep, which transfers to predicting what they will discover. Caveat: one training seed; in earlier runs SASRec moved by up to 0.005 between seeds, so the new-item gain is at the edge of seed noise.

**5. Optimising the off-policy estimate alone fails, visibly.** Off-policy REINFORCE with the π/β̂ correction raises the SNIPS reward by 21% (0.0866 vs 0.0714 for π_ref) while NDCG collapses (new 0.0259, −42%) and the effective sample share drops to 0.47. The policy moved far from the data (KL 0.75) and exploits errors in β̂ and the importance weights. This is the central lesson for offline RL in recsys: an off-policy value estimate is a gate, not a target, and must be read together with relevance on held-out users.

**6. The closed-form optimum shows how much signal the reward model has.** The exact tilted policy π_ref·exp(r̂/b) barely moves anything (SNIPS 0.0725 at b = 0.3, NDCG unchanged). The held-out check explains it: the push reward model ranks pushes worse than the item mean (AUC 0.585 vs 0.700), because the reward is sparse (0.066 per push) and the model was trained briefly. That is why the gains come from AWR's use of the *logged* rewards rather than from optimising a learned reward model, the opposite of the RLHF-style setup.

**7. Imitating the recommender is the worst policy for discovery.** β̂ (a GPT-2 trained to predict what the system pushes) is far below everything else (0.0152 new, 0.0052 organic new): what the old system pushes is not what users go on to find or keep.

**Production view.** ALS + short-window popularity as candidate generators; a GPT-2 sequence model fine-tuned with AWR on push outcomes as the ranker for what to push; β̂ + SNIPS + organic-new NDCG as the offline gate; the decision itself only through an A/B test.

**Not run (time budget):** a second GPT-2 seed, EASE (dropped: a dense inverse over the 338K-item vocabulary does not fit, and the top-35K restriction crashed the server once), a two-tower model, tuning of the RL policies (one KL weight, 300 steps).

## 6. Limitations

- **Small tuning grids** (2-6 configs per base model) because of the time budget; EASE dropped.
- **One training seed** for GPT-2 and the RL policies in this run. Two earlier full runs showed SASRec moving by up to 0.005 NDCG@10 between seeds, so differences below that are not conclusive even when the user-level CI excludes 0.
- **RL policies are untuned** (one KL weight, 300 steps), and the push reward model is weak (below the item-mean baseline on held-out pushes); a better reward model is the first thing to improve before RL against it.
- **SNIPS depends on β̂.** If the logging-policy model is miscalibrated, the importance weights are biased; weights are clipped at 10 and the effective sample share is reported. The reward also uses a 3-day return window so test-week pushes can be scored inside the week.
- **The return reward is sparse** (a few % of pushes) and slightly undercounted (only the first organic listen of the artist after the push is checked).
- **Organic is a proxy for intent,** not ground truth: users' own library playback also counts as organic.
