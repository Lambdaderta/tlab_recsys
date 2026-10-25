# T-Lab RecSys test case: automatic features and RL on Yambda-50M

Two studies on [Yambda-50M](https://huggingface.co/datasets/yandex/yambda) (Yandex Music; 10K users, 46.5M listens over 301 days, with played ratio, likes, dislikes and an `is_organic` flag):

| | Task | Report | Code |
|---|---|---|---|
| **Task 1** | Manual vs automatic features (OpenFE) for a CatBoost reranker; a user-item-artist graph model as an alternative to engineered aggregates | [`task1_feature_engineering/README.md`](task1_feature_engineering/README.md) | `openfe.py`, `graph.py` |
| **Task 2** | GPT-2 next-item model, reward model, RL fine-tuning, compared with tuned standard recommenders | [`rl_experiments/README.md`](rl_experiments/README.md) | `rl_experiments/*.py` |

Every design choice and the reason for it is logged in [`rl_experiments/DECISIONS.md`](rl_experiments/DECISIONS.md). An earlier version of the RL study (different reward, unequal information between models) is kept as [`rl_experiments/report_v2.md`](rl_experiments/report_v2.md) for reference.

## Problem

For each user at a fixed moment, recommend tracks for the next 7 days. Two co-primary targets, both on items the user never had before (seen items are filtered out):
- **organic new:** new tracks the user went and found on their own (`is_organic = 1`). User intent, least shaped by the old recommender. Model selection uses this target.
- **new:** any new track played to at least 50%.

**Evaluation protocol** (shared by both tasks, `rl_experiments/common.py`): one global time split `train | 1d gap | val 7d | 1d gap | test 7d`. Validation is used for every choice; test is run once, with models refit on everything before the second gap. Metrics: NDCG@10 (also @50/@100), Recall@100, coverage; paired bootstrap 95% CIs over users.

## Approach

**Data analysis first.** On Yambda, pushed tracks (`is_organic = 0`) are skipped *less* than organic ones: pushes are mostly background radio. A listen of a pushed track is no better than a skip at predicting that the user later seeks the artist out (1.04% vs 0.97%), while a like is ~7× stronger. So "listened ≥ 50%" is a weak reward, and organic behaviour is the cleaner signal of interest.

**Task 2 (RL).**
- Fair baselines: DecayPop, ItemKNN, co-visitation and ALS, each tuned on validation with the full user history.
- **GPT-2** (Hugging Face `GPT2Model`), item ids as tokens plus feedback and organic embeddings, with a prefix token summarising the user's full history before its 200-event window.
- **Push-based RL:** actions = items the recommender pushed; reward = like + later organic listen of *another* track of the same artist − dislike; a GPT-2 trained to predict pushes estimates the logging policy, so policies can be evaluated off-policy (SNIPS) on held-out pushes.
- Policies: AWR, off-policy REINFORCE with top-K correction (Chen et al. 2019), and the exact KL-regularised optimum π_ref·exp(r̂/b).

**Task 1 (features).**
- Candidates (ALS ∪ DecayPop), 15 manual features, CatBoost reranker.
- **OpenFE:** 480 generated features screened by single-feature boosting gain over the base model, with successive halving.
- **Graph model:** heterogeneous LightGCN on users, items and artists, with event types as relations (listen, skip, organic, artist) and temporal training (graph from before T, targets after T); plain and multi-hash ablations; a linear probe to test whether embeddings learn the manual aggregates.

## Main results

**Task 2, 10K users, test week, NDCG@10** (Δ vs GPT-2 SASRec, paired 95% CI):

| Model | Organic new | New |
|---|---|---|
| ALS (tuned) | **0.0297** (+0.012*) | **0.0689** (+0.024*) |
| DecayPop, 2-day half-life | 0.0287 (+0.011*) | 0.0343 (−0.010*) |
| GPT-2 SASRec (full-history prefix) | 0.0182 | 0.0447 |
| GPT-2 + AWR on pushes | 0.0199 (+0.0017*) | 0.0505 (+0.0058*) |
| GPT-2 + off-policy REINFORCE | 0.0141 (−0.004*) | 0.0259 (−0.019*) |
| Exact tilted π_ref·exp(r̂) | 0.0185 (+0.0003*) | 0.0450 (+0.0003) |

**Task 1, 2K users, held-out users, NDCG@10 (and @50) on new items:**

| Ranker | New @10 | New @50 | Organic new @10 |
|---|---|---|---|
| ALS only | 0.0525 | 0.0570 | 0.0249 |
| Graph model alone (LightGCN, user-item-artist) | 0.0659 | 0.0666 | 0.0357 |
| CatBoost on 15 manual features | 0.0755 | 0.0703 | 0.0376 |
| + 2 OpenFE features | 0.0800 | 0.0721 | 0.0345 |
| **+ OpenFE + graph score** | **0.0873** | **0.0765** | **0.0440** (+0.0095* vs without graph) |

(CatBoost rows from `graph.py`; `openfe.py` alone gives manual 0.0798 → + OpenFE 0.0826, Δ +0.0028 with CI including 0.)

## Conclusions

1. **Tuned ALS is the model to beat**, and short-window popularity nearly matches it on organic discoveries: what users find themselves is driven by what is trending this week.
2. **RL helps when it learns from logged outcomes, not from a learned reward.** AWR on pushes weighted by real engagement improves the GPT-2 on both targets. Optimising the off-policy estimate directly (REINFORCE with importance weights) raises it by 21% while relevance collapses: the off-policy estimate is a gate, not a target. The closed-form optimum shows the learned push reward model has little signal to exploit, and the held-out check confirms it ranks pushes worse than the item mean (AUC 0.585 vs 0.700).
3. **Reranking is the big win in Task 1** (+44-52% over ALS); OpenFE adds two sensible relative-popularity features with a small, not yet significant gain on 2K users.
4. **A graph model learns item-level aggregates but not user-level ones.** A linear probe recovers item popularity / organic share / skip rate from its embeddings (R² 0.44-0.80) but not user activity or skip rate (R² ≈ 0). Alone it beats ALS; as one extra feature it significantly improves the boosted reranker, which is the best Task 1 model. Event types and time weighting did not beat plain LightGCN at this training budget; multi-hash item embeddings cut parameters by 45% for a ~4% loss.

Limitations (one training seed for the RL part, small tuning grids, 2K users for Task 1) are listed in each report.

## Reproduce

Requirements: [uv](https://docs.astral.sh/uv/), Python 3.13. A CUDA GPU is recommended for Task 2 (RTX 5070 Ti: ~1.5 h); Task 1 runs on a laptop (~10 min).

```bash
uv sync                          # environment from pyproject.toml / uv.lock
scripts/download_data.sh         # Yambda-50M flat files (~0.8 GB) -> data/yambda/
scripts/reproduce_rl.sh          # Task 2: analysis, tuned baselines, GPT-2, push-based RL, tables
scripts/reproduce_task1.sh       # Task 1: OpenFE + graph model (uses the same split)
```

Run jobs one at a time: each 10K-user job peaks at roughly 10-15 GB of RAM. Outputs go to `rl_experiments/out/` and `task1_feature_engineering/out/`; the numbers in the reports are copied in `rl_experiments/results/` and `task1_feature_engineering/out/*.json`.

## Repository layout

```
rl_experiments/            Task 2 and the shared data / split / metrics code
  common.py                split, events with feedback classes, vocabulary, metrics, bootstrap
  baselines.py             DecayPop, recently played, ItemKNN, co-visitation, ALS (EASE kept for reference)
  tune_baselines.py        validation tuning + one test run per model
  seq.py, train_seq.py     GPT-2 (HF GPT2Model) sequence model with full-history prefix, sampled softmax
  rl_push.py               push rewards, logging-policy model, reward model, AWR / off-policy REINFORCE / tilted, SNIPS
  rm_push_quality.py       held-out reward-model check
  organic_analysis.py      organic vs pushed listening analysis
  summarize_v3.py          result tables
  rl_kl.py, bandit.py, judge.py, organic_eval.py, make_report.py   earlier study (report_v2.md)
task1_feature_engineering/
  openfe.py                candidates, manual features, OpenFE generation + screening, evaluation
  graph.py                 heterogeneous temporal LightGCN, ablations, probe, comparison with boosting
scripts/                   data download and reproduction scripts
```
