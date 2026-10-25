# Decisions log

Each entry: what, why. Agreed with you before you left = **[agreed]**; my own calls while you were away = **[mine]**.

## Setup
- **[agreed]** Own uv venv in this folder: polars, numpy, scipy, torch (MPS), matplotlib, implicit, catboost, pyarrow. Mac only.
- **[agreed]** Nested user samples from one fixed permutation (seed 42): 2K ⊂ 5K ⊂ 10K. Dev on 2K, scale check on 5K / 10K.

## Split (fixed for every model, `common.py`)
- **[agreed]** Global time cut: `train | gap 1d | val 7d | gap 1d | test 7d`, end = max timestamp.
  - train < 24,617,605; val [24,704,005, 25,308,805); test [25,395,205, 26,000,005).
  - Global (not per-user leave-last-out) so no future popularity leaks into train.
  - 7-day windows: ~77% of sampled users have a target (1-day windows: ~50%), so tighter CIs.
  - 1-day gaps: a model fitted tonight serves tomorrow; also stops session continuation from inflating scores.
- **[agreed]** Val stage: fit on `< train_end`, predict val. Test stage: refit on `< val_end` (train + gap + val), predict test; the 1-day gap before test is kept. Hyperparameters are chosen on val only; test is run once per final config.

## Feedback and reward
- **[agreed]** Reward per listen: listen+ (played ≥ 50%) +1, like +2, dislike −1, skip 0.
- **[mine]** A like/dislike belongs to a listen if it is on the same (user, item) within 30 min after the listen start (`join_asof` forward). Only 0.7% of listens get a like, 0.1% a dislike; most of the reward signal is listen+ vs skip. Likes without a nearby listen are ignored.
- **[mine]** Window reward of (user, item) for `reward@10`: dislike > like > listen+ > skip (any dislike in the window → −1).

## Targets and metrics
- **[agreed]** Targets = unique items with listen+ in the window. Two modes:
  - `all`: any such item. Recommendations unfiltered.
  - `new` (primary): items the user never had in the fit data. Recommendations filter out seen items.
- **[mine]** Eval users = users with ≥1 target and some fit history. Cold users are counted (`n_cold_users`, 0–4 on 2K) and not scored: every model would fall back to popularity for them.
- **[mine]** NDCG@10 with the ideal DCG over min(|targets|, 10) (the official Yambda code uses the wrong ideal and is effectively HitRate). Targets outside the vocabulary count as misses.
- **[mine]** recall@100 divides by min(|targets|, 100), so it is not ≥ recall@10 (mean target set is ~90 items). It is a candidate-generation metric.
- **[mine]** reward@10 = sum of window rewards of the 10 recommended items (0 if not interacted, −1 disliked). Logged-feedback proxy of what RL optimises.
- **[mine]** Paired bootstrap 95% CIs over users (`paired_ci`).

## Vocabulary
- **[agreed]** Items with ≥ 5 fit listens (139K items on 2K users, 338K on 10K). Others → OOV token, never recommended.
- **[mine]** EASE on the 20K most popular items only: the dense 139K² inverse does not fit in 16 GB. 20K² float32 inverse takes ~100 s.

## Baselines
- **[mine]** User-item matrices: listen+ = 1, like = 2, skip/dislike = 0, exponential time decay (half-life 30–60 days).
- **[mine]** ItemKNN computed as ((q D^-½) Xᵀ)(X D^-½): exact cosine ItemKNN without materialising the item×item matrix (the 139K² co-occurrence would have billions of non-zeros because users have ~4.6K listens each).
- **[mine]** Co-visitation: next 3 listen+ events, row-normalised, divided by popularity^0.5, query = last 20 listen+ items.
- **[mine]** ALS: implicit, BM25 weighting, 128 factors.

## Neural models (`seq.py`, `train_seq.py`)
- **[mine]** Input token = item + feedback class (skip / listen+ / like / dislike) + organic flag + position. 2-layer causal transformer, d = 64, L = 200. Each user keeps the last 2,000 fit events; training windows end at a uniformly random position.
- **[mine]** SASRec target: next item if it is listen+ / like (skips are context, not targets).
- **[mine]** Sampled softmax with 8,192 shared negatives per step and logQ correction, accidental hits masked. Full softmax over 139K items took 0.8 s/step on MPS (0.09 s for the encoder).
  - Negatives: 50% ∝ popularity^0.75 + 50% uniform. With popularity sampling only, rare items never got a negative gradient and topped the full-vocab ranking (seen in a smoke run: novelty 19.8).
  - Item biases start at log-popularity. Embeddings init N(0, 0.02): PyTorch's default N(0, 1) gave initial logits with std ≈ 8 and CE stuck at ~13.
- **[mine]** SQN (Xin et al. 2020): CE head + Q head (separate output table), double-DQN TD loss on all transitions, γ = 0.5, EMA target net (0.01). The max over actions in the TD target runs over the sampled candidate set. Ranking uses the CE head (the paper's choice); the Q-head ranking is reported as `*_qhead`.
- **[mine]** SA2C-style: CE on all next items (skips included) weighted by exp(clip(advantage, ±2)), advantage = Q(s,a) − E_π Q(s,·) over the candidate set, weights renormalised to mean 1.
- **[mine]** Two-tower: user = mean of the last 50 (item + feedback) embeddings → MLP; item = embedding → MLP; cosine / 0.05; in-batch softmax with logQ; duplicate items in a batch masked.

## RL B (`rl_kl.py`)
- **[mine]** RM = separate encoder (seed 1) + bilinear head, MSE to the logged reward of the next event: E[reward | state, item started]. This is conditional on the item being played, the same selection issue as in your friend's repo; the KL to π_ref partly compensates (π_ref ≈ chance the item is played).
- **[mine]** RM must beat item-mean / user-mean / item+user on held-out val-window transitions (Spearman, MAE, AUC of listen+).
- **[mine]** RLOO: K = 8 samples per state, leave-one-out baseline, exact KL over the full vocabulary, policy in eval mode (no dropout), Adam 1e-4.
- **[mine]** Tilted policy: rank by log π_ref + r̂/β, β ∈ {0.1, 0.3, 1, 3}.

## RL C (`bandit.py`)
- **[mine]** Toy simulator, labelled as such: the RM is the "true" world, the agent only sees the SASRec prior on each user's top-200 pool and learns item offsets online (shared across users). Slates of 10, no repeats within an episode, 8 rounds (80 of 200 pool items; with 20 rounds every policy showed the whole pool and all tied), noise σ = 0.7.

## Bug found and fixed (all neural results rerun)
- **[mine]** The vocab was sorted by count with arbitrary tie order (polars `group_by` is not order-stable), so a checkpoint reloaded in another process mapped indices to different items. In-process metrics were right; anything reloaded (RM, RLOO reference, judge scoring of lists) was wrong. Fixed by tie-breaking on item_id; saved lists now store raw item ids. Everything was retrained / rerun after the fix.

## Configs fixed on 2K val, then run once on test
- **[mine]** Neural models: 6,000 steps, B = 128, lr 1e-3 (CE still decreasing slowly at 6K; more steps not affordable for 3 sample sizes on the Mac). Same steps for every size.
- **[mine]** RLOO: β ∈ {0.3, 1}, 600 steps. On val, β = 0.3 was where the tilted policy started to gain RM reward without a large NDCG loss; β = 1 is the near-reference point.
- **[mine]** Tilted policy: β ∈ {0.1, 0.3, 1, 3} on every stage (no training, cheap).
- **[mine]** 5K / 10K: baselines + SASRec + SQN + SA2C + RM + tilted + RLOO + bandit + judge. Two-tower only on 2K (clearly behind on val, 7 min per run but not informative to scale).
- **[mine]** Val observation used for no decision but worth noting: RLOO at β = 0.3 reached E[r̂] 0.742 at KL 0.041, while the exact tilted policy at β = 1 reached 0.756 at KL 0.019, so RLOO sits *below* the closed-form frontier (more KL, less reward).

## Organic evaluation (added after you asked, `organic_eval.py`)
- **[agreed]** Primary metric stays new-item NDCG@10. Added a stricter target: new items with an *organic* listen+ in the window (`is_organic = 1`: the user found it, the recommender did not push it). Less tied to the logging policy, so closer to user preference than to "what the old system showed".
- **[mine]** Computed post-hoc from the saved top-10 'new'-mode lists (no retraining): NDCG@10, Recall@10, Reward@10 + paired CI vs SASRec. Recall@100 is not available there (only top 10 is saved).
- **[mine]** Not added to the running test chain (editing a running zsh script is unsafe); run per sample size after it finishes.

## Stopped on the Mac (2026-10-25), moving to the GPU server
- Finished: 2K val (everything), 2K test: baselines, SASRec, SQN (+ Q head). Not run: 2K test SA2C / two-tower / RM / tilted / RLOO / bandit / judge; all 5K and 10K runs.
- `seq.py` DEVICE now prefers CUDA, then MPS, then CPU.

## Second full run with NDCG@50 / @100 (2026-10-25, GPU server)
- **[agreed]** Added NDCG@50 and NDCG@100 (ideal DCG over min(|targets|, k)) for all, new and organic-new targets; top-100 lists are now saved. Full pipeline rerun on the server.
- **[mine]** The rerun is effectively a second training seed (same code, data, configs; CUDA training is not bit-identical). Run 1 results kept in `results_run1/`. Finding: SASRec moves by up to 0.005 NDCG@10 between runs, the same size as the RL A gain, so RL A's per-run significance did not replicate; ALS's lead and RL B conclusions did.

## Session 2026-10-26: organic analysis, fair re-staging, GPT-2, push-based RL (2-hour budget)
- **[agreed]** Critique: the old comparison gave models different information (ALS full history vs SASRec last 200 events, EASE top 20K items, untuned configs). Re-staged as below; old results stay in `out_run2/` (run 2) and `out_v1/` (run 1).
- **Organic analysis** (`organic_analysis.py`, all 10K users, `out/organic_analysis.json`):
  - Pushed listens are skipped *less* than organic ones (31% vs 41%): pushes are mostly background radio, so listen+ measures listening mode more than preference.
  - Organic listens are 81% repeats; organic first listens are of less popular items (median pop 377 vs 755). 65% of first-ever item listens come from pushes.
  - Per-user organic share: q10 9%, median 54%, q90 97% (radio vs library listeners).
  - After a pushed first artist contact, an organic listen of *another* track of the artist 1-7 days later: skip 0.97%, listen+ 1.04%, like 6.9%, dislike 0.72%. "Like -> return" is mostly the same liked track replayed from the collection (50 of 52 points). Listen+ on a push carries ~no artist-discovery signal.
- **[agreed]** Targets: organic new + all new co-primary; model selection on organic-new NDCG@10. `org_new` is now a built-in mode of `Stage` (evaluated on the seen-filtered 'new' lists).
- **[agreed]** GPT-2 is mandatory (Task 2): `GPT2Encoder` = Hugging Face `GPT2Model` (2 layers, d 128, 4 heads, L 200) via `inputs_embeds`; our item table tied to the output; tokens = item + feedback + organic.
- **[agreed]** Full history for sequence models: a prefix token = projection of the mean (item + feedback) embedding of up to 256 events sampled from the history *before* the window (random in training, evenly spaced at inference; point-in-time, no leakage). No 2,000-event cap. Steps chosen on 10K val (eval at 3K/6K/9K/12K), 2 seeds on test.
- **[agreed]** Base algorithms tuned on 10K val, selection on organic-new NDCG@10. **[mine]** Grids cut to 2-3 configs per model to fit the 2-hour budget (~1 min per config on 10K users). EASE: 35K items with a CPU float32 inverse (the 16 GB GPU could not hold factor + inverse + workspace); coverage of new targets is reported.
- **[mine]** Early tuning result: DecayPop with a 1-day half-life is the best popularity variant on organic new (0.0285 vs 0.021 at 7 days): organic discoveries are dominated by what is trending that day.
- **[agreed]** Push-based RL (`rl_push.py`): actions = pushes (`is_organic = 0`); reward = 1[like] + 1[organic listen of another track of the artist in (t+1d, t+3d]] − 1[dislike] (3-day window so test-week pushes can be scored inside the week); π_ref = GPT-2 SASRec; β̂ = GPT-2 trained on pushed items (logging-policy / propensity model); r̂ = GPT-2 + bilinear head on push rewards. Policies: AWR (+KL), off-policy REINFORCE with clipped π/β̂ and top-K correction (Chen et al. 2019) (+KL), exact tilted π_ref·exp(r̂/b). **[mine]** One config each (b_kl 0.1, 500 steps, lr 1e-4), one seed: untuned. Evaluation: NDCG on the test week + SNIPS on held-out test-week pushes (weights clipped at 10, effective sample share reported).
- **[mine]** Asof-join shortcut in the return reward: only the *first* organic listen of the artist after t+1d is checked; if it is the same track, a later other-track return is missed (slight undercount).
- **Server crash (2026-10-25 ~22:40).** Baseline tuning (with a dense 35K² EASE on the CPU, ~15 GB peak) ran in parallel with the 10K GPT-2 val run; RAM ran out and the server stopped responding until rebooted. Lost: EASE tuning, base-algorithm test runs, the push reward model and RL fine-tunes, GPT-2 seed 1. Kept: tuning of DecayPop / ItemKNN / co-visitation / ALS, GPT-2 step selection (9K steps), GPT-2 test seed 0, β̂.
- **[agreed]** From then on: strictly sequential jobs, each in `systemd-run --user --scope -p MemoryMax=22G -p MemorySwapMax=0`, `nice -n 10`, 8 BLAS threads (`run_final.sh`).
- **[mine]** Because of the remaining time: EASE runs one untuned config (λ 500, half-life 60 d, 35K items); GPT-2 has one seed (run-to-run spread of SASRec from runs 1-2 is ~0.005 NDCG@10, see above); RL auxiliaries 3K steps, fine-tunes 300 steps.
- **[agreed]** EASE dropped from the final comparison (the dense inverse does not scale to the 338K-item vocabulary; the top-35K variant caused the server crash). Its time went to a second tuning round: DecayPop half-life 0.5 / 2 d, ItemKNN 3 / 7 d, co-visitation (10, 100, α 0.3) / (20, 200), ALS 64 / 192 factors and (128, reg 0.001, 14 d). Chosen: DecayPop 2 d, ItemKNN 14 d, co-visitation (10, 100, 0.3), ALS (128, 0.001, 14 d).
- **[mine]** Held-out check of the push reward model (`rm_push_quality.py`): AUC 0.585 vs 0.700 for the item mean; the reward model is weak, which explains why the tilted policy does not move.
