## 10K users, test week

Users with targets: 5210 (organic new), 7075 (new), 7731 (all). Δ = paired bootstrap 95% CI vs `sasrec_s0`, * = excludes 0.

| Group | Model | NDCG@10 organic new | Δ organic new | NDCG@10 new | Δ new | NDCG@100 new | Recall@100 new | Cov@10 new | NDCG@10 all |
|---|---|---|---|---|---|---|---|---|---|
| Heuristics | DecayPop (tuned half-life) | 0.0287 | +0.0105 [+0.0071, +0.0139]* | 0.0343 | -0.0104 [-0.0136, -0.0073]* | 0.0388 | 0.060 | 0.0002 | 0.0573 |
| Heuristics | Recently played (all targets only) | 0.0198 | +0.0016 [-0.0014, +0.0048] | 0.0234 | -0.0213 [-0.0243, -0.0185]* | 0.0320 | 0.053 | 0.0003 | 0.3737 |
| Co-occurrence / CF | ItemKNN (tuned) | 0.0085 | -0.0097 [-0.0119, -0.0074]* | 0.0281 | -0.0166 [-0.0192, -0.0139]* | 0.0288 | 0.038 | 0.1122 | 0.1278 |
| Co-occurrence / CF | Co-visitation (tuned) | 0.0146 | -0.0036 [-0.0059, -0.0013]* | 0.0197 | -0.0250 [-0.0275, -0.0226]* | 0.0309 | 0.053 | 0.1428 | 0.1884 |
| Co-occurrence / CF | ALS (tuned) | 0.0297 | +0.0115 [+0.0088, +0.0143]* | 0.0689 | +0.0242 [+0.0210, +0.0272]* | 0.0853 | 0.129 | 0.0255 | 0.1990 |
| GPT-2 | GPT-2 SASRec, seed 0 | 0.0182 | — | 0.0447 | — | 0.0589 | 0.092 | 0.0661 | 0.1914 |
| Push-based RL | β̂: imitate pushes (logging policy) | 0.0052 | -0.0130 [-0.0152, -0.0106]* | 0.0152 | -0.0295 [-0.0319, -0.0272]* | 0.0217 | 0.035 | 0.0059 | 0.0548 |
| Push-based RL | AWR on pushes + KL | 0.0199 | +0.0017 [+0.0007, +0.0027]* | 0.0505 | +0.0058 [+0.0045, +0.0071]* | 0.0647 | 0.100 | 0.0578 | 0.1903 |
| Push-based RL | Off-policy REINFORCE (π/β̂, top-K) + KL | 0.0141 | -0.0041 [-0.0053, -0.0030]* | 0.0259 | -0.0188 [-0.0204, -0.0171]* | 0.0373 | 0.061 | 0.0574 | 0.1597 |
| Push-based RL | Exact tilted π_ref·exp(r̂/1) | 0.0185 | +0.0003 [+0.0000, +0.0006]* | 0.0450 | +0.0003 [-0.0000, +0.0007] | 0.0589 | 0.092 | 0.0656 | 0.1916 |
| Push-based RL | Exact tilted π_ref·exp(r̂/0.3) | 0.0184 | +0.0002 [-0.0003, +0.0007] | 0.0445 | -0.0002 [-0.0009, +0.0004] | 0.0581 | 0.090 | 0.0639 | 0.1913 |

### Tuning (10K val, selection on organic-new NDCG@10)

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

### GPT-2 SASRec step selection (10K val)

| Steps | organic new | new |
|---|---|---|
| 3000 | 0.0169 | 0.0383 |
| 6000 | 0.0193 | 0.0446 |
| 9000 | 0.0200 | 0.0474 |
| 12000 | 0.0200 | 0.0472 |

### Off-policy value on held-out test-week pushes (SNIPS, weights clipped at 10)

| Policy | SNIPS reward / push | Logged mean | Effective sample share |
|---|---|---|---|
| push_ref | 0.0714 | 0.0661 | 0.61 |
| push_beta | 0.0661 | 0.0661 | 1.00 |
| push_tilt_b0.3 | 0.0725 | 0.0661 | 0.61 |
| push_tilt_b1 | 0.0717 | 0.0661 | 0.61 |
| push_awr | 0.0705 | 0.0661 | 0.65 |
| push_opc | 0.0866 | 0.0661 | 0.47 |
