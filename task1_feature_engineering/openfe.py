"""Task 1: OpenFE-style automatic feature generation for a CatBoost reranker on Yambda (2K users).

Pipeline
  1. Candidates at the train cutoff: top-50 ALS ∪ top-50 DecayPop (unseen items only), for the users with targets
     in the validation week of the fixed split (rl_experiments/common.py: train | 1d gap | val 7d | ...).
  2. Base (manual) features: model scores/ranks, item / user / artist statistics, all from fit data only.
  3. OpenFE: generate candidate features with operators (unary, binary, group-by), screen them with
     feature_gain (a tiny booster on ONE feature on top of the base model's raw scores) and successive halving,
     then keep the survivors with positive importance in a full model (stage 2).
  4. Check: NDCG@10 on held-out users, base vs base + OpenFE features (paired bootstrap CI).
Users are split 50 / 25 / 25 into tr (fit), va (gain / selection), te (final check): no user in two parts.

uv run python openfe.py   -> out/openfe_report.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from catboost import CatBoostClassifier, Pool

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'rl_experiments'))
from common import DAY, LIKE, PLUS, SKIP, Stage, decay, paired_ci, user_item_matrix  # noqa: E402

ARTIST = HERE.parent / 'data' / 'yambda' / 'artist_item_mapping.parquet'
OUT = HERE / 'out'
OUT.mkdir(exist_ok=True)
N_USERS, N_CAND, SEED = 2000, 50, 0
THREADS = 8


# ---------- 1. candidates ----------

def top_unseen(scores, seen_rows, k):
    s = scores.copy()
    s[:, :2] = -np.inf
    r, c = seen_rows.nonzero()
    s[r, c] = -np.inf
    part = np.argpartition(-s, k, axis=1)[:, :k]
    return part, np.take_along_axis(s, part, 1)


def build_candidates(stage):
    from implicit.als import AlternatingLeastSquares
    from implicit.nearest_neighbours import bm25_weight
    X, uids = user_item_matrix(stage, half_life_days=30)  # tuned ALS config from rl_experiments
    als = AlternatingLeastSquares(factors=128, regularization=0.01, iterations=15, random_state=SEED)
    als.fit(bm25_weight(X, K1=100, B=0.8).tocsr().astype(np.float32), show_progress=False)
    row = {u: i for i, u in enumerate(uids)}
    U = als.user_factors[[row[u] for u in stage.users]]
    V = als.item_factors
    f = stage.fit.filter(pl.col('fb').is_in([PLUS, LIKE]))
    pop1 = np.bincount(f['idx'].to_numpy(), weights=decay(f['timestamp'].to_numpy(), stage.fit_end, 1),
                       minlength=stage.vocab.size).astype(np.float32)
    seen = stage.seen_matrix()
    frames = []
    for s in range(0, len(stage.users), 256):
        rows = np.arange(s, min(s + 256, len(stage.users)))
        a_idx, _ = top_unseen(U[rows] @ V.T, seen[rows], N_CAND)
        p_idx, _ = top_unseen(np.tile(pop1, (len(rows), 1)), seen[rows], N_CAND)
        for i, r in enumerate(rows):
            idx = np.union1d(a_idx[i], p_idx[i])
            frames.append(pl.DataFrame({'row': np.full(len(idx), r, np.int32), 'idx': idx.astype(np.int32)}))
    cand = pl.concat(frames)
    cand = cand.with_columns(
        als_score=pl.Series([float(U[r] @ V[i]) for r, i in zip(cand['row'], cand['idx'])], dtype=pl.Float32),
        pop1=pl.Series(pop1[cand['idx'].to_numpy()]))
    return cand


# ---------- 2. base features + labels ----------

def base_features(stage, cand):
    items = np.concatenate([[0, 0], stage.vocab.items])
    cand = cand.with_columns(uid=pl.Series(stage.users[cand['row'].to_numpy()]),
                             item_id=pl.Series(items[cand['idx'].to_numpy()]))
    art = pl.read_parquet(ARTIST).unique('item_id')
    fit = stage.fit.join(art, on='item_id', how='left').with_columns(
        plus=pl.col('fb').is_in([PLUS, LIKE]).cast(pl.Float32), skip=(pl.col('fb') == SKIP).cast(pl.Float32),
        recent=(pl.col('timestamp') >= stage.fit_end - 30 * DAY))
    item = fit.group_by('item_id').agg(item_cnt=pl.len(), item_skip=pl.col('skip').mean(),
                                       item_org=pl.col('is_organic').mean(), item_recent=pl.col('recent').sum())
    user = fit.group_by('uid').agg(u_events=pl.len(), u_org=pl.col('is_organic').mean(), u_skip=pl.col('skip').mean(),
                                   u_recent=pl.col('recent').sum())
    artist = fit.group_by('artist_id').agg(artist_pop=pl.col('plus').sum())
    ua = fit.group_by('uid', 'artist_id').agg(ua_cnt=pl.len(), ua_plus=pl.col('plus').mean())
    df = (cand.join(art, on='item_id', how='left').join(item, on='item_id', how='left')
          .join(user, on='uid', how='left').join(artist, on='artist_id', how='left')
          .join(ua, on=['uid', 'artist_id'], how='left')
          .with_columns(pl.col('ua_cnt', 'artist_pop', 'item_recent').fill_null(0),
                        als_rank=pl.col('als_score').rank(descending=True).over('uid'),
                        pop_rank=pl.col('pop1').rank(descending=True).over('uid')))
    tgt_new = {u: t for u, t in zip(stage.users, stage.targets['new'])}
    tgt_org = {u: t for u, t in zip(stage.users, stage.targets['org_new'])}
    df = df.with_columns(
        y=pl.Series([int(i in tgt_new[u]) for u, i in zip(df['uid'], df['item_id'])]),
        y_org=pl.Series([int(i in tgt_org[u]) for u, i in zip(df['uid'], df['item_id'])]))
    return df.sort('uid', 'item_id')


BASE = ['als_score', 'als_rank', 'pop1', 'pop_rank', 'item_cnt', 'item_skip', 'item_org', 'item_recent',
        'u_events', 'u_org', 'u_skip', 'u_recent', 'artist_pop', 'ua_cnt', 'ua_plus']


# ---------- 3. OpenFE: generation, feature_gain, successive halving, stage 2 ----------

def generate(base):
    """Candidate features as polars expressions (OpenFE operators: unary, binary, group-by)."""
    c = pl.col
    ex = {}
    for a in base:
        ex[f'log({a})'] = c(a).abs().log1p()
        ex[f'urank({a})'] = c(a).rank().over('uid') / pl.len().over('uid')   # rank within the user's candidates
        ex[f'udiff({a})'] = c(a) - c(a).mean().over('uid')                   # deviation from the user's mean
        ex[f'amean({a})'] = c(a).mean().over('artist_id')                    # artist-level aggregate
    for i, a in enumerate(base):
        for b in base[i + 1:]:
            ex[f'{a}+{b}'] = c(a) + c(b)
            ex[f'{a}-{b}'] = c(a) - c(b)
            ex[f'{a}*{b}'] = c(a) * c(b)
            ex[f'{a}/{b}'] = c(a) / (c(b).abs() + 1e-6)
    return ex


def logloss_raw(y, raw):
    return float(np.mean(np.logaddexp(0, raw) - y * raw))  # stable: no log(p + eps) clipping


def feature_gain(f_tr, y_tr, base_tr, f_va, y_va, base_va, iters=50):
    """Your criterion: a tiny booster on ONE feature, starting from the base model's raw scores.
    > 0 means the feature adds information the base model does not have."""
    m = CatBoostClassifier(iterations=iters, depth=3, learning_rate=0.1, verbose=0, random_seed=SEED,
                           thread_count=THREADS)
    m.fit(Pool(f_tr.reshape(-1, 1), y_tr, baseline=base_tr))
    raw_va = base_va + m.predict(f_va.reshape(-1, 1), prediction_type='RawFormulaVal')
    return logloss_raw(y_va, base_va) - logloss_raw(y_va, raw_va)


def fit_cb(X, y, iters=300):
    m = CatBoostClassifier(iterations=iters, depth=6, learning_rate=0.08, verbose=0, random_seed=SEED,
                           thread_count=THREADS)
    m.fit(X, y)
    return m


def oof_raw(X, y, groups, X_va, k=3):
    """Out-of-fold raw scores of the base model on its own training rows (folds by user), and va scores as the
    mean of the SAME fold models, so tr and va baselines are equally calibrated. (A va baseline from one model fit
    on all of tr is more confident; each single-feature booster then learns a calibration shift that is wrong on va,
    and every feature looks harmful.)"""
    out, va = np.zeros(len(y)), np.zeros(len(X_va))
    ug = np.unique(groups)
    fold = dict(zip(ug, np.random.default_rng(SEED).integers(0, k, len(ug))))
    f = np.array([fold[g] for g in groups])
    for i in range(k):
        m = fit_cb(X[f != i], y[f != i])
        out[f == i] = m.predict(X[f == i], prediction_type='RawFormulaVal')
        va += m.predict(X_va, prediction_type='RawFormulaVal') / k
    return out, va


def successive_halving(F, names, y_tr, b_tr, y_va, b_va, users_tr, users_va, rounds=(0.25, 0.5, 1.0), keep_final=20):
    """Score all candidates on a user subsample, keep the better half BY RANK, double the data, repeat.
    The sign filter (gain > 0) is applied only in the last round: on small subsamples single-feature gains
    are mostly noise, so filtering by sign early throws away good features."""
    rng = np.random.default_rng(SEED)
    order_tr, order_va = rng.permutation(np.unique(users_tr)), rng.permutation(np.unique(users_va))
    alive = list(range(len(names)))
    history = []
    for r, frac in enumerate(rounds):
        mtr = np.isin(users_tr, order_tr[:max(1, int(frac * len(order_tr)))])
        mva = np.isin(users_va, order_va[:max(1, int(frac * len(order_va)))])
        t = time.time()
        gains = [feature_gain(F['tr'][:, j][mtr], y_tr[mtr], b_tr[mtr], F['va'][:, j][mva], y_va[mva], b_va[mva])
                 for j in alive]
        ranked = sorted(zip(gains, alive), reverse=True)
        print('  top gains:', [(names[j], round(gn, 6)) for gn, j in ranked[:5]], flush=True)
        n_keep = keep_final if r == len(rounds) - 1 else max(keep_final, len(alive) // 2)
        last = r == len(rounds) - 1
        alive = [j for g, j in ranked[:n_keep] if g > 0 or not last]
        history.append({'round': r, 'frac': frac, 'scored': len(gains), 'kept': len(alive), 'sec': round(time.time() - t)})
        print(history[-1], flush=True)
    final = {names[j]: g for g, j in ranked if j in alive}
    return final, history


# ---------- 4. evaluation ----------

def ndcg_per_user(df, score, stage, mode, k=10):
    """NDCG@k over each user's candidates; the ideal DCG uses ALL the user's targets (missed candidates count)."""
    disc = 1 / np.log2(np.arange(2, k + 2))
    tgt = {u: t for u, t in zip(stage.users, stage.targets[mode])}
    d = df.select('uid', 'item_id').with_columns(s=pl.Series(score)).sort(['uid', 's'], descending=[False, True])
    out = {}
    for u, its in d.group_by('uid', maintain_order=True).agg(pl.col('item_id').head(k)).iter_rows():
        t = tgt[u]
        if t:
            hit = np.array([i in t for i in its] + [False] * (k - len(its)))
            out[u] = float((hit * disc).sum() / disc[:min(len(t), k)].sum())
    return out


def main():
    t0 = time.time()
    stage = Stage(N_USERS, 'val')
    df = base_features(stage, build_candidates(stage))
    print('candidates', df.height, 'positives', int(df['y'].sum()), 'org positives', int(df['y_org'].sum()),
          f'{time.time() - t0:.0f}s', flush=True)

    users = np.random.default_rng(SEED).permutation(stage.users)
    part = {u: ('tr' if i < len(users) // 2 else 'va' if i < 3 * len(users) // 4 else 'te') for i, u in enumerate(users)}
    df = df.with_columns(part=pl.Series([part[u] for u in df['uid']]))
    exprs = generate(BASE)
    names = list(exprs)
    gen = df.select(**exprs).fill_nan(None).fill_null(0)
    S = {p: df['part'] == p for p in ('tr', 'va', 'te')}
    Xb = {p: df.filter(S[p]).select(BASE).fill_null(0).to_numpy() for p in S}
    F = {p: gen.filter(S[p]).to_numpy().astype(np.float32) for p in S}
    y = {p: df.filter(S[p])['y'].to_numpy() for p in S}
    g = {p: df.filter(S[p])['uid'].to_numpy() for p in S}
    # drop constant candidates (e.g. udiff of a user-level feature is 0 everywhere)
    keep = [j for j in range(len(names)) if F['tr'][:, j].std() > 0 and F['va'][:, j].std() > 0]
    names = [names[j] for j in keep]
    F = {p: F[p][:, keep] for p in F}
    print('generated candidate features:', len(exprs), 'non-constant:', len(names), flush=True)

    base_tr, base_va = oof_raw(Xb['tr'], y['tr'], g['tr'], Xb['va'])
    screened, history = successive_halving(F, names, y['tr'], base_tr, y['va'], base_va, g['tr'], g['va'])

    # stage 2: full model on base + screened features; keep those with positive importance
    cols = [names.index(n) for n in screened]
    m2 = fit_cb(np.hstack([Xb['tr'], F['tr'][:, cols]]), y['tr'])
    imp = m2.get_feature_importance()[len(BASE):]
    selected = [n for n, v in zip(screened, imp) if v > 0]
    sel = [names.index(n) for n in selected]

    # final check on held-out users: same model settings, base vs base + selected
    m_full = fit_cb(np.vstack([np.hstack([Xb['tr'], F['tr'][:, sel]]), np.hstack([Xb['va'], F['va'][:, sel]])]),
                    np.concatenate([y['tr'], y['va']]))
    m_base2 = fit_cb(np.vstack([Xb['tr'], Xb['va']]), np.concatenate([y['tr'], y['va']]))
    te = df.filter(S['te'])
    scores = {'als_only': te['als_score'].to_numpy(), 'popularity_only': te['pop1'].to_numpy(),
              'catboost_base': m_base2.predict_proba(Xb['te'])[:, 1],
              'catboost_base+openfe': m_full.predict_proba(np.hstack([Xb['te'], F['te'][:, sel]]))[:, 1]}
    report = {'n_candidates_rows': df.height, 'generated': len(names), 'screened': screened, 'selected': selected,
              'stage2_importance': dict(zip(screened, map(float, imp))), 'halving': history, 'test': {}}
    for mode in ('new', 'org_new'):
        for k in (10, 50):
            per = {n: ndcg_per_user(te, v, stage, mode, k) for n, v in scores.items()}
            common_users = sorted(set.intersection(*[set(p) for p in per.values()]))
            arr = {n: np.array([p[u] for u in common_users]) for n, p in per.items()}
            report['test'][f'{mode}@{k}'] = {'users': len(common_users), **{n: float(v.mean()) for n, v in arr.items()},
                                             'delta_openfe_vs_base': paired_ci(arr['catboost_base+openfe'], arr['catboost_base'])}
    report['sec'] = round(time.time() - t0)
    (OUT / 'openfe_report.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report['test'], indent=1))
    print('selected:', selected)


if __name__ == '__main__':
    main()
