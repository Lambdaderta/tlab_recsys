"""Non-neural baselines: popularity, re-listen, ItemKNN, co-visitation, EASE, ALS.

uv run python baselines.py --n 2000 --stage val [--models pop,recent,...]
"""
import argparse
import time

import numpy as np
import polars as pl
import scipy.sparse as sp

from common import DAY, PLUS, LIKE, Stage, decay, recommend, save_result, user_item_matrix


def pop_scores(stage, half_life_days=None):
    f = stage.fit.filter(pl.col('fb').is_in([PLUS, LIKE]))
    w = np.ones(f.height, np.float32) if half_life_days is None else decay(f['timestamp'].to_numpy(), stage.fit_end, half_life_days)
    return np.bincount(f['idx'].to_numpy(), weights=w, minlength=stage.vocab.size).astype(np.float32)


def run_pop(stage, hls=(None, 7), names=('mostpop', 'decaypop')):
    out = {}
    for name, hl in zip(names, hls):
        p = pop_scores(stage, hl)
        out[name] = save_result(name, stage, recommend(stage, lambda rows: np.tile(p, (len(rows), 1))))
    return out


def run_recent(stage):
    """Re-listen baseline: the user's own items by decayed listen+ weight, popularity as a tiny tie-break."""
    X, _ = user_item_matrix(stage, half_life_days=14, users=stage.users)
    p = pop_scores(stage, 7)
    p = 1e-6 * p / p.max()
    return save_result('recent', stage, recommend(stage, lambda rows: X[rows].toarray() + p))


def run_itemknn(stage, hl=30, name='itemknn'):
    """Cosine ItemKNN without building the item x item matrix:
    scores = q (D^-1/2 X^T X D^-1/2) = ((q D^-1/2) X^T) (X D^-1/2), which is O(users^2 + nnz)."""
    X, uids = user_item_matrix(stage, half_life_days=hl)
    d = np.asarray(X.power(2).sum(0)).ravel()
    inv = sp.diags(np.where(d > 0, 1 / np.sqrt(d), 0).astype(np.float32))
    A = (X @ inv).tocsr()
    Q, _ = user_item_matrix(stage, half_life_days=hl, users=stage.users)
    Q = (Q @ inv).tocsr()
    At = A.T.tocsr()
    return save_result(name, stage, recommend(stage, lambda rows: ((Q[rows] @ At) @ A).toarray()), {'hl': hl})


def run_covisit(stage, window=3, last_n=20, alpha=0.5, name='covisit'):
    """Sequential co-visitation (OTTO style): i -> items listened (listen+) within the next `window` events.
    Row-normalised and divided by target popularity^alpha; user score sums rows of their last items."""
    f = stage.fit.filter(pl.col('fb').is_in([PLUS, LIKE]) & (pl.col('idx') > 1)).sort('uid', 'timestamp')
    idx, uid = f['idx'].to_numpy(), f['uid'].to_numpy()
    w = decay(f['timestamp'].to_numpy(), stage.fit_end, 60)
    src, dst, val = [], [], []
    for lag in range(1, window + 1):
        same = uid[:-lag] == uid[lag:]
        src.append(idx[:-lag][same]); dst.append(idx[lag:][same]); val.append(w[lag:][same] / lag)
    V = stage.vocab.size
    C = sp.csr_matrix((np.concatenate(val), (np.concatenate(src), np.concatenate(dst))), shape=(V, V))
    C.sum_duplicates()
    rs = np.asarray(C.sum(1)).ravel()
    cs = np.asarray(C.sum(0)).ravel()
    C = sp.diags(np.where(rs > 0, 1 / rs, 0)) @ C @ sp.diags(np.where(cs > 0, cs ** -alpha, 0))
    C = C.tocsr().astype(np.float32)
    # query: each eval user's last `last_n` listen+ items, more recent ones weighted higher
    last = (f.filter(pl.col('uid').is_in(pl.Series(stage.users).implode()))
            .group_by('uid').tail(last_n).with_columns(pos=pl.int_range(pl.len()).over('uid')))
    rows = np.array([stage.uid_row[u] for u in last['uid'].to_numpy()])
    q = sp.csr_matrix((0.9 ** (last_n - 1 - last['pos'].to_numpy()), (rows, last['idx'].to_numpy())),
                      shape=(len(stage.users), V), dtype=np.float32)
    return save_result(name, stage, recommend(stage, lambda r: (q[r] @ C).toarray()),
                       {'window': window, 'last_n': last_n, 'alpha': alpha})


def run_ease(stage, n_items=35_000, lam=500.0, hl=60, name='ease'):
    """EASE (Steck 2019) on the n_items most popular vocab items: B = I - P diag(1/diag P), P = (X^T X + lam I)^-1.
    Dense float32 inverse on the CPU (35K^2 = 4.9 GB; on the 16 GB GPU the factor + inverse + workspace did not fit).
    Items outside the kept set are never recommended; `target_coverage_new` says how much of the targets they cover."""
    X, _ = user_item_matrix(stage, half_life_days=hl)
    keep = np.arange(2, 2 + min(n_items, stage.vocab.size - 2))  # vocab is sorted by popularity
    Xs = X[:, keep].tocsc()
    t = time.time()
    G = (Xs.T @ Xs).toarray().astype(np.float32)
    G[np.diag_indices_from(G)] += lam
    P = np.linalg.inv(G)
    del G
    B = P / (-np.diag(P))
    del P
    B[np.diag_indices_from(B)] = 0
    print(f'EASE inverse {time.time() - t:.0f}s')
    Q, _ = user_item_matrix(stage, half_life_days=hl, users=stage.users)
    Q = Q[:, keep].tocsr()
    kept = set(stage.vocab.items[keep - 2])
    tgt = [i for t in stage.targets['new'] for i in t]
    cov = float(np.mean([i in kept for i in tgt]))

    def scores(rows):
        s = np.full((len(rows), stage.vocab.size), -1e9, np.float32)
        s[:, keep] = Q[rows] @ B
        return s
    return save_result(name, stage, recommend(stage, scores),
                       {'ease_items': len(keep), 'lambda': lam, 'hl': hl, 'target_coverage_new': cov})


def run_als(stage, factors=128, reg=0.05, iters=15, hl=60, k1=100, name='als'):
    from implicit.als import AlternatingLeastSquares
    from implicit.nearest_neighbours import bm25_weight
    X, uids = user_item_matrix(stage, half_life_days=hl)
    W = bm25_weight(X, K1=k1, B=0.8).tocsr().astype(np.float32)
    model = AlternatingLeastSquares(factors=factors, regularization=reg, iterations=iters, random_state=0)
    model.fit(W, show_progress=False)
    row = {u: i for i, u in enumerate(uids)}
    U = model.user_factors[[row[u] for u in stage.users]]
    return save_result(name, stage, recommend(stage, lambda r: U[r] @ model.item_factors.T),
                       {'factors': factors, 'reg': reg, 'iters': iters, 'hl': hl, 'k1': k1})


RUNNERS = {'pop': run_pop, 'recent': run_recent, 'itemknn': run_itemknn, 'covisit': run_covisit,
           'ease': run_ease, 'als': run_als}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--stage', default='val')
    ap.add_argument('--models', default=','.join(RUNNERS))
    a = ap.parse_args()
    stage = Stage(a.n, a.stage)
    for m in a.models.split(','):
        t = time.time()
        RUNNERS[m](stage)
        print(f'{m}: {time.time() - t:.0f}s')
