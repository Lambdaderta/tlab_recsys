"""Task 1, research question 2: a user-item-artist graph model vs manual aggregate features and boosting.

Model: heterogeneous LightGCN.
  Nodes: users, items, artists. Relations (event types as edge types, learned weight per relation):
    pos  - user-item listen+ (+ like counts double), weight = time-decayed sum (half-life 30 d)
    skip - user-item skips, time-decayed
    org  - user-item ORGANIC listen+ (user chose it), time-decayed
    art  - item-artist membership
  Propagation: H_{l+1} = sum_r w_r * Ahat_r H_l (symmetric-normalised adjacency per relation), output = mean of layers.
Temporal training (no future edges, cf. temporal neighbour sampling in Utushkin et al., CIKM 2026, arXiv 2608.27413):
  graph from events before T0 = cutoff - 14 d; BPR on the NEW listen+ items users picked up in [T0, cutoff);
  inference: the same embeddings propagated over the graph of all events before the cutoff.
Ablations: plain LightGCN (one binary user-item relation, no decay, no types); multi-hash item embeddings
  (2 hashes into 2^15 buckets, summed; Utushkin et al.) instead of a full item table.
Evaluation on the openfe.py candidates / user split: graph alone, CatBoost base + graph score, linear probe of
  manual aggregates from the embeddings. NDCG@10 and @50 on held-out users.

uv run python graph.py   -> out/graph_report.json
"""
import json
import time

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from openfe import (ARTIST, BASE, OUT, SEED, build_candidates, base_features, fit_cb, generate, ndcg_per_user)
from common import DAY, LIKE, PLUS, SKIP, Stage, decay, paired_ci

torch.manual_seed(SEED)
D, LAYERS, STEPS, BATCH, HIST_DAYS = 64, 2, 150, 4096, 14


class Graph:
    """Node ids: users [0, U), items U + vocab idx, artists U + V + artist index."""

    def __init__(self, stage):
        self.uids = np.sort(stage.fit['uid'].unique().to_numpy())
        self.U, self.V = len(self.uids), stage.vocab.size
        art = pl.read_parquet(ARTIST).unique('item_id')
        m = pl.DataFrame({'item_id': stage.vocab.items, 'idx': np.arange(2, self.V)}).join(art, on='item_id', how='left')
        a_codes, self.item_artist = np.unique(m['artist_id'].fill_null(-1).to_numpy(), return_inverse=True)
        self.A = len(a_codes)
        self.item_idx = m['idx'].to_numpy()
        self.N = self.U + self.V + self.A
        self.urow = pl.DataFrame({'uid': self.uids, 'u': np.arange(self.U)})

    def adj(self, rows, cols, w):
        """Symmetric-normalised sparse adjacency of an undirected bipartite edge list."""
        r = np.concatenate([rows, cols]); c = np.concatenate([cols, rows]); w = np.concatenate([w, w]).astype(np.float32)
        deg = np.bincount(r, weights=w, minlength=self.N)
        norm = (w / np.sqrt(deg[r] * deg[c] + 1e-12)).astype(np.float32)
        return torch.sparse_coo_tensor(np.stack([r, c]), norm, (self.N, self.N)).coalesce()

    def relations(self, events, t_end, plain=False):
        e = events.filter((pl.col('timestamp') < t_end) & (pl.col('idx') > 1)).join(self.urow, on='uid')
        e = e.with_columns(d=pl.Series(decay(e['timestamp'].to_numpy(), t_end, 30)))
        pos = pl.col('fb').is_in([PLUS, LIKE])
        if plain:
            g = e.filter(pos).select('u', 'idx').unique()
            rels = {'pos': (g['u'].to_numpy(), self.U + g['idx'].to_numpy(), np.ones(g.height))}
        else:
            g = e.group_by('u', 'idx').agg(
                pos=(pl.col('d') * pos.cast(pl.Float32) * (1 + (pl.col('fb') == LIKE).cast(pl.Float32))).sum(),
                skip=(pl.col('d') * (pl.col('fb') == SKIP).cast(pl.Float32)).sum(),
                org=(pl.col('d') * (pos & (pl.col('is_organic') == 1)).cast(pl.Float32)).sum())
            rels = {}
            for r in ('pos', 'skip', 'org'):
                h = g.filter(pl.col(r) > 0)
                rels[r] = (h['u'].to_numpy(), self.U + h['idx'].to_numpy(), h[r].to_numpy())
        rels['art'] = (self.U + self.item_idx, self.U + self.V + self.item_artist[self.item_idx - 2],
                       np.ones(len(self.item_idx)))
        return {r: self.adj(*v) for r, v in rels.items()}


class HeteroLightGCN(torch.nn.Module):
    def __init__(self, g, rel_names, multihash=False, buckets=2 ** 15):
        super().__init__()
        self.g, self.rels, self.multihash = g, rel_names, multihash
        self.w = torch.nn.Parameter(torch.ones(len(rel_names)))
        if multihash:  # users and artists keep full tables; items share 2 hashed tables
            self.ua = torch.nn.Embedding(g.U + g.A, D)
            self.h = torch.nn.ModuleList([torch.nn.Embedding(buckets, D) for _ in range(2)])
            rng = np.random.default_rng(SEED)
            self.register_buffer('hidx', torch.tensor(rng.integers(0, buckets, (2, g.V)), dtype=torch.long))
            for e in (self.ua, *self.h):
                torch.nn.init.normal_(e.weight, std=0.1)
        else:
            self.emb = torch.nn.Embedding(g.N, D)
            torch.nn.init.normal_(self.emb.weight, std=0.1)

    def base(self):
        if not self.multihash:
            return self.emb.weight
        items = self.h[0](self.hidx[0]) + self.h[1](self.hidx[1])
        w = self.ua.weight
        return torch.cat([w[:self.g.U], items, w[self.g.U:]])

    def forward(self, A):
        h = self.base()
        out = [h]
        for _ in range(LAYERS):
            h = sum(self.w[i] * torch.sparse.mm(A[r], h) for i, r in enumerate(self.rels))
            out.append(h)
        return torch.stack(out).mean(0)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def temporal_targets(stage, g, T0):
    """New listen+ (user, item) pairs in [T0, cutoff): items the user had not listened to before T0."""
    f = stage.fit.filter(pl.col('idx') > 1).join(g.urow, on='uid')
    before = f.filter(pl.col('timestamp') < T0).select('u', 'idx').unique()
    after = f.filter((pl.col('timestamp') >= T0) & pl.col('fb').is_in([PLUS, LIKE])).select('u', 'idx').unique()
    new = after.join(before, on=['u', 'idx'], how='anti')
    return new['u'].to_numpy(), new['idx'].to_numpy()


def train(stage, g, plain=False, multihash=False):
    T0 = stage.fit_end - HIST_DAYS * DAY
    A0 = g.relations(stage.fit, T0, plain)
    model = HeteroLightGCN(g, list(A0), multihash)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    pu, pi = temporal_targets(stage, g, T0)
    pop = np.bincount(pi, minlength=g.V).astype(np.float64) ** 0.75
    pop[:2] = 0
    pop /= pop.sum()
    rng = np.random.default_rng(SEED)
    t = time.time()
    for step in range(1, STEPS + 1):
        b = rng.integers(0, len(pu), BATCH)
        neg = np.where(rng.random(BATCH) < 0.5, rng.integers(2, g.V, BATCH), rng.choice(g.V, BATCH, p=pop))
        H = model(A0)
        u, p, n = H[pu[b]], H[g.U + pi[b]], H[g.U + neg]
        loss = -F.logsigmoid((u * p).sum(1) - (u * n).sum(1)).mean() + 1e-4 * (u.pow(2).sum() + p.pow(2).sum() + n.pow(2).sum()) / BATCH
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 25 == 0:
            print({'step': step, 'bpr': round(loss.item(), 4), 'w': [round(x, 3) for x in model.w.tolist()],
                   'sec': round(time.time() - t)}, flush=True)
    with torch.no_grad():
        H = model(g.relations(stage.fit, stage.fit_end, plain))
    return H.numpy(), dict(zip(A0, model.w.tolist())), model.n_params()


def ridge_r2(Xtr, ytr, Xte, yte, lam=1.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    w = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ (ytr - ytr.mean()))
    pred = Xte @ w + ytr.mean()
    return float(1 - ((yte - pred) ** 2).sum() / ((yte - yte.mean()) ** 2).sum())


def main():
    t0 = time.time()
    stage = Stage(2000, 'val')
    df = base_features(stage, build_candidates(stage))
    users = np.random.default_rng(SEED).permutation(stage.users)  # same split as openfe.py
    part = {u: ('tr' if i < len(users) // 2 else 'va' if i < 3 * len(users) // 4 else 'te') for i, u in enumerate(users)}
    df = df.with_columns(part=pl.Series([part[u] for u in df['uid']]))
    g = Graph(stage)
    uidx = dict(zip(g.uids, range(g.U)))
    u_rows = np.array([uidx[u] for u in df['uid']])
    i_rows = g.U + df['idx'].to_numpy()

    variants = {}
    for name, kw in [('graph_typed_temporal', {}), ('graph_plain', {'plain': True}),
                     ('graph_typed_multihash', {'multihash': True})]:
        print('==', name, flush=True)
        H, w, n_params = train(stage, g, **kw)
        variants[name] = {'H': H, 'rel_weights': w, 'params': n_params}
        df = df.with_columns(pl.Series(name, (H[u_rows] * H[i_rows]).sum(1)))

    # does the embedding learn the manual aggregates? linear probe from (user, item, user*item) embeddings
    H = variants['graph_typed_temporal']['H']
    Z = np.hstack([H[u_rows], H[i_rows], H[u_rows] * H[i_rows]])
    tr, te = (df['part'] == 'tr').to_numpy(), (df['part'] == 'te').to_numpy()
    probe = {}
    for col in ['ua_cnt', 'item_cnt', 'artist_pop', 'u_events', 'u_skip', 'item_skip', 'u_org', 'item_org']:
        y = df[col].fill_null(0).to_numpy().astype(np.float64)
        if col in ('ua_cnt', 'item_cnt', 'artist_pop', 'u_events'):
            y = np.log1p(y)
        probe[col] = ridge_r2(Z[tr], y[tr], Z[te], y[te])
    print('probe R2:', {k: round(v, 3) for k, v in probe.items()}, flush=True)

    # ranking: graph alone vs boosting with/without the graph score (and with the OpenFE features)
    of = json.loads((OUT / 'openfe_report.json').read_text())['selected']
    exprs = generate(BASE)
    df = df.with_columns(**{n: exprs[n] for n in of}).fill_nan(None)
    fit_rows = (df['part'] != 'te').to_numpy()
    y = df['y'].to_numpy()
    te_df = df.filter(pl.col('part') == 'te')
    sets = {'catboost_base': BASE, 'catboost_base+graph': BASE + ['graph_typed_temporal'],
            'catboost_base+openfe': BASE + of, 'catboost_base+openfe+graph': BASE + of + ['graph_typed_temporal']}
    scores = {'als_only': te_df['als_score'].to_numpy()}
    for v in variants:
        scores[v] = te_df[v].to_numpy()
    for name, cols in sets.items():
        X = df.select(cols).fill_null(0).to_numpy()
        scores[name] = fit_cb(X[fit_rows], y[fit_rows]).predict_proba(X[~fit_rows])[:, 1]
    report = {'relation_weights': {k: v['rel_weights'] for k, v in variants.items()},
              'params': {k: v['params'] for k, v in variants.items()}, 'probe_r2': probe, 'test': {}}
    for mode in ('new', 'org_new'):
        for k in (10, 50):
            per = {n: ndcg_per_user(te_df, s, stage, mode, k) for n, s in scores.items()}
            common = sorted(set.intersection(*[set(p) for p in per.values()]))
            arr = {n: np.array([p[u] for u in common]) for n, p in per.items()}
            res = {'users': len(common), **{n: float(a.mean()) for n, a in arr.items()}}
            for n in ('catboost_base+graph', 'catboost_base+openfe+graph', 'graph_typed_temporal'):
                ref = 'catboost_base+openfe' if 'openfe' in n else ('graph_plain' if n.startswith('graph') else 'catboost_base')
                res[f'delta {n} vs {ref}'] = paired_ci(arr[n], arr[ref])
            report['test'][f'{mode}@{k}'] = res
    report['sec'] = round(time.time() - t0)
    (OUT / 'graph_report.json').write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != 'test'}, indent=1))
    print(json.dumps(report['test'], indent=1))


if __name__ == '__main__':
    main()
