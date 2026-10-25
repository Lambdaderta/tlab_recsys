"""Shared pieces: fixed split, sampled events with feedback labels, vocab, metrics."""
from pathlib import Path
import json

import numpy as np
import polars as pl
import scipy.sparse as sp

ROOT = Path(__file__).parent
DATA = ROOT.parent / 'data' / 'yambda' / 'flat' / '50m'
CACHE = ROOT / 'cache'
OUT = ROOT / 'out'
CACHE.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

DAY = 86_400
T_END = 26_000_005  # max timestamp is 26,000,000; +5 s rounding step
# train | gap 1d | val 7d | gap 1d | test 7d
TEST_START = T_END - 7 * DAY
VAL_END = TEST_START - DAY
VAL_START = VAL_END - 7 * DAY
TRAIN_END = VAL_START - DAY

STAGES = {
    # fit_end: history / training data is everything before it; the window is what we predict
    'val': dict(fit_end=TRAIN_END, start=VAL_START, end=VAL_END),
    'test': dict(fit_end=VAL_END, start=TEST_START, end=T_END),
}

# feedback classes of a listen event and their reward
SKIP, PLUS, LIKE, DISLIKE = 0, 1, 2, 3
REWARD = np.array([0.0, 1.0, 2.0, -1.0], dtype=np.float32)
FEEDBACK_WINDOW = 1800


def reward_expr(col='fb'):
    return pl.col(col).replace_strict(dict(enumerate(REWARD.tolist())), return_dtype=pl.Float32)


_ = None  # (keeps the constants below grouped)  # a like/dislike within 30 min after the listen start belongs to it
MIN_ITEM_LISTENS = 5
K = 10
K_CAND = 100


def sample_uids(n, seed=42):
    """Nested samples: the first n users of one fixed permutation, so 2K is inside 5K inside 10K."""
    uids = (pl.scan_parquet(DATA / 'multi_event.parquet').select(pl.col('uid').unique())
            .collect()['uid'].sort().to_numpy())
    return np.random.default_rng(seed).permutation(uids)[:n]


def load_events(n):
    """Listens of n sampled users with a feedback class (skip / listen+ / like / dislike)."""
    path = CACHE / f'events_{n}.parquet'
    if path.exists():
        return pl.read_parquet(path)
    uids = pl.Series('uid', sample_uids(n)).implode()
    cols = ['uid', 'timestamp', 'item_id']
    listens = (pl.scan_parquet(DATA / 'listens.parquet').filter(pl.col('uid').is_in(uids))
               .select(*cols, 'is_organic', 'played_ratio_pct').sort('timestamp').collect())

    def attach(ev, name):
        marks = (pl.scan_parquet(DATA / f'{name}.parquet').filter(pl.col('uid').is_in(uids))
                 .select(*cols).with_columns(pl.lit(True).alias(name)).sort('timestamp').collect())
        return ev.join_asof(marks, on='timestamp', by=['uid', 'item_id'], strategy='forward',
                            tolerance=FEEDBACK_WINDOW).with_columns(pl.col(name).fill_null(False))

    ev = attach(attach(listens, 'likes'), 'dislikes')
    ev = ev.with_columns(
        fb=pl.when(pl.col('dislikes')).then(DISLIKE)
        .when(pl.col('likes')).then(LIKE)
        .when(pl.col('played_ratio_pct') >= 50).then(PLUS)
        .otherwise(SKIP).cast(pl.UInt8)
    ).select('uid', 'timestamp', 'item_id', 'is_organic', 'fb').sort('uid', 'timestamp')
    ev.write_parquet(path)
    return ev


class Vocab:
    """Items with >= MIN_ITEM_LISTENS listens in the fit data. Index 0 is padding, 1 is OOV."""

    def __init__(self, fit):
        # tie-break by item_id: group_by order is not stable, and a checkpoint is only valid
        # in another process if the idx <-> item mapping is identical
        items = (fit.group_by('item_id').len().filter(pl.col('len') >= MIN_ITEM_LISTENS)
                 .sort(['len', 'item_id'], descending=[True, False]))
        self.items = items['item_id'].to_numpy()
        self.counts = items['len'].to_numpy()
        self.size = len(self.items) + 2
        self.map = pl.DataFrame({'item_id': self.items, 'idx': np.arange(2, self.size, dtype=np.int32)})

    def encode(self, df):
        return (df.join(self.map, on='item_id', how='left')
                .with_columns(pl.col('idx').fill_null(1)))


class Stage:
    """Everything a model needs for one stage: fit events, eval users, their targets."""

    def __init__(self, n, stage):
        cfg = STAGES[stage]
        ev = load_events(n)
        self.name, self.n = stage, n
        self.fit_end = cfg['fit_end']
        self.fit = ev.filter(pl.col('timestamp') < cfg['fit_end'])
        win = ev.filter(pl.col('timestamp').is_between(cfg['start'], cfg['end'], closed='left'))
        self.vocab = Vocab(self.fit)
        self.fit = self.vocab.encode(self.fit)

        hist_users = self.fit['uid'].unique()
        # per (user, item) reward in the window: dislike wins, then like, then listen+, else skip
        win_r = (win.group_by('uid', 'item_id')
                 .agg(r=pl.when((pl.col('fb') == DISLIKE).any()).then(-1.0)
                      .when((pl.col('fb') == LIKE).any()).then(2.0)
                      .when((pl.col('fb') == PLUS).any()).then(1.0).otherwise(0.0),
                      org=(pl.col('fb').is_in([PLUS, LIKE]) & (pl.col('is_organic') == 1)).any()))
        seen = self.fit.select('uid', 'item_id').unique().with_columns(seen=pl.lit(True))
        win_r = win_r.join(seen, on=['uid', 'item_id'], how='left').with_columns(pl.col('seen').fill_null(False))
        pos = win_r.filter(pl.col('r') >= 1.0)
        self.n_cold = pos.filter(~pl.col('uid').is_in(hist_users.implode()))['uid'].n_unique()
        pos = pos.filter(pl.col('uid').is_in(hist_users.implode()))

        self.users = np.sort(pos['uid'].unique().to_numpy())
        self.uid_row = {u: i for i, u in enumerate(self.users)}
        self.targets = {'all': self._sets(pos), 'new': self._sets(pos.filter(~pl.col('seen'))),
                        # organic new: found by the user, not pushed (is_organic = 1); evaluated on 'new' lists
                        'org_new': self._sets(pos.filter(~pl.col('seen') & pl.col('org')))}
        self.rewards = {(u, i): r for u, i, r in win_r.select('uid', 'item_id', 'r').iter_rows()}
        self.item_pop = dict(zip(self.vocab.items, self.vocab.counts))
        self.n_fit_events = self.fit.height

    def _sets(self, df):
        groups = df.group_by('uid').agg('item_id')
        d = {u: set(items) for u, items in groups.iter_rows()}
        return [d.get(u, set()) for u in self.users]

    def seen_matrix(self):
        """CSR [eval users x vocab] of items each eval user had in the fit data (for 'new' filtering)."""
        s = (self.fit.filter(pl.col('uid').is_in(pl.Series(self.users).implode()) & (pl.col('idx') > 1))
             .select('uid', 'idx').unique())
        rows = np.array([self.uid_row[u] for u in s['uid'].to_numpy()])
        return sp.csr_matrix((np.ones(len(rows), np.float32), (rows, s['idx'].to_numpy())),
                             shape=(len(self.users), self.vocab.size))


def topk(scores, k, exclude=None):
    """Row-wise top-k indices of a dense score block; `exclude` is a CSR block of the same shape."""
    scores = scores.copy()
    scores[:, :2] = -np.inf  # padding and OOV are never recommended
    if exclude is not None:
        r, c = exclude.nonzero()
        scores[r, c] = -np.inf
    part = np.argpartition(-scores, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, part, 1), axis=1)
    return np.take_along_axis(part, order, 1)


def recommend(stage, score_fn, batch=256):
    """score_fn(rows) -> dense [len(rows), vocab] scores. Returns top-K_CAND for both target modes."""
    seen = stage.seen_matrix()
    out = {'all': [], 'new': []}
    for start in range(0, len(stage.users), batch):
        rows = np.arange(start, min(start + batch, len(stage.users)))
        s = np.asarray(score_fn(rows), dtype=np.float32)
        out['all'].append(topk(s, K_CAND))
        out['new'].append(topk(s, K_CAND, exclude=seen[rows]))
    return {m: np.concatenate(v) for m, v in out.items()}


def decay(ts, fit_end, half_life_days):
    return np.power(0.5, (fit_end - ts) / (half_life_days * DAY)).astype(np.float32)


def user_item_matrix(stage, half_life_days=30, weights=(0.0, 1.0, 2.0, 0.0), users=None):
    """Sparse [users x vocab], feedback-weighted and time-decayed. Default rows: all fit users."""
    f = stage.fit.filter(pl.col('idx') > 1)
    uids = np.sort(f['uid'].unique().to_numpy()) if users is None else users
    f = f.filter(pl.col('uid').is_in(pl.Series(uids).implode()))
    row_of = pl.DataFrame({'uid': uids, 'row': np.arange(len(uids), dtype=np.int32)})
    f = f.join(row_of, on='uid')
    w = np.asarray(weights, np.float32)[f['fb'].to_numpy()] * decay(f['timestamp'].to_numpy(), stage.fit_end, half_life_days)
    m = sp.csr_matrix((w, (f['row'].to_numpy(), f['idx'].to_numpy())), shape=(len(uids), stage.vocab.size))
    m.eliminate_zeros()
    return m, uids


def evaluate(stage, recs_idx, mode):
    """Per-user metrics for top-K_CAND vocab indices. Targets outside the vocab count as misses."""
    items = np.concatenate([[0, 0], stage.vocab.items])
    rec_items = items[recs_idx]
    disc = 1.0 / np.log2(np.arange(2, K_CAND + 2))
    per = {m: [] for m in ['recall@10', 'ndcg@10', 'ndcg@50', 'ndcg@100', 'recall@100', 'reward@10']}
    rows = []
    for row, (u, tgt) in enumerate(zip(stage.users, stage.targets[mode])):
        if not tgt:
            continue
        rows.append(row)
        top = rec_items[row]
        hit = np.array([i in tgt for i in top])
        per['recall@10'].append(hit[:K].sum() / min(len(tgt), K))
        per['recall@100'].append(hit.sum() / min(len(tgt), K_CAND))
        for k in (K, 50, K_CAND):  # ideal DCG over min(|targets|, k) hits
            per[f'ndcg@{k}'].append((hit[:k] * disc[:k]).sum() / disc[:min(len(tgt), k)].sum())
        per['reward@10'].append(sum(stage.rewards.get((u, i), 0.0) for i in top[:K]))
    per = {m: np.array(v) for m, v in per.items()}
    top10 = recs_idx[rows, :K]
    pop = np.array([stage.item_pop.get(i, 1) for i in stage.vocab.items])
    p = np.concatenate([[1, 1], pop]) / pop.sum()
    summary = {m: float(v.mean()) for m, v in per.items()}
    summary['coverage@10'] = float(len(np.unique(top10)) / (stage.vocab.size - 2))
    summary['novelty@10'] = float(-np.log2(p[top10]).mean())
    summary['users'] = len(rows)
    return summary, per


def paired_ci(a, b, n_boot=2000, seed=0):
    """95% bootstrap CI of mean(a - b) over the same users."""
    d = a - b
    idx = np.random.default_rng(seed).integers(0, len(d), (n_boot, len(d)))
    m = d[idx].mean(1)
    return float(d.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def save_result(name, stage, recs_by_mode, extra=None):
    """Evaluate both target modes, store summary json + per-user arrays for later paired CIs."""
    res = {'model': name, 'stage': stage.name, 'n_users_sample': stage.n,
           'n_cold_users': stage.n_cold, **(extra or {})}
    if 'new' in recs_by_mode:
        recs_by_mode = {**recs_by_mode, 'org_new': recs_by_mode['new']}
    for mode, recs in recs_by_mode.items():
        summary, per = evaluate(stage, recs, mode)
        res[mode] = summary
        top_items = np.concatenate([[0, 0], stage.vocab.items])[recs]  # raw item ids (top-100), not vocab idx
        np.savez(OUT / f'per_{stage.n}_{stage.name}_{name}_{mode}.npz', **per, top_items=top_items)
    with open(OUT / f'res_{stage.n}_{stage.name}_{name}.json', 'w') as f:
        json.dump(res, f, indent=1)
    print(name, stage.name, {m: {k: round(v, 4) for k, v in res[m].items()} for m in recs_by_mode})
    return res
