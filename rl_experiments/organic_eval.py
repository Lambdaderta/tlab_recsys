"""Post-hoc evaluation on ORGANIC new-item targets, from the saved top-100 lists (no retraining).

Target (u, i): in the window the user had a listen+ / like of i that was organic (is_organic = 1, found by
the user, not pushed by the recommender), and i is not in the user's fit history. Lists are the 'new' mode
lists (seen items filtered). Organic listens depend less on the logging recommender, so this target is
closer to "what the user wants" than "what the old system showed".

uv run python organic_eval.py --n 2000 --stage test
"""
import argparse
import json

import numpy as np
import polars as pl

from common import K, K_CAND, LIKE, OUT, PLUS, STAGES, Stage, load_events, paired_ci


def organic_targets(stage):
    cfg = STAGES[stage.name]
    win = load_events(stage.n).filter(pl.col('timestamp').is_between(cfg['start'], cfg['end'], closed='left'))
    seen = stage.fit.select('uid', 'item_id').unique()
    org = (win.filter(pl.col('fb').is_in([PLUS, LIKE]) & (pl.col('is_organic') == 1))
           .select('uid', 'item_id').unique()
           .join(seen, on=['uid', 'item_id'], how='anti'))
    d = {u: set(items) for u, items in org.group_by('uid').agg('item_id').iter_rows()}
    return [d.get(u, set()) for u in stage.users]


def evaluate_lists(stage, targets, tops):
    disc = 1.0 / np.log2(np.arange(2, K_CAND + 2))
    per = {'ndcg@10': [], 'ndcg@50': [], 'ndcg@100': [], 'recall@10': [], 'recall@100': [], 'reward@10': []}
    for u, tgt, top in zip(stage.users, targets, tops):
        if not tgt:
            continue
        hit = np.array([i in tgt for i in top])
        for k in (K, 50, K_CAND):
            per[f'ndcg@{k}'].append((hit[:k] * disc[:k]).sum() / disc[:min(len(tgt), k)].sum())
        per['recall@10'].append(hit[:K].sum() / min(len(tgt), K))
        per['recall@100'].append(hit.sum() / min(len(tgt), K_CAND))
        per['reward@10'].append(sum(stage.rewards.get((u, i), 0.0) for i in top[:K]))
    return {m: np.array(v) for m, v in per.items()}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--stage', default='test')
    ap.add_argument('--ref', default='sasrec')
    a = ap.parse_args()
    stage = Stage(a.n, a.stage)
    targets = organic_targets(stage)
    n_users = sum(bool(t) for t in targets)
    print(f'users with organic new targets: {n_users} / {len(stage.users)}; '
          f'mean targets {np.mean([len(t) for t in targets if t]):.1f}')
    pre = f'per_{a.n}_{a.stage}_'
    per_model = {}
    for f in sorted(OUT.glob(f'{pre}*_new.npz')):
        name = f.stem[len(pre):-len('_new')]
        d = np.load(f)
        if 'top_items' in d:
            per_model[name] = evaluate_lists(stage, targets, d['top_items'])
    out = {'n_users': n_users, 'models': {}}
    for name, per in per_model.items():
        row = {m: float(v.mean()) for m, v in per.items()}
        if a.ref in per_model and name != a.ref:
            row['delta_ndcg_vs_ref'] = paired_ci(per['ndcg@10'], per_model[a.ref]['ndcg@10'])
            row['delta_ndcg100_vs_ref'] = paired_ci(per['ndcg@100'], per_model[a.ref]['ndcg@100'])
        out['models'][name] = row
        print(name, {k: (round(v, 4) if isinstance(v, float) else [round(x, 4) for x in v]) for k, v in row.items()})
    with open(OUT / f'organic_{a.n}_{a.stage}.json', 'w') as fh:
        json.dump(out, fh, indent=1)
