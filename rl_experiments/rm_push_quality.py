"""Held-out quality of the push reward model r_hat(s, a) (rl_push.py) on test-week pushes,
against item-mean, user-mean and global-mean baselines fitted on the training pushes.

uv run python rm_push_quality.py --n 10000 --stage test   -> out/rm_push_quality_<n>_<stage>.json
"""
import argparse
import json

import numpy as np
import polars as pl
import torch

from common import CACHE, OUT, STAGES, Stage, load_events
from rl_kl import auc, within_user_auc
from rl_push import PushData, load_gpt2


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=10_000)
    ap.add_argument('--stage', default='test')
    ap.add_argument('--max_events', type=int, default=50_000)
    a = ap.parse_args()
    stage = Stage(a.n, a.stage)
    cfg = STAGES[a.stage]
    train = PushData(stage.fit, horizon_end=cfg['fit_end'])
    full = stage.vocab.encode(load_events(a.n).filter(pl.col('timestamp') < cfg['end']))
    test = PushData(full, horizon_end=cfg['end'], start=cfg['start'])
    rm = load_gpt2(CACHE / f'ckpt_{a.n}_{a.stage}_push_rm.pt', stage.vocab.size, q_head=True)

    rng = np.random.default_rng(0)
    pos = rng.choice(test.pos, min(a.max_events, len(test.pos)), replace=False)
    pred, y, items, users = [], [], [], []
    for s in range(0, len(pos), 512):
        state, act, r = test.batch(rng, None, pos[s:s + 512])
        pred.append(rm.qvalues(rm.enc(*state)[:, -1], act).cpu().numpy())
        y.append(r.cpu().numpy()); items.append(act.cpu().numpy())
        users.append(test.user_of[pos[s:s + 512]])
    pred, y, items, users = map(np.concatenate, (pred, y, items, users))

    tr_r, tr_i = train.r[train.pos], train.seqs.idx[train.pos]
    tr_u = train.seqs.uids[train.user_of[train.pos]]
    gm = float(tr_r.mean())
    item_mean = pl.DataFrame({'i': tr_i, 'r': tr_r}).group_by('i').agg(pl.col('r').mean())
    user_mean = pl.DataFrame({'u': tr_u, 'r': tr_r}).group_by('u').agg(pl.col('r').mean())
    im, um = dict(item_mean.iter_rows()), dict(user_mean.iter_rows())
    uids = test.seqs.uids[users]
    preds = {'reward_model': pred,
             'item_mean': np.array([im.get(i, gm) for i in items]),
             'user_mean': np.array([um.get(u, gm) for u in uids])}
    pos_label = y > 0
    rank = lambda v: np.argsort(np.argsort(v))
    out = {'n_events': int(len(y)), 'positive_share': float(pos_label.mean()), 'mean_reward': float(y.mean())}
    for k, p in preds.items():
        out[k] = {'auc_positive': auc(p, pos_label), 'spearman': float(np.corrcoef(rank(p), rank(y))[0, 1]),
                  'mae': float(np.abs(p - y).mean()), 'within_user_auc': within_user_auc(p, pos_label, uids, min_each=3)}
    (OUT / f'rm_push_quality_{a.n}_{a.stage}.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
