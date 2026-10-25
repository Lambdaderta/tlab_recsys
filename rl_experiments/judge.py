"""Independent judge: CatBoost on point-in-time (user, item) features predicting the reward of a listen.

Different model class and features from the RM, never used for training any policy, so a policy that
games the RM should not automatically look good here. Same blind spot as the RM: it predicts the reward
*given the item is played* (logged data has no outcome for items the user never started).

uv run python judge.py --n 2000 --stage val          # train, check quality, score every saved top-10
"""
import argparse
import json

import numpy as np
import polars as pl
from catboost import CatBoostRegressor

from common import DAY, DATA, LIKE, OUT, PLUS, Stage, load_events, reward_expr
from rl_kl import auc

ARTIST = DATA.parent.parent / 'artist_item_mapping.parquet'


def features(hist, pairs, T):
    """Features of (uid, item_id) pairs from events strictly before T."""
    h = hist.filter(pl.col('timestamp') < T).with_columns(
        plus=pl.col('fb').is_in([PLUS, LIKE]).cast(pl.Float32), like=(pl.col('fb') == LIKE).cast(pl.Float32))
    art = pl.read_parquet(ARTIST).unique('item_id')
    h = h.join(art, on='item_id', how='left')
    ui = h.group_by('uid', 'item_id').agg(ui_cnt=pl.len(), ui_plus=pl.col('plus').mean(), ui_like=pl.col('like').max(),
                                          ui_age=(T - pl.col('timestamp').max()) / DAY)
    it = h.group_by('item_id').agg(i_cnt=pl.len(), i_plus=pl.col('plus').mean(), i_like=pl.col('like').mean(),
                                   i_users=pl.col('uid').n_unique(), i_age=(T - pl.col('timestamp').max()) / DAY)
    us = h.group_by('uid').agg(u_cnt=pl.len(), u_plus=pl.col('plus').mean(), u_org=pl.col('is_organic').mean(),
                               u_recent=(pl.col('timestamp') >= T - 7 * DAY).sum())
    ua = h.group_by('uid', 'artist_id').agg(ua_cnt=pl.len(), ua_plus=pl.col('plus').mean())
    x = (pairs.join(art, on='item_id', how='left')
         .join(ui, on=['uid', 'item_id'], how='left').join(it, on='item_id', how='left')
         .join(us, on='uid', how='left').join(ua, on=['uid', 'artist_id'], how='left'))
    x = x.with_columns(pl.col('ui_cnt', 'ua_cnt', 'i_cnt', 'i_users').fill_null(0),
                       ui_seen=pl.col('ui_plus').is_not_null().cast(pl.Float32))
    return x


FEATS = ['ui_cnt', 'ui_plus', 'ui_like', 'ui_age', 'ui_seen', 'i_cnt', 'i_plus', 'i_like', 'i_users', 'i_age',
         'u_cnt', 'u_plus', 'u_org', 'u_recent', 'ua_cnt', 'ua_plus']


def train_judge(stage, label_days=14, max_rows=400_000, seed=0):
    """Labels: listens in the last `label_days` of the fit data; features from before that."""
    T0 = stage.fit_end - label_days * DAY
    lab = stage.fit.filter(pl.col('timestamp') >= T0).select('uid', 'item_id', 'fb')
    lab = lab.sample(min(max_rows, lab.height), seed=seed).with_columns(y=reward_expr())
    x = features(stage.fit, lab, T0)
    users = x['uid'].unique().sample(fraction=0.1, seed=seed)
    va = x['uid'].is_in(users.implode())
    model = CatBoostRegressor(iterations=1500, learning_rate=0.08, depth=6, loss_function='RMSE',
                              random_seed=seed, verbose=0, od_type='Iter', od_wait=100, thread_count=6)
    model.fit(x.filter(~va)[FEATS].to_pandas(), x.filter(~va)['y'].to_numpy(),
              eval_set=(x.filter(va)[FEATS].to_pandas(), x.filter(va)['y'].to_numpy()))
    return model


def judge_quality(model, stage):
    """Held-out: listens in the stage window, features at fit_end. Compared to the item-mean baseline."""
    ev = load_events(stage.n)
    win = ev.filter(pl.col('timestamp').is_between(stage.fit_end + DAY, stage.fit_end + 8 * DAY, closed='left'))
    win = win.sample(min(100_000, win.height), seed=0).with_columns(y=reward_expr())
    x = features(stage.fit, win.select('uid', 'item_id', 'y'), stage.fit_end)
    pred = model.predict(x[FEATS].to_pandas())
    y = x['y'].to_numpy()
    imean = x['i_plus'].fill_null(x['i_plus'].mean()).to_numpy()
    rank = lambda v: np.argsort(np.argsort(v))
    sp = lambda p: float(np.corrcoef(rank(p), rank(y))[0, 1])
    return {'judge': {'spearman': sp(pred), 'auc_plus': auc(pred, y >= 1)},
            'item_mean': {'spearman': sp(imean), 'auc_plus': auc(imean, y >= 1)}, 'n_events': int(len(y))}


def score_lists(model, stage):
    """judge@10 = mean predicted reward of each saved top-10 list (all stage users)."""
    out = {}
    for f in sorted(OUT.glob(f'per_{stage.n}_{stage.name}_*.npz')):
        name = f.stem[len(f'per_{stage.n}_{stage.name}_'):]
        d = np.load(f)
        if 'top_items' not in d:
            continue
        top = d['top_items'][:, :10]
        pairs = pl.DataFrame({'uid': np.repeat(stage.users, top.shape[1]), 'item_id': top.ravel()})
        x = features(stage.fit, pairs, stage.fit_end)
        out[name] = float(model.predict(x[FEATS].to_pandas()).mean())
        print(name, round(out[name], 4))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--stage', default='val')
    ap.add_argument('--score_only', action='store_true')
    a = ap.parse_args()
    stage = Stage(a.n, a.stage)
    path = OUT.parent / 'cache' / f'judge_{a.n}_{a.stage}.cbm'
    if a.score_only and path.exists():
        model = CatBoostRegressor()
        model.load_model(str(path))
    else:
        model = train_judge(stage)
        model.save_model(str(path))
        q = judge_quality(model, stage)
        print(json.dumps(q, indent=1))
        with open(OUT / f'judge_quality_{a.n}_{a.stage}.json', 'w') as f:
            json.dump(q, f, indent=1)
    with open(OUT / f'judge_scores_{a.n}_{a.stage}.json', 'w') as f:
        json.dump(score_lists(model, stage), f, indent=1)
