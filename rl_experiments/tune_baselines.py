"""Equal-budget tuning of the basic algorithms on the 10K val week, then one test run per model.

Selection metric: organic-new NDCG@10 (all-new is logged for every config too).
uv run python tune_baselines.py [--models ease,als,...]  -> out/tuning_baselines.json, res_10000_test_<model>.json
"""
import argparse
import itertools
import json
import time

from baselines import run_als, run_covisit, run_itemknn, run_pop, run_recent
from common import OUT, Stage

N = 10_000
GRIDS = {  # small grids to fit the 2-hour budget; second round added after the first results (EASE dropped)
    'decaypop': [dict(hl=h) for h in [1, 7, 30, 0.5, 2]],
    'itemknn': [dict(hl=h) for h in [14, 60, 3, 7]],
    'covisit': [dict(window=3, last_n=20, alpha=0.5), dict(window=10, last_n=100, alpha=0.5),
                dict(window=10, last_n=100, alpha=0.3), dict(window=20, last_n=200, alpha=0.5)],
    'als': [dict(factors=128, reg=0.01, hl=30), dict(factors=256, reg=0.1, hl=120), dict(factors=256, reg=0.01, hl=60),
            dict(factors=64, reg=0.01, hl=30), dict(factors=192, reg=0.01, hl=30), dict(factors=128, reg=0.001, hl=14)],
}


def run(model, stage, cfg, name):
    if model == 'decaypop':
        return run_pop(stage, hls=(cfg['hl'],), names=(name,))[name]
    return {'itemknn': run_itemknn, 'covisit': run_covisit, 'als': run_als}[model](stage, **cfg, name=name)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default=','.join(GRIDS))
    ap.add_argument('--test_only', action='store_true', help='use the saved best configs; untuned models get GRIDS[m][0]')
    a = ap.parse_args()
    path = OUT / 'tuning_baselines.json'
    log = json.loads(path.read_text()) if path.exists() else {}
    if a.test_only:
        for m in a.models.split(','):
            log.setdefault(m, {'grid': [], 'best': GRIDS[m][0], 'untuned': True})
        path.write_text(json.dumps(log, indent=1))
    val = None if a.test_only else Stage(N, 'val')
    for model in ([] if a.test_only else a.models.split(',')):
        rows = log.get(model, {}).get('grid', [])  # configs already run are reused, not rerun
        done = [r['cfg'] for r in rows]
        for i, cfg in enumerate(GRIDS[model]):
            if cfg in done:
                continue
            t = time.time()
            r = run(model, val, cfg, f'tune_{model}_{i}')
            rows.append({'cfg': cfg, 'org_new_ndcg10': r['org_new']['ndcg@10'], 'new_ndcg10': r['new']['ndcg@10'],
                         'sec': round(time.time() - t)})
            print(model, rows[-1], flush=True)
        best = max(rows, key=lambda x: x['org_new_ndcg10'])
        log[model] = {'grid': rows, 'best': best['cfg']}
        path.write_text(json.dumps(log, indent=1))
    test = Stage(N, 'test')
    for model in a.models.split(','):
        run(model, test, log[model]['best'], model)
    run_recent(test)  # untuned reference for the 'all' targets
