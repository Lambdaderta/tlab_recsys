"""Main tables of the re-staged run (10K users, test week): tuned base algorithms, GPT-2 SASRec, push-based RL.

uv run python summarize_v3.py   -> out/tables_v3.md
"""
import json

import numpy as np

from common import OUT, paired_ci

N, STAGE, REF = 10_000, 'test', 'sasrec_s0'
ROWS = [
    ('Heuristics', 'decaypop', 'DecayPop (tuned half-life)'),
    ('Heuristics', 'recent', 'Recently played (all targets only)'),
    ('Co-occurrence / CF', 'itemknn', 'ItemKNN (tuned)'),
    ('Co-occurrence / CF', 'covisit', 'Co-visitation (tuned)'),
    ('Co-occurrence / CF', 'als', 'ALS (tuned)'),
    ('GPT-2', 'sasrec_s0', 'GPT-2 SASRec, seed 0'),
    ('GPT-2', 'sasrec_s1', 'GPT-2 SASRec, seed 1'),
    ('Push-based RL', 'push_beta', 'β̂: imitate pushes (logging policy)'),
    ('Push-based RL', 'push_awr', 'AWR on pushes + KL'),
    ('Push-based RL', 'push_opc', 'Off-policy REINFORCE (π/β̂, top-K) + KL'),
    ('Push-based RL', 'push_tilt_b1', 'Exact tilted π_ref·exp(r̂/1)'),
    ('Push-based RL', 'push_tilt_b0.3', 'Exact tilted π_ref·exp(r̂/0.3)'),
]


def load(name):
    p = OUT / f'res_{N}_{STAGE}_{name}.json'
    return json.loads(p.read_text()) if p.exists() else None


def ci(name, mode, metric='ndcg@10'):
    a, b = OUT / f'per_{N}_{STAGE}_{name}_{mode}.npz', OUT / f'per_{N}_{STAGE}_{REF}_{mode}.npz'
    if name == REF or not (a.exists() and b.exists()):
        return '—'
    d, lo, hi = paired_ci(np.load(a)[metric], np.load(b)[metric])
    return f'{d:+.4f} [{lo:+.4f}, {hi:+.4f}]' + ('*' if lo > 0 or hi < 0 else '')


def main():
    ref = load(REF)
    out = [f'## {N // 1000}K users, test week',
           '',
           f'Users with targets: {ref["org_new"]["users"]} (organic new), {ref["new"]["users"]} (new), {ref["all"]["users"]} (all). '
           f'Δ = paired bootstrap 95% CI vs `{REF}`, * = excludes 0.',
           '',
           '| Group | Model | NDCG@10 organic new | Δ organic new | NDCG@10 new | Δ new | NDCG@100 new | Recall@100 new | Cov@10 new | NDCG@10 all |',
           '|---|---|---|---|---|---|---|---|---|---|']
    for g, name, label in ROWS:
        r = load(name)
        if r is None:
            continue
        o, n, a = r['org_new'], r['new'], r['all']
        out.append(f'| {g} | {label} | {o["ndcg@10"]:.4f} | {ci(name, "org_new")} | {n["ndcg@10"]:.4f} | {ci(name, "new")} | '
                   f'{n["ndcg@100"]:.4f} | {n["recall@100"]:.3f} | {n["coverage@10"]:.4f} | {a["ndcg@10"]:.4f} |')
    tun = OUT / 'tuning_baselines.json'
    if tun.exists():
        out += ['', '### Tuning (10K val, selection on organic-new NDCG@10)', '', '| Model | Config | organic new | new |', '|---|---|---|---|']
        for m, v in json.loads(tun.read_text()).items():
            for row in v['grid']:
                mark = ' **(chosen)**' if row['cfg'] == v['best'] else ''
                out.append(f'| {m} | `{row["cfg"]}`{mark} | {row["org_new_ndcg10"]:.4f} | {row["new_ndcg10"]:.4f} |')
    trainlog = OUT / f'trainlog_{N}_val_sasrec.json'
    if trainlog.exists():
        ev = [r for r in json.loads(trainlog.read_text()) if 'org_new_ndcg10' in r]
        out += ['', '### GPT-2 SASRec step selection (10K val)', '', '| Steps | organic new | new |', '|---|---|---|']
        out += [f'| {r["step"]} | {r["org_new_ndcg10"]:.4f} | {r["new_ndcg10"]:.4f} |' for r in ev]
    rl = OUT / f'rl_push_{N}_{STAGE}.json'
    if rl.exists():
        out += ['', '### Off-policy value on held-out test-week pushes (SNIPS, weights clipped at 10)', '',
                '| Policy | SNIPS reward / push | Logged mean | Effective sample share |', '|---|---|---|---|']
        for m, v in json.loads(rl.read_text()).items():
            o = v['ope']
            out.append(f'| {m} | {o["snips"]:.4f} | {o["logged_mean_r"]:.4f} | {o["ess_share"]:.2f} |')
    (OUT / 'tables_v3.md').write_text('\n'.join(out) + '\n')
    print('\n'.join(out))


if __name__ == '__main__':
    main()
