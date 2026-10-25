"""Collect out/res_*.json (+ judge, RM, bandit) into markdown tables and two figures.

uv run python make_report.py   -> out/tables.md, figures/frontier_*.png, figures/scaling.png
"""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import OUT, paired_ci

FIG = Path(__file__).parent / 'figures'
FIG.mkdir(exist_ok=True)
BLUE, ORANGE, AQUA, YELLOW, GRAY = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#8a8985'
INK, MUTED = '#0b0b0b', '#52514e'

GROUPS = [
    ('Popularity / heuristics', ['mostpop', 'decaypop', 'recent']),
    ('Co-occurrence / CF', ['itemknn', 'covisit', 'ease', 'als']),
    ('Neural, supervised', ['twotower', 'sasrec']),
    ('RL A: offline Q-learning', ['sqn', 'sa2c', 'sqn_qhead', 'sa2c_qhead']),
    ('RL B: KL-regularised', None),  # filled with tilt_* / rloo_*
]


def load(n, stage):
    res = {}
    for f in OUT.glob(f'res_{n}_{stage}_*.json'):
        r = json.loads(f.read_text())
        res[r['model']] = r
    judge = OUT / f'judge_scores_{n}_{stage}.json'
    js = json.loads(judge.read_text()) if judge.exists() else {}
    return res, js


def ci_vs(n, stage, a, b, mode, metric='ndcg@10'):
    fa, fb = OUT / f'per_{n}_{stage}_{a}_{mode}.npz', OUT / f'per_{n}_{stage}_{b}_{mode}.npz'
    if not (fa.exists() and fb.exists()):
        return ''
    d, lo, hi = paired_ci(np.load(fa)[metric], np.load(fb)[metric])
    star = '*' if lo > 0 or hi < 0 else ''
    return f'{d:+.4f} [{lo:+.4f}, {hi:+.4f}]{star}'


def table(n, stage, ref='sasrec'):
    res, js = load(n, stage)
    if not res:
        return ''
    rl_b = sorted([m for m in res if m.startswith(('tilt_', 'rloo_'))] + [m for m in res if m.endswith('_ref')])
    lines = [f'### {n} users, {stage} stage',
             '',
             f'Eval users: {next(iter(res.values()))["all"]["users"]} (all targets), '
             f'{next(iter(res.values()))["new"]["users"]} (new-item targets); cold users skipped: '
             f'{next(iter(res.values()))["n_cold_users"]}. Δ = paired bootstrap 95% CI of NDCG@10 vs `{ref}`, * = excludes 0.',
             '',
             '| Group | Model | NDCG@10 new | NDCG@50 new | NDCG@100 new | Recall@100 new | Reward@10 new | Cov@10 new '
             f'| Novelty new | Judge@10 new | NDCG@10 all | NDCG@100 all | Reward@10 all | Δ NDCG@10 new vs {ref} '
             f'| Δ NDCG@100 new vs {ref} |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for g, models in GROUPS:
        for m in (models if models is not None else rl_b):
            if m not in res:
                continue
            a, nw = res[m]['all'], res[m]['new']
            j = js.get(f'{m}_new')
            lines.append(f'| {g} | `{m}` | {nw["ndcg@10"]:.4f} | {nw["ndcg@50"]:.4f} | {nw["ndcg@100"]:.4f} | '
                         f'{nw["recall@100"]:.4f} | {nw["reward@10"]:.3f} | '
                         f'{nw["coverage@10"]:.4f} | {nw["novelty@10"]:.1f} | {"" if j is None else f"{j:.3f}"} | '
                         f'{a["ndcg@10"]:.4f} | {a["ndcg@100"]:.4f} | {a["reward@10"]:.3f} | '
                         f'{ci_vs(n, stage, m, ref, "new") if m != ref else "—"} | '
                         f'{ci_vs(n, stage, m, ref, "new", "ndcg@100") if m != ref else "—"} |')
    return '\n'.join(lines) + '\n'


def frontier(n, stage):
    res, _ = load(n, stage)
    tilt = sorted([r for m, r in res.items() if m.startswith('tilt_')], key=lambda r: r['KL'])
    rloo = [r for m, r in res.items() if m.startswith('rloo_')]
    ref = res.get('sasrec_ref')
    if not tilt:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    ax.plot([r['KL'] for r in tilt], [r['E_r'] for r in tilt], color=BLUE, lw=2, marker='o', ms=6,
            label='exact tilted policy (no training)')
    for r in tilt:
        ax.annotate(f'β={r["beta"]:g}', (r['KL'], r['E_r']), textcoords='offset points', xytext=(6, -10),
                    fontsize=8, color=MUTED)
    if rloo:
        ax.scatter([r['KL'] for r in rloo], [r['E_r'] for r in rloo], color=ORANGE, s=50, zorder=3,
                   edgecolor='white', linewidth=1.5, label='RLOO (trained)')
        for r in rloo:
            ax.annotate(f'β={r["beta"]:g}', (r['KL'], r['E_r']), textcoords='offset points', xytext=(6, 4),
                        fontsize=8, color=MUTED)
    if ref:
        ax.scatter([0], [ref['E_r']], color=GRAY, s=50, zorder=3, label='SASRec reference')
    ax.set_xscale('symlog', linthresh=0.01)
    ax.set_xlabel('KL(π ‖ π_ref), nats, eval-user states', color=INK)
    ax.set_ylabel('E_π[r̂] under the RM', color=INK)
    ax.set_title(f'Reward-KL frontier, {n} users, {stage}', color=INK, fontsize=11, loc='left')
    ax.grid(alpha=0.25)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / f'frontier_{n}_{stage}.png')
    plt.close(fig)


def scaling(stage='test', models=('decaypop', 'als', 'ease', 'sasrec', 'sqn', 'sa2c'), metric='ndcg@10'):
    ns = sorted({int(f.name.split('_')[1]) for f in OUT.glob(f'res_*_{stage}_*.json')})
    if len(ns) < 2:
        return
    colors = [BLUE, ORANGE, AQUA, YELLOW, '#e87ba4', '#6250d6']
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    for m, c in zip(models, colors):
        pts = [(n, load(n, stage)[0][m]['new'][metric]) for n in ns if m in load(n, stage)[0]]
        if pts:
            ax.plot(*zip(*pts), color=c, lw=2, marker='o', ms=6, label=m)
            ax.annotate(m, pts[-1], textcoords='offset points', xytext=(6, 0), fontsize=8, color=MUTED, va='center')
    ax.set_xscale('log')
    ax.set_xticks(ns, [f'{n // 1000}K' for n in ns])
    ax.set_xlabel('sampled users (nested samples)', color=INK)
    ax.set_ylabel(f'{metric}, new-item targets', color=INK)
    ax.set_title(f'How results scale with data, {stage} stage', color=INK, fontsize=11, loc='left')
    ax.grid(alpha=0.25)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / f'scaling_{stage}.png')
    plt.close(fig)


def extras():
    out = []
    for f in sorted(OUT.glob('rm_quality_*.json')):
        d = json.loads(f.read_text())
        out += [f'#### RM quality, {f.stem[11:]} (held-out window transitions, n = {d["n_events"]})', '',
                '| Predictor | Spearman | MAE | AUC listen+ | Within-user AUC |', '|---|---|---|---|---|']
        out += [f'| {k} | {v["spearman"]:.3f} | {v["mae"]:.3f} | {v["auc_plus"]:.3f} | {v.get("within_user_auc", float("nan")):.3f} |'
                for k, v in d.items() if isinstance(v, dict)]
        out.append('')
    for f in sorted(OUT.glob('judge_quality_*.json')):
        d = json.loads(f.read_text())
        out += [f'#### Judge quality, {f.stem[14:]} (n = {d["n_events"]})', '', '| Predictor | Spearman | AUC listen+ |', '|---|---|---|']
        out += [f'| {k} | {v["spearman"]:.3f} | {v["auc_plus"]:.3f} |' for k, v in d.items() if isinstance(v, dict)]
        out.append('')
    for f in sorted(OUT.glob('organic_*.json')):
        d = json.loads(f.read_text())
        out += [f'#### Organic new-item targets, {f.stem[8:]} (users with targets: {d["n_users"]})', '',
                '| Model | NDCG@10 | NDCG@50 | NDCG@100 | Recall@10 | Recall@100 | Reward@10 | Δ NDCG@10 vs sasrec [95% CI] '
                '| Δ NDCG@100 vs sasrec |', '|---|---|---|---|---|---|---|---|---|']
        rows = sorted(d['models'].items(), key=lambda kv: -kv[1]['ndcg@10'])
        for m, r in rows:
            fmt = lambda ci: '—' if ci is None else f'{ci[0]:+.4f} [{ci[1]:+.4f}, {ci[2]:+.4f}]' + ('*' if ci[1] > 0 or ci[2] < 0 else '')
            out.append(f'| `{m}` | {r["ndcg@10"]:.4f} | {r["ndcg@50"]:.4f} | {r["ndcg@100"]:.4f} | {r["recall@10"]:.4f} | '
                       f'{r["recall@100"]:.4f} | {r["reward@10"]:.3f} | {fmt(r.get("delta_ndcg_vs_ref"))} | '
                       f'{fmt(r.get("delta_ndcg100_vs_ref"))} |')
        out.append('')
    for f in sorted(OUT.glob('bandit_*.json')):
        d = json.loads(f.read_text())
        out += [f'#### Bandit simulator, {f.stem[7:]}', '', '| Policy | Mean true reward / rec | ± sd over seeds | Cum. regret | Items covered |',
                '|---|---|---|---|---|']
        out += [f'| {r["policy"]} | {r["mean_reward"]:.4f} | {r["mean_reward_sd"]:.4f} | {r["cum_regret"]:.3f} | {r["coverage_items"]:.0f} |' for r in d]
        out.append('')
    return '\n'.join(out)


if __name__ == '__main__':
    parts = []
    for stage in ['test', 'val']:
        for n in [2000, 5000, 10000]:
            t = table(n, stage)
            if t:
                parts.append(t)
                frontier(n, stage)
    scaling('test')
    parts.append(extras())
    (OUT / 'tables.md').write_text('\n'.join(parts))
    print((OUT / 'tables.md').read_text())
