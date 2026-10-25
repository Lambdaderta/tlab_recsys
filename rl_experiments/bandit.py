"""RL direction C: exploration in a simulator (a toy, not an offline metric).

World:  each eval user u has a fixed state s_u (after all fit events). The true mean reward of item a is
        mu(u, a) = RM(s_u, a); an observed reward is mu + N(0, noise^2). The RM is the frozen reward model
        from rl_kl.py, so the simulator inherits its biases.
Agent:  knows only the SASRec prior z(u, a) (log pi_ref, standardised per user) on a pool of the user's top-200
        SASRec items, and learns online:  est(u, a) = c0 + c1 * z(u, a) + theta_a,
        with (c0, c1) by least squares on all observations and theta_a a Gaussian item offset shared across users.
Each round every user gets a slate of S items they have not been shown before in the episode.
Policies: greedy, eps-greedy, UCB, Thompson sampling; plus 'prior' (SASRec order) and 'oracle' (true mu).

uv run python bandit.py --n 2000 --stage val
"""
import argparse
import json

import numpy as np
import torch

from common import OUT, Stage
from rl_kl import eval_states, load_model, policy_logp
from seq import DEVICE, Sequences


@torch.no_grad()
def build_world(n, stage, pool=200):
    seqs = Sequences.from_stage(stage)
    ref, rm = load_model(n, stage.name, 'sasrec'), load_model(n, stage.name, 'rm', q_head=True)
    h_ref, h_rm = eval_states(ref, seqs, stage.users), eval_states(rm, seqs, stage.users)
    out = []
    # chunks of users: full-vocab logits for 7.7K users x 338K items (10K sample) is ~10 GB at once
    for s in range(0, len(stage.users), 512):
        top = policy_logp(ref, h_ref[s:s + 512]).topk(pool, -1)
        items, z = top.indices, top.values
        z = (z - z.mean(1, keepdim=True)) / z.std(1, keepdim=True)
        mu = (h_rm[s:s + 512, None, :] * rm.q(items)).sum(-1) + rm.qb[items]
        out.append((items.cpu().numpy(), z.cpu().numpy(), mu.cpu().numpy()))
    return tuple(np.concatenate(x) for x in zip(*out))


def simulate(items, z, mu, policy, rounds=8, slate=10, noise=0.7, prior_var=0.25, eps=0.1, ucb_c=1.0, seed=0):
    rng = np.random.default_rng(seed)
    U, P = items.shape
    uniq, item_col = np.unique(items, return_inverse=True)
    item_col = item_col.reshape(U, P)
    # Gaussian posterior of theta_a on the residuals: precision and precision-weighted mean
    prec = np.full(len(uniq), 1 / prior_var)
    pm = np.zeros(len(uniq))
    shown = np.zeros((U, P), bool)
    X, Y = [], []
    c = np.array([mu.mean(), 0.0])  # before any data: the global mean, no slope
    total, regret, cov = [], [], set()
    best = -np.sort(-mu, 1)  # oracle per-round gain: next best unseen items
    for t in range(rounds):
        theta_mean = pm / prec
        if policy == 'thompson':
            theta = theta_mean + rng.standard_normal(len(uniq)) / np.sqrt(prec)
        elif policy == 'ucb':
            theta = theta_mean + ucb_c * noise / np.sqrt(prec)
        else:
            theta = theta_mean
        est = c[0] + c[1] * z + theta[item_col]
        if policy == 'prior':
            est = z.copy()
        elif policy == 'oracle':
            est = mu.copy()
        est[shown] = -np.inf
        pick = np.argpartition(-est, slate, 1)[:, :slate]
        if policy == 'eps':
            explore = rng.random((U, slate)) < eps
            rand = np.argsort(rng.random((U, P)) + shown * 2, 1)[:, :slate]  # random unseen
            pick = np.where(explore, rand, pick)
        rows = np.repeat(np.arange(U), slate)
        cols = pick.ravel()
        shown[rows, cols] = True
        r = mu[rows, cols] + noise * rng.standard_normal(len(cols))
        total.append(float(mu[rows, cols].mean()))
        regret.append(float(best[:, t * slate:(t + 1) * slate].mean() - mu[rows, cols].mean()))
        cov.update(items[rows, cols].tolist())
        # learn: refit (c0, c1) on all data, then update item offsets with the residuals of this round
        X.append(np.stack([np.ones(len(cols)), z[rows, cols]], 1)); Y.append(r)
        Xa, Ya = np.concatenate(X), np.concatenate(Y)
        c = np.linalg.lstsq(Xa, Ya, rcond=None)[0]
        res = r - (c[0] + c[1] * z[rows, cols])
        ic = item_col[rows, cols]
        np.add.at(prec, ic, 1 / noise ** 2)
        np.add.at(pm, ic, res / noise ** 2)
    return {'policy': policy, 'mean_reward': float(np.mean(total)), 'cum_regret': float(np.sum(regret)),
            'reward_by_round': total, 'coverage_items': len(cov)}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--stage', default='val')
    ap.add_argument('--rounds', type=int, default=8)  # 8 x 10 = 80 of the 200-item pool; at 20 rounds every policy shows the whole pool
    ap.add_argument('--seeds', type=int, default=3)
    a = ap.parse_args()
    stage = Stage(a.n, a.stage)
    items, z, mu = build_world(a.n, stage)
    print('world: users', items.shape[0], 'pool', items.shape[1], 'mu mean', mu.mean().round(3), 'std', mu.std().round(3),
          'corr(z, mu)', np.corrcoef(z.ravel(), mu.ravel())[0, 1].round(3))
    out = []
    for pol in ['prior', 'greedy', 'eps', 'ucb', 'thompson', 'oracle']:
        runs = [simulate(items, z, mu, pol, rounds=a.rounds, seed=s) for s in range(a.seeds)]
        m = {k: float(np.mean([r[k] for r in runs])) for k in ['mean_reward', 'cum_regret', 'coverage_items']}
        m['mean_reward_sd'] = float(np.std([r['mean_reward'] for r in runs]))
        m['reward_by_round'] = np.mean([r['reward_by_round'] for r in runs], 0).tolist()
        m['policy'] = pol
        out.append(m)
        print(pol, {k: round(v, 4) for k, v in m.items() if isinstance(v, float)})
    with open(OUT / f'bandit_{a.n}_{a.stage}.json', 'w') as f:
        json.dump(out, f, indent=1)
