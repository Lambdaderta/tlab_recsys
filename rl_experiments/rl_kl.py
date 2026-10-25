"""RL direction B: KL-regularised fine-tuning of SASRec against a learned reward model.

  1. RM: separate encoder + bilinear head, regression of the logged reward of the next event
     r(s, a) = E[reward | state s, item a started]  (skip 0, listen+ 1, like 2, dislike -1).
     Checked against item-mean / user-mean baselines on val-window transitions.
  2. RLOO with exact KL to the SFT policy over the full vocabulary.
  3. Exact tilted policy  pi*(a|s) ∝ pi_ref(a|s) exp(r(s,a)/beta)  -- the closed-form optimum
     of the one-step objective E[r] - beta KL, computed without training.

uv run python rl_kl.py --n 2000 --stage val rm        # train + check the reward model
uv run python rl_kl.py --n 2000 --stage val tilt      # tilted policies for several betas
uv run python rl_kl.py --n 2000 --stage val rloo --beta 0.3
"""
import argparse
import copy
import json
import time

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from common import CACHE, OUT, REWARD, STAGES, Stage, load_events, recommend, reward_expr, save_result
from seq import DEVICE, NegSampler, Sequences, SeqRec

L = 200


def load_model(n, stage, name, q_head=False):
    ck = torch.load(CACHE / f'ckpt_{n}_{stage}_{name}.pt', map_location=DEVICE)
    a = ck['args']
    m = SeqRec(ck['state']['bias'].shape[0], q_head=q_head or 'q.weight' in ck['state'], d=a['d'], L=a['L']).to(DEVICE)
    m.load_state_dict(ck['state'])
    return m.eval()


def train_rm(stage, seqs, steps, seed=1, B=128, P=4096, lr=1e-3):
    """Reward model = encoder + Q head trained with gamma = 0 (plain regression on logged transitions)."""
    from seq import next_positions, subsample, REWARD_T
    torch.manual_seed(seed)
    rng, gen = np.random.default_rng(seed), torch.Generator().manual_seed(seed)
    rm = SeqRec(stage.vocab.size, q_head=True, L=L).to(DEVICE)
    opt = torch.optim.AdamW(rm.parameters(), lr=lr)
    t0 = time.time()
    for step in range(1, steps + 1):
        rm.train()
        it, fb, org = seqs.batch(rng, B, L)
        h = rm.enc(it, fb, org)
        b, t = subsample(next_positions(it, fb, False), P, gen)
        loss = F.mse_loss(rm.qvalues(h[b, t], it[b, t + 1]), REWARD_T[fb[b, t + 1]])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0:
            print({'step': step, 'mse': round(loss.item(), 4), 'sec': round(time.time() - t0)}, flush=True)
    return rm.eval()


def window_transitions(n, stage_name, vocab):
    """Sequences over fit + window events, and the (user row, position) of every window event,
    so a model sees the true history before each held-out event."""
    cfg = STAGES[stage_name]
    ev = load_events(n).filter(pl.col('timestamp') < cfg['end'])
    ev = vocab.encode(ev)
    seqs = Sequences(ev, hist=10**9)
    ts = ev.sort('uid', 'timestamp', maintain_order=True)['timestamp'].to_numpy()
    is_win = ts >= cfg['start']
    return seqs, is_win


@torch.no_grad()
def rm_quality(rm, n, stage, max_events=60_000, seed=0):
    """Spearman / MAE of the RM vs item-mean and user-mean on held-out window transitions (in-vocab items)."""
    seqs, is_win = window_transitions(n, stage.name, stage.vocab)
    pos = np.nonzero(is_win)[0]
    user_of = np.searchsorted(seqs.off, pos, side='right') - 1
    keep = (seqs.idx[pos] > 1) & (pos - seqs.off[user_of] >= 1)
    pos, user_of = pos[keep], user_of[keep]
    sel = np.random.default_rng(seed).choice(len(pos), min(max_events, len(pos)), replace=False)
    pos, user_of = pos[sel], user_of[sel]
    y = REWARD[seqs.fb[pos]]
    pred = []
    for s in range(0, len(pos), 512):
        w = [seqs.window(u, p - seqs.off[u], L) for u, p in zip(user_of[s:s + 512], pos[s:s + 512])]
        it, fb, org = [torch.tensor(np.stack(x), device=DEVICE) for x in zip(*w)]
        a = torch.tensor(seqs.idx[pos[s:s + 512]], device=DEVICE)
        pred.append(rm.qvalues(rm.enc(it, fb, org)[:, -1], a).cpu().numpy())
    pred = np.concatenate(pred)
    # baselines from fit data only
    fit = stage.fit.with_columns(r=reward_expr())
    item_mean = dict(fit.group_by('idx').agg(pl.col('r').mean()).iter_rows())
    user_mean = dict(fit.group_by('uid').agg(pl.col('r').mean()).iter_rows())
    gm = float(fit['r'].mean())
    uids = seqs.uids[user_of]
    base = {'item_mean': np.array([item_mean.get(i, gm) for i in seqs.idx[pos]]),
            'user_mean': np.array([user_mean.get(u, gm) for u in uids])}
    base['item+user'] = base['item_mean'] + base['user_mean'] - gm

    def score(p):
        rank = lambda x: np.argsort(np.argsort(x))
        return {'spearman': float(np.corrcoef(rank(p), rank(y))[0, 1]), 'mae': float(np.abs(p - y).mean()),
                'auc_plus': auc(p, y >= 1), 'within_user_auc': within_user_auc(p, y >= 1, uids)}
    out = {'rm': score(pred), **{k: score(v) for k, v in base.items()}, 'n_events': int(len(y))}
    return out


def within_user_auc(p, y, groups, min_each=5):
    """Mean AUC inside each user (users with >= min_each positives and negatives). User-level offsets
    such as the user's skip rate score exactly 0.5 here, so this is what matters for ranking items."""
    out = []
    for g in np.unique(groups):
        m = groups == g
        if min(y[m].sum(), (~y[m]).sum()) >= min_each:
            out.append(auc(p[m] + 1e-9 * np.random.default_rng(0).random(m.sum()), y[m]))
    return float(np.mean(out)) if out else float('nan')


def auc(p, y):
    o = np.argsort(p)
    r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1)
    npos = y.sum()
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * (len(y) - npos)))


@torch.no_grad()
def eval_states(model, seqs, users):
    hs = []
    for s in range(0, len(users), 256):
        it, fb, org = seqs.last(users[s:s + 256], L)
        hs.append(model.enc(it, fb, org)[:, -1])
    return torch.cat(hs)


def policy_logp(model, h):
    # -1e9 rather than -inf: keeps (log p - log p_ref) finite for padding/OOV so KL gradients stay clean
    lg = model.logits(h)
    lg[:, :2] = -1e9
    return lg.log_softmax(-1)


def tilted_scores(ref, rm, seqs, stage, beta):
    h_ref, h_rm = eval_states(ref, seqs, stage.users), eval_states(rm, seqs, stage.users)

    @torch.no_grad()
    def fn(rows):
        r = torch.as_tensor(rows, device=DEVICE)
        return (policy_logp(ref, h_ref[r]) + rm.qvalues(h_rm[r]) / beta).cpu().numpy()
    return fn


def policy_scores(policy, seqs, stage):
    h = eval_states(policy, seqs, stage.users)

    @torch.no_grad()
    def fn(rows):
        return policy_logp(policy, h[torch.as_tensor(rows, device=DEVICE)]).cpu().numpy()
    return fn


@torch.no_grad()
def reward_stats(policy_fn, rm, ref, seqs, stage, beta):
    """E_pi[r_hat] and KL(pi || pi_ref) on eval-user states, for the reward-KL frontier."""
    h_rm, h_ref = eval_states(rm, seqs, stage.users), eval_states(ref, seqs, stage.users)
    er, kl = [], []
    for s in range(0, len(stage.users), 256):
        rows = torch.arange(s, min(s + 256, len(stage.users)), device=DEVICE)
        lp = torch.as_tensor(policy_fn(rows.cpu().numpy()), device=DEVICE).log_softmax(-1)
        lr = policy_logp(ref, h_ref[rows])
        p = lp.exp()
        er.append((p * rm.qvalues(h_rm[rows])).sum(-1))
        kl.append((p * (lp - lr)).sum(-1))
    return float(torch.cat(er).mean()), float(torch.cat(kl).mean())


def rloo(ref, rm, seqs, steps, beta, K=8, B=256, lr=1e-4, seed=0, log_every=100):
    """RLOO on states = ends of random training windows; reward from the frozen RM; exact KL."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    policy = copy.deepcopy(ref)
    policy.eval()  # no dropout: sampling and KL must use the same deterministic policy as the reference
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    log, t0 = [], time.time()
    for step in range(1, steps + 1):
        it, fb, org = seqs.batch(rng, B, L)
        with torch.no_grad():
            lref = policy_logp(ref, ref.enc(it, fb, org)[:, -1])
            h_rm = rm.enc(it, fb, org)[:, -1]
        lp = policy_logp(policy, policy.enc(it, fb, org)[:, -1])
        a = torch.multinomial(lp.detach().exp(), K, replacement=True)            # [B, K]
        with torch.no_grad():
            r = (h_rm[:, None, :] * rm.q(a)).sum(-1) + rm.qb[a]                  # [B, K]
            adv = r - (r.sum(1, keepdim=True) - r) / (K - 1)                      # leave-one-out baseline
        pg = -(adv * lp.gather(1, a)).mean()
        p = lp.exp()
        kl = (p * (lp - lref)).sum(-1).mean()
        loss = pg + beta * kl
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        if step % log_every == 0:
            log.append({'step': step, 'r': r.mean().item(), 'kl': kl.item(), 'sec': round(time.time() - t0)})
            print(log[-1], flush=True)
    return policy.eval(), log


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['rm', 'rmq', 'tilt', 'rloo'])
    ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--stage', default='val')
    ap.add_argument('--rm_steps', type=int, default=4000)
    ap.add_argument('--steps', type=int, default=600)
    ap.add_argument('--beta', type=float, default=0.3)
    ap.add_argument('--betas', default='0.1,0.3,1,3')
    ap.add_argument('--sft', default='sasrec')
    a = ap.parse_args()
    stage = Stage(a.n, a.stage)
    seqs = Sequences.from_stage(stage)
    rm_path = CACHE / f'ckpt_{a.n}_{a.stage}_rm.pt'

    if a.cmd in ('rm', 'rmq'):
        if a.cmd == 'rm':
            rm = train_rm(stage, seqs, a.rm_steps)
            torch.save({'state': rm.state_dict(), 'args': {'d': 64, 'L': L}}, rm_path)
        else:
            rm = load_model(a.n, a.stage, 'rm', q_head=True)
        q = rm_quality(rm, a.n, stage)
        print(json.dumps(q, indent=1))
        with open(OUT / f'rm_quality_{a.n}_{a.stage}.json', 'w') as f:
            json.dump(q, f, indent=1)
    else:
        rm = load_model(a.n, a.stage, 'rm', q_head=True)
        ref = load_model(a.n, a.stage, a.sft)
        if a.cmd == 'tilt':
            for beta in map(float, a.betas.split(',')):
                fn = tilted_scores(ref, rm, seqs, stage, beta)
                er, kl = reward_stats(fn, rm, ref, seqs, stage, beta)
                save_result(f'tilt_b{beta:g}', stage, recommend(stage, fn), {'beta': beta, 'E_r': er, 'KL': kl})
            fn = policy_scores(ref, seqs, stage)
            er, kl = reward_stats(fn, rm, ref, seqs, stage, 1.0)
            save_result(f'{a.sft}_ref', stage, recommend(stage, fn), {'E_r': er, 'KL': kl})
        else:
            policy, log = rloo(ref, rm, seqs, a.steps, a.beta)
            fn = policy_scores(policy, seqs, stage)
            er, kl = reward_stats(fn, rm, ref, seqs, stage, a.beta)
            save_result(f'rloo_b{a.beta:g}', stage, recommend(stage, fn),
                        {'beta': a.beta, 'E_r': er, 'KL': kl, 'steps': a.steps, 'log': log})
