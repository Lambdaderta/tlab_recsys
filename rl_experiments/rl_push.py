"""Push-based RL (actions = items the recommender pushed, is_organic = 0).

Reward of a push at time t:  1[like] + 1[organic listen of ANOTHER track of the same artist in (t + 1d, t + 3d]]
                             - 1[dislike]   (listen+ alone carries ~no signal, see DECISIONS.md / organic analysis)
Models (all GPT-2 encoders, test stage = fit on train + gap + val):
  pi_ref : GPT-2 SASRec on all events (train_seq.py, ckpt_<n>_<stage>_sasrec<tag>.pt)
  beta   : GPT-2 trained to predict the next PUSHED item = estimated logging policy (propensities)
  rm     : GPT-2 + bilinear head, regression of the push reward  r(s, a)
Policies:
  awr    : reward-weighted imitation of pushes + beta_kl * KL(pi || pi_ref)
  opc    : off-policy REINFORCE with clipped pi/beta weight and top-K correction (Chen et al. 2019) + KL
  tilt   : exact optimum of E[r] - b KL:  pi ∝ pi_ref exp(rm / b), no training
Evaluation: organic-new / new / all NDCG on the window (save_result) + SNIPS value on held-out window pushes.

uv run python rl_push.py --n 10000 --stage test --ref sasrec_s0
"""
import argparse
import copy
import json
import time

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from common import CACHE, DAY, DATA, DISLIKE, LIKE, OUT, STAGES, Stage, load_events, recommend, save_result
from seq import DEVICE, NegSampler, Sequences, SeqRec, candidate_logits, subsample

ARTIST = DATA.parent.parent / 'artist_item_mapping.parquet'
L, M = 200, 256
RET_GAP, RET_END = 1 * DAY, 3 * DAY


def push_rewards(ev):
    """ev: encoded events sorted by (uid, timestamp). Returns reward per event (NaN for organic events)."""
    art = pl.read_parquet(ARTIST).unique('item_id')
    e = ev.with_row_index('row').join(art, on='item_id', how='left').sort('uid', 'timestamp', maintain_order=True)
    org = (e.filter((pl.col('is_organic') == 1) & pl.col('artist_id').is_not_null())
           .select('uid', 'artist_id', pl.col('timestamp').alias('t_org'), pl.col('item_id').alias('item_org'))
           .sort('t_org'))
    push = (e.filter(pl.col('is_organic') == 0)
            .with_columns(t_from=pl.col('timestamp') + RET_GAP).sort('t_from'))
    # first organic listen of the artist after t + 1d; counts if within t + 3d and a different track
    # (if that first one is the same track, a later other-track return is missed: a slight undercount)
    nxt = push.join_asof(org, left_on='t_from', right_on='t_org', by=['uid', 'artist_id'], strategy='forward')
    nxt = nxt.with_columns(
        ret=((pl.col('t_org') <= pl.col('timestamp') + RET_END) & (pl.col('item_org') != pl.col('item_id'))).fill_null(False),
        r=(pl.col('fb') == LIKE).cast(pl.Float32) - (pl.col('fb') == DISLIKE).cast(pl.Float32))
    nxt = nxt.with_columns(r=pl.col('r') + pl.col('ret').cast(pl.Float32))
    r = np.full(ev.height, np.nan, np.float32)
    r[nxt['row'].to_numpy()] = nxt['r'].to_numpy()
    return r


class PushData:
    """Sequences + per-event push reward; positions usable as (state, action, reward) transitions."""

    def __init__(self, events, horizon_end, start=None):
        ev = events.sort('uid', 'timestamp', maintain_order=True)
        self.seqs = Sequences(ev, hist=10**9, M=M)
        self.r = push_rewards(ev)
        ts = ev['timestamp'].to_numpy()
        user_of = np.searchsorted(self.seqs.off, np.arange(ev.height), side='right') - 1
        ok = ~np.isnan(self.r) & (ts + RET_END <= horizon_end) & (self.seqs.idx > 1)
        ok &= np.arange(ev.height) - self.seqs.off[user_of] >= 1  # needs some history
        if start is not None:
            ok &= ts >= start
        self.pos = np.nonzero(ok)[0]
        self.user_of = user_of

    def batch(self, rng, B, pos=None):
        p = rng.choice(self.pos, B) if pos is None else pos
        u = self.user_of[p]
        loc = p - self.seqs.off[u]
        w = [self.seqs.window(ui, e, L) + self.seqs.longterm(ui, e, L, rng) for ui, e in zip(u, loc)]
        state = [torch.tensor(np.stack(x), device=DEVICE) for x in zip(*w)]
        a = torch.tensor(self.seqs.idx[p], device=DEVICE)
        r = torch.tensor(self.r[p], device=DEVICE)
        return state, a, r


def new_gpt2(V, q_head=False):
    return SeqRec(V, q_head=q_head, arch='gpt2', d=128, L=L, layers=2, heads=4, prefix=True).to(DEVICE)


def load_gpt2(path, V, q_head=False):
    m = new_gpt2(V, q_head)
    m.load_state_dict(torch.load(path, map_location=DEVICE)['state'])
    return m.eval()


def log_policy(model, state):
    lg = model.logits(model.enc(*state)[:, -1])
    lg[:, :2] = -1e9
    return lg.log_softmax(-1)


def train_beta(data, V, steps, seed=0, B=256, lr=1e-3):
    """Logging-policy model: sampled-softmax CE on the pushed item given the state before it."""
    torch.manual_seed(seed)
    rng, gen = np.random.default_rng(seed), torch.Generator().manual_seed(seed)
    m = new_gpt2(V)
    sampler = NegSampler(data.seqs, V)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    for step in range(1, steps + 1):
        m.train()
        state, a, _ = data.batch(rng, B)
        h = m.enc(*state)[:, -1]
        lg = candidate_logits(h, a, sampler.draw(), m.enc.item, m.bias, sampler.log_q)
        loss = F.cross_entropy(lg, torch.zeros(len(a), dtype=torch.long, device=DEVICE))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 1000 == 0:
            print({'beta_step': step, 'ce': round(loss.item(), 3)}, flush=True)
    return m.eval()


def train_rm(data, V, steps, seed=1, B=256, lr=1e-3):
    """Push reward model: bilinear Q head (gamma = 0), MSE to the push reward."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = new_gpt2(V, q_head=True)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    for step in range(1, steps + 1):
        m.train()
        state, a, r = data.batch(rng, B)
        loss = F.mse_loss(m.qvalues(m.enc(*state)[:, -1], a), r)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 1000 == 0:
            print({'rm_step': step, 'mse': round(loss.item(), 4)}, flush=True)
    return m.eval()


def finetune(kind, ref, beta, data, steps, b_kl, seed=0, B=256, lr=1e-4, tau=1.0, K=10, clip=10.0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    pol = copy.deepcopy(ref).eval()  # eval mode: no dropout, so KL to the reference is exact
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    t0 = time.time()
    for step in range(1, steps + 1):
        state, a, r = data.batch(rng, B)
        lp = log_policy(pol, state)
        with torch.no_grad():
            lref = log_policy(ref, state)
            adv = r - r.mean()
        lpa = lp.gather(1, a[:, None]).squeeze(1)
        if kind == 'awr':
            w = torch.exp((adv / tau).clamp(-2, 2))
            pg = -(w / w.mean() * lpa).mean()
        else:  # off-policy REINFORCE with top-K correction
            with torch.no_grad():
                lb = log_policy(beta, state).gather(1, a[:, None]).squeeze(1)
                ratio = torch.exp(lpa.detach() - lb).clamp(max=clip)
                lam = K * (1 - lpa.detach().exp()) ** (K - 1)
            pg = -(ratio * lam * adv * lpa).mean()
        kl = (lp.exp() * (lp - lref)).sum(-1).mean()
        loss = pg + b_kl * kl
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            print({kind: step, 'pg': round(pg.item(), 4), 'kl': round(kl.item(), 4), 'sec': round(time.time() - t0)}, flush=True)
    return pol.eval()


@torch.no_grad()
def snips(policy_logp_fn, beta, test, n=20_000, clip=10.0, seed=0):
    """SNIPS value on held-out window pushes: sum(w r) / sum(w), w = min(pi / beta_hat, clip)."""
    rng = np.random.default_rng(seed)
    pos = rng.choice(test.pos, min(n, len(test.pos)), replace=False)
    ws, rs = [], []
    for s in range(0, len(pos), 256):
        state, a, r = test.batch(rng, None, pos[s:s + 256])
        lp = policy_logp_fn(state).gather(1, a[:, None]).squeeze(1)
        lb = log_policy(beta, state).gather(1, a[:, None]).squeeze(1)
        ws.append(torch.exp(lp - lb).clamp(max=clip).cpu()); rs.append(r.cpu())
    w, r = torch.cat(ws).numpy(), torch.cat(rs).numpy()
    return {'snips': float((w * r).sum() / w.sum()), 'logged_mean_r': float(r.mean()),
            'ess_share': float(w.sum() ** 2 / (w ** 2).sum() / len(w)), 'n': int(len(w))}


def scores_fn(logp_fn, seqs, stage):
    @torch.no_grad()
    def fn(rows):
        state = seqs.last(stage.users[rows], L)
        return logp_fn(state).cpu().numpy()
    return fn


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=10_000)
    ap.add_argument('--stage', default='test')
    ap.add_argument('--ref', default='sasrec_s0', help='checkpoint name of the GPT-2 SASRec (pi_ref)')
    ap.add_argument('--aux_steps', type=int, default=6000)
    ap.add_argument('--steps', type=int, default=500)
    ap.add_argument('--b_kl', type=float, default=0.1)
    ap.add_argument('--tilt_betas', default='0.3,1')
    a = ap.parse_args()
    stage = Stage(a.n, a.stage)
    cfg = STAGES[a.stage]
    V = stage.vocab.size
    train = PushData(stage.fit, horizon_end=cfg['fit_end'])
    full = stage.vocab.encode(load_events(a.n).filter(pl.col('timestamp') < cfg['end']))
    test = PushData(full, horizon_end=cfg['end'], start=cfg['start'])
    print('push transitions: train', len(train.pos), 'held-out window', len(test.pos),
          'mean r train', float(np.nanmean(train.r[train.pos])), flush=True)
    seqs = Sequences(stage, hist=10**9, M=M)

    ref = load_gpt2(CACHE / f'ckpt_{a.n}_{a.stage}_{a.ref}.pt', V)
    bpath, rpath = CACHE / f'ckpt_{a.n}_{a.stage}_push_beta.pt', CACHE / f'ckpt_{a.n}_{a.stage}_push_rm.pt'
    if bpath.exists():
        beta = load_gpt2(bpath, V)
    else:
        beta = train_beta(train, V, a.aux_steps)
        torch.save({'state': beta.state_dict()}, bpath)
    if rpath.exists():
        rm = load_gpt2(rpath, V, q_head=True)
    else:
        rm = train_rm(train, V, a.aux_steps)
        torch.save({'state': rm.state_dict()}, rpath)

    results = {}

    def report(name, logp_fn, extra):
        ope = snips(logp_fn, beta, test)
        res = save_result(name, stage, recommend(stage, scores_fn(logp_fn, seqs, stage)), {**extra, **ope})
        results[name] = {'ope': ope, 'org_new_ndcg10': res['org_new']['ndcg@10'], 'new_ndcg10': res['new']['ndcg@10']}
        print(name, results[name], flush=True)

    report('push_ref', lambda s: log_policy(ref, s), {'policy': 'pi_ref'})
    report('push_beta', lambda s: log_policy(beta, s), {'policy': 'logging-policy model'})
    for b in map(float, a.tilt_betas.split(',')):
        report(f'push_tilt_b{b:g}', lambda s, b=b: (log_policy(ref, s) + rm.qvalues(rm.enc(*s)[:, -1]) / b).log_softmax(-1),
               {'policy': 'exact tilted', 'b': b})
    for kind in ('awr', 'opc'):
        pol = finetune(kind, ref, beta, train, a.steps, a.b_kl)
        report(f'push_{kind}', lambda s, pol=pol: log_policy(pol, s), {'policy': kind, 'b_kl': a.b_kl, 'steps': a.steps})
    with open(OUT / f'rl_push_{a.n}_{a.stage}.json', 'w') as f:
        json.dump(results, f, indent=1)
