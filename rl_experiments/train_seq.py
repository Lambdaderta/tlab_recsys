"""Train and evaluate sequential models.

uv run python train_seq.py --n 2000 --stage val --model sasrec|sqn|sa2c|twotower [--steps 4000]
"""
import argparse
import copy
import json
import time

import numpy as np
import torch

from common import CACHE, OUT, Stage, recommend, save_result
from seq import DEVICE, NegSampler, Sequences, SeqRec, TwoTower, seqrec_loss, twotower_loss


@torch.no_grad()
def seqrec_scores(model, seqs, stage, L, head='ce', kappa=0.0):
    model.eval()
    hs = []
    for s in range(0, len(stage.users), 256):
        hs.append(model.enc(*seqs.last(stage.users[s:s + 256], L))[:, -1])
    h = torch.cat(hs)

    @torch.no_grad()
    def fn(rows):
        hr = h[torch.as_tensor(rows, device=DEVICE)]
        if head == 'q':
            s = model.qvalues(hr)
        else:
            s = model.logits(hr).log_softmax(-1)
            if kappa:
                s = s + kappa * model.qvalues(hr)
        return s.float().cpu().numpy()
    return fn


@torch.no_grad()
def twotower_scores(model, seqs, stage, L):
    model.eval()
    it, fb, _ = seqs.last(stage.users, L - 1)
    u = model.user(it, fb)
    v = model.items_vec()
    return torch.no_grad()(lambda rows: (u[torch.as_tensor(rows, device=DEVICE)] @ v.T).cpu().numpy())


def train(args, stage, seqs):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    V = stage.vocab.size
    if args.model == 'twotower':
        model = TwoTower(V, d=args.d).to(DEVICE)
        cnt = np.bincount(seqs.idx, minlength=V).astype(np.float32) + 1
        log_q = torch.tensor(np.log(cnt / cnt.sum()), device=DEVICE)
    else:
        kw = dict(layers=args.layers, heads=args.heads, prefix=args.M > 0) if args.arch == 'gpt2' else {}
        model = SeqRec(V, q_head=args.model in ('sqn', 'sa2c'), arch=args.arch, d=args.d, L=args.L, **kw).to(DEVICE)
        target = copy.deepcopy(model).eval()
        sampler = NegSampler(seqs, V)
        with torch.no_grad():  # start item biases at log-popularity
            cnt = np.bincount(seqs.idx, minlength=V).astype(np.float32) + 1
            model.bias.copy_(torch.tensor(np.log(cnt / cnt.sum()), device=DEVICE))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    log, t0 = [], time.time()
    for step in range(1, args.steps + 1):
        model.train()
        batch = seqs.batch(rng, args.B, args.L)
        if args.model == 'twotower':
            loss, logs = twotower_loss(model, batch, log_q)
        else:
            mode = 'ce' if args.model == 'sasrec' else args.model
            loss, logs = seqrec_loss(model, target, batch, mode, args.P, gen, sampler)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if args.model != 'twotower':
            with torch.no_grad():  # EMA target network for the TD target
                for p, tp in zip(model.parameters(), target.parameters()):
                    tp.lerp_(p, 0.01)
        if step % 200 == 0:
            logs.update(step=step, sec=round(time.time() - t0))
            log.append(logs)
            print(logs, flush=True)
        if step in args.eval_at:  # val checkpoints: pick the step count for the test refit
            r = save_result(f'{args.model}{args.tag}_s{step}', stage, recommend(stage, seqrec_scores(model, seqs, stage, args.L)),
                            {'steps': step, 'seed': args.seed})
            log.append({'step': step, 'org_new_ndcg10': r['org_new']['ndcg@10'], 'new_ndcg10': r['new']['ndcg@10']})
    return model, log


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--stage', default='val')
    ap.add_argument('--model', default='sasrec')
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--B', type=int, default=128)
    ap.add_argument('--L', type=int, default=200)
    ap.add_argument('--P', type=int, default=2048)
    ap.add_argument('--d', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tag', default='')
    ap.add_argument('--arch', default='gpt2')
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--M', type=int, default=256, help='long-term prefix sample size (0 = no prefix)')
    ap.add_argument('--hist', type=int, default=10**9, help='fit events kept per user (default: all)')
    ap.add_argument('--eval_at', default='', help='comma-separated steps to evaluate on this stage')
    a = ap.parse_args()
    a.eval_at = {int(x) for x in a.eval_at.split(',') if x}
    if a.model == 'twotower':
        a.L, a.B = 51, 2048
    stage = Stage(a.n, a.stage)
    seqs = Sequences(stage, hist=a.hist, M=a.M if a.arch == 'gpt2' and a.model != 'twotower' else 0)
    model, log = train(a, stage, seqs)
    name = a.model + a.tag
    torch.save({'state': model.state_dict(), 'args': {**vars(a), 'eval_at': sorted(a.eval_at)}}, CACHE / f'ckpt_{a.n}_{a.stage}_{name}.pt')
    with open(OUT / f'trainlog_{a.n}_{a.stage}_{name}.json', 'w') as f:
        json.dump(log, f)
    extra = {'steps': a.steps, 'seed': a.seed, 'train_sec': log[-1]['sec'] if log else 0}
    if a.model == 'twotower':
        save_result(name, stage, recommend(stage, twotower_scores(model, seqs, stage, a.L)), extra)
    else:
        save_result(name, stage, recommend(stage, seqrec_scores(model, seqs, stage, a.L)), extra)
        if model.q is not None:
            save_result(name + '_qhead', stage, recommend(stage, seqrec_scores(model, seqs, stage, a.L, head='q')), extra)
