"""Sequence data and models: feedback-aware SASRec encoder, CE / Q heads, two-tower."""
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import LIKE, PLUS, REWARD

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
PAD_FB = 4
HIST = 2000  # keep the most recent HIST fit events per user


class Sequences:
    """Per-user fit sequences (idx, fb, organic), concatenated with offsets."""

    def __init__(self, events, hist=HIST, M=0):
        """events: an encoded events frame (uid, timestamp, idx, fb, is_organic) or a Stage (its fit data).
        M > 0: every window also gets up to M events sampled from the history BEFORE the window
        (point-in-time long-term summary, used as a GPT-2 prefix token)."""
        self.M = M
        events = getattr(events, 'fit', events)
        f = events.sort('uid', 'timestamp', maintain_order=True).group_by('uid', maintain_order=True).tail(hist)
        self.uids = f['uid'].unique(maintain_order=True).to_numpy()
        lens = f.group_by('uid', maintain_order=True).len()['len'].to_numpy()
        self.off = np.concatenate([[0], np.cumsum(lens.astype(np.int64))])
        self.idx = f['idx'].to_numpy().astype(np.int64)
        self.fb = f['fb'].to_numpy().astype(np.int64)
        self.org = f['is_organic'].to_numpy().astype(np.int64)
        self.row = {u: i for i, u in enumerate(self.uids)}

    @classmethod
    def from_stage(cls, stage, hist=HIST, M=0):
        return cls(stage.fit, hist, M)

    def longterm(self, user_row, end, L, rng=None):
        """Up to M (item, fb) pairs from positions [0, end - L) of the user: random in training,
        evenly spaced at inference."""
        s = self.off[user_row]
        n = max(0, end - L)
        if n == 0:
            pick = np.zeros(0, dtype=np.int64)
        elif n <= self.M:
            pick = np.arange(n)
        elif rng is not None:
            pick = np.sort(rng.choice(n, self.M, replace=False))
        else:
            pick = np.linspace(0, n - 1, self.M).astype(np.int64)
        it, fb = self.idx[s + pick], self.fb[s + pick]
        pad = self.M - len(pick)
        return [np.concatenate([np.zeros(pad, np.int64), it]), np.concatenate([np.full(pad, PAD_FB), fb])]

    def window(self, user_row, end, L):
        s, e = self.off[user_row], self.off[user_row] + end
        b = max(s, e - L)
        out = []
        for a, pad in ((self.idx, 0), (self.fb, PAD_FB), (self.org, 0)):
            w = a[b:e]
            out.append(np.concatenate([np.full(L - len(w), pad), w]))
        return out

    def batch(self, rng, B, L):
        """Random training windows: user uniform, end position uniform over the user's kept history."""
        rows = rng.integers(0, len(self.uids), B)
        lens = self.off[rows + 1] - self.off[rows]
        ends = np.maximum(2, (rng.random(B) * lens).astype(int) + 1)
        ends = np.minimum(ends, lens)
        w = [self.window(r, e, L) + (self.longterm(r, e, L, rng) if self.M else []) for r, e in zip(rows, ends)]
        return [torch.tensor(np.stack(x), device=DEVICE) for x in zip(*w)]

    def last(self, uids, L):
        """Most recent window for each eval user (state after all fit events)."""
        out = []
        for u in uids:
            r = self.row[u]
            end = self.off[r + 1] - self.off[r]
            out.append(self.window(r, end, L) + (self.longterm(r, end, L) if self.M else []))
        return [torch.tensor(np.stack(x), device=DEVICE) for x in zip(*out)]


class Encoder(nn.Module):
    """Causal transformer over tokens = item + feedback + organic + position."""

    def __init__(self, V, d=64, L=200, layers=2, heads=2, drop=0.1):
        super().__init__()
        self.item = nn.Embedding(V, d, padding_idx=0)
        self.fb = nn.Embedding(5, d)
        self.org = nn.Embedding(2, d)
        self.pos = nn.Embedding(L, d)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, drop, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d)
        self.register_buffer('mask', torch.triu(torch.full((L, L), float('-inf')), 1))
        for e in (self.item, self.fb, self.org, self.pos):  # default N(0, 1) gives logits with std ~ sqrt(d)
            nn.init.normal_(e.weight, std=0.02)

    def forward(self, items, fb, org):
        L = items.shape[1]
        x = self.item(items) + self.fb(fb) + self.org(org) + self.pos.weight[:L]
        return self.norm(self.tf(x, mask=self.mask[:L, :L], is_causal=True))


class GPT2Encoder(nn.Module):
    """Hugging Face GPT2Model over tokens = item + feedback + organic (GPT-2 adds learned positions and ln_f).
    Item ids are the tokens; our own item table (tied to the output) replaces GPT-2's wte via inputs_embeds.
    Optional prefix token = projection of the mean (item + feedback) embedding of the long-term sample."""

    def __init__(self, V, d=128, L=200, layers=2, heads=4, drop=0.1, prefix=True):
        super().__init__()
        from transformers import GPT2Config, GPT2Model
        self.item = nn.Embedding(V, d, padding_idx=0)
        self.fb = nn.Embedding(5, d)
        self.org = nn.Embedding(2, d)
        self.prefix = nn.Linear(d, d) if prefix else None
        cfg = GPT2Config(vocab_size=1, n_positions=L + 1, n_embd=d, n_layer=layers, n_head=heads,
                         resid_pdrop=drop, embd_pdrop=drop, attn_pdrop=drop)
        self.gpt = GPT2Model(cfg)
        for e in (self.item, self.fb, self.org):
            nn.init.normal_(e.weight, std=0.02)

    def forward(self, items, fb, org, lt_items=None, lt_fb=None):
        x = self.item(items) + self.fb(fb) + self.org(org)
        mask = (items > 0).long()
        if self.prefix is not None:
            m = (lt_items > 0).float().unsqueeze(-1)
            lt = ((self.item(lt_items) + self.fb(lt_fb)) * m).sum(1) / m.sum(1).clamp(min=1)
            x = torch.cat([self.prefix(lt)[:, None], x], 1)
            mask = torch.cat([torch.ones_like(mask[:, :1]), mask], 1)
        h = self.gpt(inputs_embeds=x, attention_mask=mask).last_hidden_state
        return h[:, 1:] if self.prefix is not None else h


class SeqRec(nn.Module):
    """Encoder + CE head (tied item embeddings) + optional Q head (separate output table)."""

    def __init__(self, V, q_head=False, arch='tf', **kw):
        super().__init__()
        self.enc = GPT2Encoder(V, **kw) if arch == 'gpt2' else Encoder(V, **kw)
        d = self.enc.item.embedding_dim
        self.bias = nn.Parameter(torch.zeros(V))
        self.q = nn.Embedding(V, d) if q_head else None
        self.qb = nn.Parameter(torch.zeros(V)) if q_head else None
        if q_head:
            nn.init.normal_(self.q.weight, std=0.01)

    def logits(self, h):
        return h @ self.enc.item.weight.T + self.bias

    def qvalues(self, h, a=None):
        if a is None:
            return h @ self.q.weight.T + self.qb
        return (h * self.q(a)).sum(-1) + self.qb[a]


def next_positions(items, fb, positives_only):
    """(batch, pos) pairs whose next event is a valid training target."""
    nxt_items, nxt_fb = items[:, 1:], fb[:, 1:]
    ok = (nxt_items > 1) & (items[:, :-1] > 0)
    if positives_only:
        ok &= (nxt_fb == PLUS) | (nxt_fb == LIKE)
    return ok.nonzero(as_tuple=True)


def subsample(pos, P, gen):
    if len(pos[0]) <= P:
        return pos
    keep = torch.randperm(len(pos[0]), generator=gen)[:P].to(pos[0].device)
    return pos[0][keep], pos[1][keep]


REWARD_T = torch.tensor(REWARD, device=DEVICE)


class NegSampler:
    """Shared negatives for sampled softmax, drawn from popularity^0.75 (with the matching logQ)."""

    def __init__(self, seqs, V, n=8192, power=0.75, uniform=0.5):
        cnt = np.bincount(seqs.idx, minlength=V).astype(np.float64)
        cnt[:2] = 0
        q = cnt ** power
        # half uniform: items that popularity sampling almost never draws still get pushed down,
        # otherwise they keep their initial logit and float to the top at full-vocab inference
        u = (cnt > 0).astype(np.float64)
        q = (1 - uniform) * q / q.sum() + uniform * u / u.sum()
        self.q = torch.tensor(q / q.sum(), dtype=torch.float32, device=DEVICE)
        self.log_q = torch.log(self.q.clamp(min=1e-12))
        self.n = n

    def draw(self):
        return torch.multinomial(self.q, self.n, replacement=True)


def candidate_logits(h, tgt, negs, out_w, out_b, log_q):
    """[P, 1 + n] logits over (target, shared negatives), logQ-corrected, accidental hits masked."""
    pos = (h * out_w(tgt)).sum(-1, keepdim=True) + out_b[tgt][:, None]
    neg = h @ out_w(negs).T + out_b[negs]
    lg = torch.cat([pos - log_q[tgt][:, None], neg - log_q[negs]], 1)
    hit = torch.cat([torch.zeros_like(tgt[:, None], dtype=torch.bool), tgt[:, None] == negs[None, :]], 1)
    return lg.masked_fill(hit, float('-inf'))


def seqrec_loss(model, target, batch, mode, P, gen, sampler, gamma=0.5, td_weight=1.0, tau=1.0):
    """mode: 'ce'   - SASRec, sampled-softmax CE on next listen+ items;
             'sqn'  - CE on listen+ + double-DQN TD loss on all transitions (Xin et al. 2020);
             'sa2c' - TD loss + CE on all next items weighted by exp(advantage) from the Q head.
    The max / expectation over actions in the TD target and the advantage runs over the
    sampled candidate set (target + shared negatives), not the full vocabulary."""
    items, fb, org = batch[:3]
    h = model.enc(*batch)
    negs = sampler.draw()
    item_w, lq = model.enc.item, sampler.log_q
    logs = {}
    if mode in ('ce', 'sqn'):
        b, t = subsample(next_positions(items, fb, True), P, gen)
        lg = candidate_logits(h[b, t], items[b, t + 1], negs, item_w, model.bias, lq)
        loss = F.cross_entropy(lg, torch.zeros(len(b), dtype=torch.long, device=h.device))
        logs['ce'] = loss.item()
    if mode in ('sqn', 'sa2c'):
        b, t = subsample(next_positions(items, fb, False), P, gen)
        a, r = items[b, t + 1], REWARD_T[fb[b, t + 1]]
        q_sa = model.qvalues(h[b, t], a)
        with torch.no_grad():
            h_tgt = target.enc(*batch)
            hn = h[b, t + 1].detach()
            a_star = negs[(hn @ model.q(negs).T + model.qb[negs]).argmax(-1)]
            y = r + gamma * target.qvalues(h_tgt[b, t + 1], a_star)
        td = F.mse_loss(q_sa, y)
        logs['td'] = td.item()
        logs['q_sa'] = q_sa.mean().item()
        if mode == 'sqn':
            loss = loss + td_weight * td
        else:
            lg = candidate_logits(h[b, t], a, negs, item_w, model.bias, lq)
            with torch.no_grad():
                hb = h[b, t].detach()
                q_cand = torch.cat([q_sa.detach()[:, None], hb @ model.q(negs).T + model.qb[negs]], 1)
                pi = lg.detach().softmax(-1)
                adv = q_sa.detach() - (pi * q_cand).sum(-1)
                w = torch.exp((adv / tau).clamp(-2, 2))
                w = w / w.mean()
            ce = (F.cross_entropy(lg, torch.zeros(len(b), dtype=torch.long, device=h.device), reduction='none') * w).mean()
            loss = ce + td_weight * td
            logs['ce'] = ce.item()
            logs['adv'] = adv.mean().item()
    return loss, logs


class TwoTower(nn.Module):
    """User tower: mean of the last events' (item + feedback) embeddings -> MLP. Item tower: embedding -> MLP."""

    def __init__(self, V, d=64):
        super().__init__()
        self.item = nn.Embedding(V, d, padding_idx=0)
        self.fb = nn.Embedding(5, d)
        self.user_mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.item_mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def user(self, items, fb):
        m = (items > 0).float().unsqueeze(-1)
        x = ((self.item(items) + self.fb(fb)) * m).sum(1) / m.sum(1).clamp(min=1)
        return F.normalize(self.user_mlp(x), dim=-1)

    def items_vec(self, idx=None):
        e = self.item.weight if idx is None else self.item(idx)
        return F.normalize(self.item_mlp(e), dim=-1)


def twotower_loss(model, batch, log_q, temp=0.05):
    """In-batch sampled softmax with logQ correction. Target = last item of the window if it is listen+;
    the user tower sees the events before it."""
    items, fb, _ = batch
    pos = ((fb[:, -1] == PLUS) | (fb[:, -1] == LIKE)) & (items[:, -1] > 1)
    items, fb = items[pos], fb[pos]
    u = model.user(items[:, :-1], fb[:, :-1])
    tgt = items[:, -1]
    v = model.items_vec(tgt)
    lg = u @ v.T / temp - log_q[tgt][None, :]
    # the same item may appear twice in a batch: mask duplicates that are not on the diagonal
    dup = (tgt[:, None] == tgt[None, :]) & ~torch.eye(len(tgt), dtype=torch.bool, device=tgt.device)
    lg = lg.masked_fill(dup, float('-inf'))
    loss = F.cross_entropy(lg, torch.arange(len(tgt), device=tgt.device))
    return loss, {'loss': loss.item()}
