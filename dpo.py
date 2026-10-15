"""The two ways to teach the start model from the comparisons, and the
measurements used on each.

  winners   copy the chosen lines only (plain next-word training)
  dpo       raise the chosen line, lower the rejected line, both measured
            against a frozen copy of the start model
"""
import copy
import math

import torch
import torch.nn.functional as F

from data import TEST, prompts
from gpt import line_logp, write
from judge import prefer, score

PAIRS_PER_STEP = 64


def loss_of(model, ref, batch, mode, beta):
    p = [x['prompt'] for x in batch]
    c = [x['chosen'] for x in batch]
    r = [x['rejected'] for x in batch]
    if mode == 'winners':
        n = sum(len(x) for x in c)
        return -line_logp(model, p, c).sum() / n
    lc, lr = line_logp(model, p, c), line_logp(model, p, r)
    with torch.no_grad():
        lc0, lr0 = line_logp(ref, p, c), line_logp(ref, p, r)
    gap = (lc - lc0) - (lr - lr0)           # winner up, loser down, both against the copy
    return -F.logsigmoid(beta * gap).mean()


def teach(base, pairs, mode, beta=0.1, steps=600, lr=1e-4, seed=0,
          check_at=(), check=None):
    """Returns the taught model and the checks made on the way."""
    torch.manual_seed(seed)
    model, ref = copy.deepcopy(base), copy.deepcopy(base)
    for q in ref.parameters():
        q.requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    order = torch.randperm(len(pairs))
    log = []
    for step in range(1, steps + 1):
        j = (step - 1) * PAIRS_PER_STEP % len(pairs)
        idx = order[j:j + PAIRS_PER_STEP].tolist()
        if len(idx) < PAIRS_PER_STEP:
            order = torch.randperm(len(pairs))
            idx = order[:PAIRS_PER_STEP].tolist()
        loss = loss_of(model, ref, [pairs[i] for i in idx], mode, beta)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if check and step in check_at:
            m = check(model)
            m.update(step=step, loss=float(loss))
            log.append(m)
            print(mode, beta, m, flush=True)
    return model, log


# ---------- measurements ----------

TEST_PROMPTS = prompts(1000, TEST)


@torch.no_grad()
def lines_of(model, seed=7):
    dice = torch.Generator().manual_seed(seed)
    return write(model, TEST_PROMPTS, dice=dice)


@torch.no_grad()
def measure(model, base, base_lines=None):
    """judge score per line, win rate against the start model, how far the
    model moved (drift), and how many different words it still uses."""
    lines = lines_of(model)
    base_lines = base_lines or lines_of(base, seed=8)
    s = [score(x) for x in lines]
    wins = ties = 0
    for a, b in zip(lines, base_lines):
        w = prefer(a, b)
        wins += w == 'a'
        ties += w is None
    lp = line_logp(model, TEST_PROMPTS, lines)
    lp0 = line_logp(base, TEST_PROMPTS, lines)
    words = [w for x in lines for w in x[:-1]]
    return {'score': sum(s) / len(s),
            'happy_lines': sum(v > 0 for v in s) / len(s),
            'sad_lines': sum(v < 0 for v in s) / len(s),
            'win': wins / len(lines), 'tie': ties / len(lines),
            'drift': float((lp - lp0).mean()),
            'distinct': len(set(words)), 'words': len(words),
            'length': len(words) / len(lines)}


@torch.no_grad()
def reward_check(model, base, test_pairs, beta):
    """The hidden judge inside the taught model: beta * (its log chance of a
    line minus the start model's). How often does it rank the judge's winner
    higher, on pairs it never saw?"""
    p = [x['prompt'] for x in test_pairs]
    c = [x['chosen'] for x in test_pairs]
    r = [x['rejected'] for x in test_pairs]
    rc = beta * (line_logp(model, p, c) - line_logp(base, p, c))
    rr = beta * (line_logp(model, p, r) - line_logp(base, p, r))
    return float((rc > rr).float().mean())


def perplexity(model, nums):
    """How well the model still guesses real Shakespeare (lower is better)."""
    from train_base import test_loss
    return math.exp(test_loss(model, nums))
