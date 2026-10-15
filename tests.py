"""Checks of the claims of the video. python3 tests.py

Reads the saved runs in files/ (results.json, pairs-info.json) and the two
saved models (base.pt, dpo.pt). No training; a few seconds on a CPU.
"""
import json
import math
from pathlib import Path

import torch

from data import TRAIN, TEST, VOCAB
from dpo import loss_of
from exercise import dpo_loss
from pairs import load

FILES = Path(__file__).parent / 'files'
RES = json.loads((FILES / 'results.json').read_text())
INFO = json.loads((FILES / 'pairs-info.json').read_text())
results = []


def check(text, ok):
    results.append(bool(ok))
    print(('ok    ' if ok else 'WRONG ') + text)


def near(a, b, tol):
    return abs(a - b) <= tol


base, after = RES['base']['measure'], RES['dpo']['log'][-1]
copy = RES['winners']['log'][-1]
one = RES['one_pair']

check(f'{len(TRAIN):,} lines teach, {len(TEST):,} are kept apart, {VOCAB:,} words',
      len(TRAIN) == 22015 and len(TEST) == 2000 and VOCAB == 4000)
check(f"{INFO['train_written']:,} pairs written, {INFO['train_kept']:,} kept (no tie)",
      INFO['train_written'] == 77768 and INFO['train_kept'] == 10593)
check(f"happy lines: start {base['happy_lines']:.3f}, copy the winners {copy['happy_lines']:.3f}, DPO {after['happy_lines']:.3f}",
      near(base['happy_lines'], 0.046, 1e-9) and near(copy['happy_lines'], 0.275, 1e-9) and near(after['happy_lines'], 0.976, 1e-9))
check(f"wins against the start model: copy {copy['win']:.3f}, DPO {after['win']:.3f}",
      near(copy['win'], 0.294, 1e-9) and near(after['win'], 0.965, 1e-9))
check(f"surprise on unseen Shakespeare: {base['perplexity']:.0f}, {RES['winners']['perplexity']:.0f}, {RES['dpo']['perplexity']:.0f}",
      near(base['perplexity'], 189.9, 0.1) and near(RES['winners']['perplexity'], 217.5, 0.1) and near(RES['dpo']['perplexity'], 1767.9, 0.1))
check(f"different words in 1,000 lines: start {base['distinct']}, DPO {after['distinct']}",
      base['distinct'] == 1029 and after['distinct'] == 770)
n = RES['test_pairs']
check(f"hidden judge agrees on {round(RES['dpo']['reward_right'] * n)} of {n} unseen pairs",
      n == 233 and round(RES['dpo']['reward_right'] * n) == 231)
check(f"one pair: gap {one['gap']:.2f}, loss {one['loss_start']:.4f} -> {one['loss_after']:.4f}",
      near(one['gap'], 36.81, 0.01) and near(one['loss_after'], 0.0249, 1e-4))
check('the exercise gives the same loss',
      near(dpo_loss(one['after']['chosen'], one['after']['rejected'],
                    one['start']['chosen'], one['start']['rejected']), one['loss_after'], 1e-9))
check('zero gap gives log 2 for every beta',
      all(near(dpo_loss(-5, -7, -5, -7, b), math.log(2), 1e-12) for b in (0.01, 0.1, 2)))
need = math.log(1 / (math.exp(0.1) - 1))
check(f'loss 0.1 needs beta x gap = {need:.2f}: gap {need / 0.01:.0f} for beta 0.01, {need / 0.1:.1f} for 0.1',
      near(math.log(1 + math.exp(-need)), 0.1, 1e-12) and near(RES['beta_gap']['0.1'], 22.52, 0.01))

# the loss the training used, on the saved models, equals the hand formula
pair = json.loads((FILES / 'pairs-test.json').read_text())[20]
m0, m1 = load(), load('files/dpo.pt')
with torch.no_grad():
    l = float(loss_of(m1, m0, [pair], 'dpo', 0.1))
check(f'dpo.loss_of on the saved models: {l:.4f}', near(l, one['loss_after'], 1e-3))
with torch.no_grad():
    l0 = float(loss_of(m0, m0, [pair], 'dpo', 0.1))
check(f'the start model against itself: {l0:.4f} = log 2', near(l0, math.log(2), 1e-6))

print(f'\n{sum(results)} of {len(results)} checks ok')
