"""Step 1: the comparisons. For each prompt (the first two words of a real
line) the start model writes two lines. The judge says which one it prefers.
Ties are thrown away. Saves files/pairs.json and files/pairs-test.json.

Pairs from TRAIN prompts teach; pairs from TEST prompts only check."""
import json

import torch

from data import TRAIN, TEST, prompts, to_text
from gpt import GPT, write
from judge import prefer, score

ROUNDS = 4          # every prompt is used this many times
BATCH = 500


def load(path='files/base.pt'):
    m = GPT()
    m.load_state_dict(torch.load(path))
    return m.eval()


def make(model, ps, rounds, dice):
    out, seen = [], 0
    for r in range(rounds):
        for i in range(0, len(ps), BATCH):
            p = ps[i:i + BATCH]
            a = write(model, p, dice=dice)
            b = write(model, p, dice=dice)
            for pi, ai, bi in zip(p, a, b):
                seen += 1
                w = prefer(ai, bi)
                if w is None:
                    continue
                c, l = (ai, bi) if w == 'a' else (bi, ai)
                out.append({'prompt': pi, 'chosen': c, 'rejected': l,
                            'scores': [score(c), score(l)]})
        print(f'round {r + 1}: {seen} pairs written, {len(out)} kept', flush=True)
    return out, seen


def main():
    torch.manual_seed(1)
    dice = torch.Generator().manual_seed(1)
    model = load()
    train_p = prompts(10 ** 6, TRAIN)
    test_p = prompts(10 ** 6, TEST)
    print(len(train_p), 'train prompts,', len(test_p), 'test prompts')
    tr, n_tr = make(model, train_p, ROUNDS, dice)
    te, n_te = make(model, test_p, 1, dice)
    json.dump(tr, open('files/pairs.json', 'w'))
    json.dump(te, open('files/pairs-test.json', 'w'))
    json.dump({'train_written': n_tr, 'train_kept': len(tr),
               'test_written': n_te, 'test_kept': len(te),
               'examples': [{'prompt': to_text(x['prompt']),
                             'chosen': to_text(x['chosen']),
                             'rejected': to_text(x['rejected']),
                             'scores': x['scores']} for x in tr[:20]]},
              open('files/pairs-info.json', 'w'), indent=1)
    for x in tr[:5]:
        print(to_text(x['prompt']), '|', to_text(x['chosen']), '>',
              to_text(x['rejected']), x['scores'])


if __name__ == '__main__':
    main()
