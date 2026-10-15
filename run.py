"""Step 3: teach the start model two ways (copy the winners, DPO) with beta
0.1, and check the hidden judge inside the DPO model. Saves files/results.json
and files/dpo.pt. This is the slow step; the saved files are
in the repo, so nothing here has to run again."""
import json
import os

import torch

from data import TEST, stream, to_text
from dpo import TEST_PROMPTS, lines_of, measure, perplexity, reward_check, teach
from pairs import load
from judge import score

STEPS = 600
CHECK_AT = (25, 50, 100, 200, 400, 600)


def show(model, n=6):
    out = []
    for p, l in zip(TEST_PROMPTS[:n], lines_of(model)[:n]):
        out.append({'line': to_text(p + l), 'score': score(l)})
    return out


def main():
    base = load()
    pairs = json.load(open('files/pairs.json'))
    test_pairs = json.load(open('files/pairs-test.json'))
    real = stream(TEST)
    base_lines = lines_of(base, seed=8)
    res = {'pairs': len(pairs), 'test_pairs': len(test_pairs)}
    if os.path.exists('files/results.json'):     # resume: keep the parts already done
        res = json.load(open('files/results.json'))

    if 'base' not in res:
        m0 = measure(base, base, base_lines)
        m0.update(step=0, perplexity=perplexity(base, real))
        res['base'] = {'measure': m0, 'lines': show(base)}
        print('base', m0, flush=True)

    check = lambda m: measure(m, base, base_lines)
    for mode in ('winners', 'dpo'):
        if mode in res:
            continue
        model, log = teach(base, pairs, mode, beta=0.1, steps=STEPS,
                           check_at=CHECK_AT, check=check)
        res[mode] = {'log': log, 'perplexity': perplexity(model, real),
                     'lines': show(model)}
        if mode == 'dpo':
            res[mode]['reward_right'] = reward_check(model, base, test_pairs, 0.1)
            torch.save(model.state_dict(), 'files/dpo.pt')
        json.dump(res, open('files/results.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
