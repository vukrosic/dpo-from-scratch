"""One pair by hand, on the CPU, with no training: the two saved models
(files/base.pt, the start; files/dpo.pt, after DPO) score one hidden pair.
Adds 'one_pair' and 'beta_gap' to files/results.json. A few seconds."""
import json
import math

import torch

from data import to_text
from gpt import line_logp
from pairs import load

PAIR = 20          # "i am | glad to see it so." against "one that go would."
BETA = 0.1
sig = lambda z: 1 / (1 + math.exp(-z))


def main():
    res = json.load(open('files/results.json'))
    x = json.load(open('files/pairs-test.json'))[PAIR]
    start, after = load(), load('files/dpo.pt')
    one = {'prompt': to_text(x['prompt']), 'chosen': to_text(x['chosen']),
           'rejected': to_text(x['rejected']), 'scores': x['scores']}
    with torch.no_grad():
        for name, m in (('start', start), ('after', after)):
            one[name] = {k: float(line_logp(m, [x['prompt']], [x[k]])[0]) for k in ('chosen', 'rejected')}
    up = one['after']['chosen'] - one['start']['chosen']          # how much more likely the winner got
    down = one['after']['rejected'] - one['start']['rejected']    # and the loser
    one.update(up=up, down=down, gap=up - down)
    one['loss_start'] = -math.log(sig(0))                          # gap 0: log 2 for every beta
    one['loss_after'] = -math.log(sig(BETA * (up - down)))
    one['chance_start'] = {k: math.exp(v) for k, v in one['start'].items()}
    one['chance_after'] = {k: math.exp(v) for k, v in one['after'].items()}
    res['one_pair'] = one
    # beta: how big the gap must be before this pair's loss is down to 0.1
    z = math.log(1 / (math.exp(0.1) - 1))                         # sig(z) = e^-0.1
    res['beta_gap'] = {str(b): z / b for b in (0.01, 0.1, 0.5, 2.0)}
    json.dump(res, open('files/results.json', 'w'), indent=1)
    print(json.dumps({'one_pair': one, 'beta_gap': res['beta_gap']}, indent=1))


if __name__ == '__main__':
    main()
