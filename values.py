"""Compact data for the film's pictures, on the CPU, with no training: reads
files/results.json, adds the word by word chances of the worked pair and the
start model's first choices after "i am". Writes ../src/values.json."""
import json

import torch
import torch.nn.functional as F

from data import WORDS, END
from gpt import pad
from pairs import load
from judge import HAPPY, SAD

PAIR = 20


@torch.no_grad()
def word_chances(model, prompt, line):
    x = pad([prompt + line])
    p = F.softmax(model(x[:, :-1]), -1)[0]
    return [{'w': WORDS[n] if n != END else 'END', 'p': float(p[len(prompt) - 1 + i, n])} for i, n in enumerate(line)]


@torch.no_grad()
def first_choices(model, prompt, k=6):
    p = F.softmax(model(pad([prompt]))[0, -1], -1)
    top = torch.topk(p, k)
    return [{'w': WORDS[int(i)], 'p': float(v)} for v, i in zip(top.values, top.indices)]


def main():
    R = json.load(open('files/results.json'))
    x = json.load(open('files/pairs-test.json'))[PAIR]
    start, after = load(), load('files/dpo.pt')
    keep = lambda m: {k: m[k] for k in ('score', 'happy_lines', 'sad_lines', 'win', 'tie', 'distinct', 'step')}
    out = {
        'pairs': R['pairs'], 'test_pairs': R['test_pairs'],
        'info': json.load(open('files/pairs-info.json')),
        'base': {**keep(R['base']['measure']), 'perplexity': R['base']['measure']['perplexity'], 'lines': R['base']['lines']},
        'winners': {'log': [keep(m) for m in R['winners']['log']], 'perplexity': R['winners']['perplexity'], 'lines': R['winners']['lines']},
        'dpo': {'log': [keep(m) for m in R['dpo']['log']], 'perplexity': R['dpo']['perplexity'], 'lines': R['dpo']['lines'],
                'reward_right': R['dpo']['reward_right']},
        'one_pair': R['one_pair'], 'beta_gap': R['beta_gap'],
        'words': {name: {k: word_chances(m, x['prompt'], x[k]) for k in ('chosen', 'rejected')}
                  for name, m in (('start', start), ('after', after))},
        'first': {name: first_choices(m, x['prompt']) for name, m in (('start', start), ('after', after))},
        'happy': HAPPY, 'sad': SAD,
        'base_log': json.load(open('files/base.json'))['log'],
    }
    json.dump(out, open('../src/values.json', 'w'))
    print(json.dumps({k: out[k] for k in ('words', 'first')}, indent=1))


if __name__ == '__main__':
    main()
