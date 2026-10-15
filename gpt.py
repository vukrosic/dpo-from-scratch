"""A small GPT: it reads words left to right and gives each possible next
word a chance. 4 blocks, 4 heads, lists of 128 numbers, 32 words at a time."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import VOCAB, END

CTX = 32
torch.set_num_threads(4)


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, mask):
        h = self.n1(x)
        x = x + self.att(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class GPT(nn.Module):
    def __init__(self, d=128, heads=4, blocks=4):
        super().__init__()
        self.word = nn.Embedding(VOCAB, d)
        self.place = nn.Embedding(CTX, d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(blocks))
        self.norm = nn.LayerNorm(d)
        nn.init.normal_(self.word.weight, std=0.02)   # small start: no wild first guesses
        nn.init.normal_(self.place.weight, std=0.02)

    def forward(self, x):
        T = x.shape[1]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
        h = self.word(x) + self.place(torch.arange(T))
        for b in self.blocks:
            h = b(h, mask)
        return self.norm(h) @ self.word.weight.T      # one score per word


def pad(seqs):
    """Rows of different length -> one table, END after the end."""
    L = max(map(len, seqs))
    x = torch.full((len(seqs), L), END, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s)
    return x


def line_logp(model, prompts, lines):
    """For each prompt + line: the sum of the log chances of the line's words
    (the prompt words are given, not judged). One number per line."""
    seqs = [p + l for p, l in zip(prompts, lines)]
    x = pad(seqs)
    logp = F.log_softmax(model(x[:, :-1]), -1)
    got = logp.gather(2, x[:, 1:, None])[:, :, 0]
    keep = torch.zeros_like(got)
    for i, (p, l) in enumerate(zip(prompts, lines)):
        keep[i, len(p) - 1:len(p) - 1 + len(l)] = 1
    return (got * keep).sum(1)


@torch.no_grad()
def write(model, prompts, longest=16, dice=None, temp=1.0):
    """Continue each prompt one word at a time, drawing each word by its
    chance, until END. Returns the new words of each line (END included)."""
    x = pad(prompts)
    lens = [len(p) for p in prompts]
    assert len(set(lens)) == 1, 'prompts of equal length'
    out = [[] for _ in prompts]
    done = torch.zeros(len(prompts), dtype=torch.bool)
    for _ in range(longest):
        logits = model(x[:, -CTX:])[:, -1] / temp
        nxt = torch.multinomial(F.softmax(logits, -1), 1, generator=dice)[:, 0]
        for i in range(len(prompts)):
            if not done[i]:
                out[i].append(int(nxt[i]))
        done |= nxt == END
        x = torch.cat([x, nxt[:, None]], 1)
        if done.all():
            break
    for o in out:
        if o[-1] != END:
            o.append(END)
    return out
