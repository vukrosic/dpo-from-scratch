"""Step 0: the start model. It learns to guess the next word of Shakespeare.
Saves files/base.pt. It took 36 minutes on an Apple laptop (files/base.out)."""
import json
import time

import torch
import torch.nn.functional as F

from data import TRAIN, TEST, stream
from gpt import GPT, CTX

STEPS, BATCH = 3000, 64


def batches(nums, dice):
    t = torch.tensor(nums)
    while True:
        i = torch.randint(0, len(t) - CTX - 1, (BATCH,), generator=dice)
        x = torch.stack([t[j:j + CTX + 1] for j in i])
        yield x[:, :-1], x[:, 1:]


@torch.no_grad()
def test_loss(model, nums):
    rows = (len(nums) - 1) // CTX
    t = torch.tensor(nums[:CTX * rows + 1])
    x = t[:-1].view(-1, CTX)
    y = t[1:].view(-1, CTX)
    return float(F.cross_entropy(model(x).flatten(0, 1), y.flatten()))


def main():
    torch.manual_seed(0)
    dice = torch.Generator().manual_seed(0)
    model = GPT()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=STEPS)
    test = stream(TEST)
    log, t0 = [], time.time()
    for step, (x, y) in zip(range(1, STEPS + 1), batches(stream(TRAIN), dice)):
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0 or step == 1:
            tl = test_loss(model, test)
            log.append({'step': step, 'train': float(loss), 'test': tl})
            print(f'step {step:5d}  loss {float(loss):.2f}  test {tl:.2f}  '
                  f'{time.time() - t0:.0f} s', flush=True)
    n = sum(p.numel() for p in model.parameters())
    torch.save(model.state_dict(), 'files/base.pt')
    json.dump({'log': log, 'numbers': n}, open('files/base.json', 'w'))
    print(n, 'numbers in the model')


if __name__ == '__main__':
    main()
