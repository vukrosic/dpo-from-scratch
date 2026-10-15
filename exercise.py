"""Exercise: the DPO loss for one pair, by hand.

Write dpo_loss. The four numbers are log chances of a whole line (the sum of
the log chances of its words): of the winner and the loser, by the frozen copy
of the start model and by our model after DPO. Guess first: at the start our
model is the copy, so the gap is 0. What is the loss then?

Then run:  python3 exercise.py
"""
import math


def dpo_loss(win, lose, win_copy, lose_copy, beta=0.1):
    # your part starts here
    gap = (win - win_copy) - (lose - lose_copy)    # winner up minus loser up
    return math.log(1 + math.exp(-beta * gap))
    # and ends here


if __name__ == '__main__':
    # "i am | glad to see it so." against "i am | one that go would." (files/results.json)
    copy_win, copy_lose = -10.318, -22.809
    ours_win, ours_lose = -14.998, -64.303
    start = dpo_loss(copy_win, copy_lose, copy_win, copy_lose)
    after = dpo_loss(ours_win, ours_lose, copy_win, copy_lose)
    print(f'at the start (our model = the copy): loss {start:.4f}   log 2 = {math.log(2):.4f}')
    print(f'after DPO: gap {(ours_win - copy_win) - (ours_lose - copy_lose):.2f}, loss {after:.4f}')
    print('right' if abs(start - math.log(2)) < 1e-9 and abs(after - 0.0249) < 1e-3 else 'not right yet')
