"""The stand-in for people. We have no people to ask which of two lines
they like better, so a fixed rule answers: count the happy words, take away
the sad words. The line with the higher count wins; equal counts: no answer.
"""
from data import NUM, split_words

HAPPY = ['love', 'joy', 'sweet', 'fair', 'happy', 'merry', 'gentle', 'kind',
         'peace', 'hope', 'glad', 'smile', 'bless', 'heaven', 'delight',
         'friend', 'dear', 'laugh']
SAD = ['death', 'die', 'dead', 'grief', 'woe', 'blood', 'hate', 'kill',
       'tears', 'sorrow', 'curse', 'weep', 'fear', 'grave', 'murder',
       'poison', 'sad', 'cruel']
H = {NUM[w] for w in HAPPY}
S = {NUM[w] for w in SAD}
assert len(H) == len(HAPPY) and len(S) == len(SAD), 'every word is in the vocabulary'


def score(nums):
    return sum(n in H for n in nums) - sum(n in S for n in nums)


def prefer(a, b):
    """'a', 'b' or None (a tie)."""
    sa, sb = score(a), score(b)
    return None if sa == sb else ('a' if sa > sb else 'b')


def score_text(text):
    return score([NUM.get(w, 1) for w in split_words(text)])


if __name__ == '__main__':
    from data import LINES
    from collections import Counter
    c = Counter(w for line in LINES for w in line)
    print('happy', [(w, c[w]) for w in HAPPY])
    print('sad', [(w, c[w]) for w in SAD])
    s = Counter(score([NUM.get(w, 1) for w in l]) for l in LINES)
    print('real lines by score', sorted(s.items()))
