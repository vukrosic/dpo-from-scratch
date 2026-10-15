"""The text: tiny Shakespeare, cut into lines of words.

A word is a run of letters (with ' inside, as in "o'er") or one sign: , . ! ? ; :
Every line ends with the sign END, so the model learns where a line stops.
The 4,000 most common words get a number each; any other word becomes '?'.
"""
import re
from collections import Counter
from pathlib import Path

TEXT = Path(__file__).parent / 'files' / 'shakespeare.txt'
VOCAB = 4000
END, UNK = 0, 1
WORD = re.compile(r"[a-z]+(?:'[a-z]+)*|[,.!?;:]")


def split_words(line):
    return WORD.findall(line.lower())


def read_lines():
    """Spoken lines only: the 'NAME:' lines and empty lines are dropped."""
    out = []
    for raw in TEXT.read_text().splitlines():
        raw = raw.strip()
        if not raw or raw.endswith(':'):
            continue
        w = split_words(raw)
        if w:
            out.append(w)
    return out


LINES = read_lines()
_count = Counter(w for line in LINES for w in line)
WORDS = ['END', '?'] + [w for w, _ in _count.most_common(VOCAB - 2)]
NUM = {w: i for i, w in enumerate(WORDS)}


def to_nums(words):
    return [NUM.get(w, UNK) for w in words]


def to_text(nums):
    return ' '.join(WORDS[n] for n in nums if n != END).replace(' ,', ',') \
        .replace(' .', '.').replace(' !', '!').replace(' ?', '?') \
        .replace(' ;', ';').replace(' :', ':')


# The last 2,000 lines are kept apart: their first two words start the
# lines the model writes when it is tested.
TRAIN, TEST = LINES[:-2000], LINES[-2000:]


def stream(lines):
    """All lines one after another, each followed by END."""
    out = []
    for line in lines:
        out += to_nums(line) + [END]
    return out


def prompts(n, lines=TEST):
    """The first two words of n test lines that have at least four words."""
    out = []
    for line in lines:
        if len(line) >= 4 and UNK not in to_nums(line[:2]):
            out.append(to_nums(line[:2]))
        if len(out) == n:
            break
    return out


if __name__ == '__main__':
    print(len(LINES), 'lines,', sum(map(len, LINES)), 'words,',
          len(_count), 'different')
    kept = sum(_count[w] for w in WORDS[2:]) / sum(_count.values())
    print(f'the {VOCAB - 2} most common words cover {kept:.3f} of the text')
    print(LINES[0], to_nums(LINES[0]))
