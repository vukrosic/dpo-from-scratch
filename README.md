# DPO from scratch

Code for the video **How AI Learns Which Answer You Prefer (DPO From Scratch)**.
A judge picks the better of two lines. DPO (Direct Preference Optimization) teaches a model from those picks with one loss: raise the chance of the winner, lower the chance of the loser, both measured against a frozen copy of the start model. This code builds a small GPT on Shakespeare, lets a word-counting judge make 10,593 picks, teaches the model two ways (copy the winners, DPO), and measures what each one gained and what it cost.
The method: Rafailov, Sharma, Mitchell, Ermon, Manning and Finn, "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", 2023.

```bash
pip install torch
sh run_all.sh         # the no-training steps, the checks and the exercise; expected output is in demo-output.txt
```

`run_all.sh` takes a few seconds on a CPU. The training is already done: the two models (`files/base.pt`, `files/dpo.pt`), the picks and every measurement are saved in `files/`. To redo it from the start, run `train_base.py`, `pairs.py` and `run.py` in that order; that takes hours.

```python
gap = (win - win_copy) - (lose - lose_copy)   # winner up minus loser up, against the frozen copy
loss = log(1 + e^(-beta * gap))               # log 2 at the start, near 0 once the gap is big
```

## Results

| | Start model | Copy the winners | DPO, beta 0.1 |
| --- | --- | --- | --- |
| happy lines out of 1,000 | 46 | 275 | 976 |
| wins against the start model | | 29.4 % | 96.5 % |
| surprise on unseen Shakespeare (perplexity) | 190 | 218 | 1,768 |
| different words in 1,000 lines | 1,029 | 1,020 | 770 |

On 233 picks it never saw, the DPO model's hidden score, beta × (its log chance − the copy's), agrees with the judge 231 times.

## Words

| In the video | Usual name | In the code |
| --- | --- | --- |
| the judge | stand-in for people: happy words minus sad words | `judge.score`, `judge.prefer` |
| a pick, a pair | preference pair: prompt, chosen line, rejected line | `files/pairs.json` |
| start model | the model before teaching (SFT model in the paper) | `files/base.pt` |
| frozen copy | reference model | `ref` in `dpo.teach` |
| log chance of a line | sum of log probabilities of its words | `gpt.line_logp` |
| copy the winners | supervised fine-tuning on the chosen lines | mode `winners` |
| surprise | perplexity | `dpo.perplexity` |
| hidden score | the implicit reward | `dpo.reward_check` |

## Files

| File | What it does |
| --- | --- |
| `data.py` | Tiny Shakespeare cut into 24,015 spoken lines; 4,000 word numbers; the last 2,000 lines are kept apart. |
| `gpt.py` | A small GPT: 4 blocks, 4 heads, 128 numbers per word, 32 words at a time; 1,309,440 numbers. |
| `train_base.py` | The start model: next-word guessing, 3,000 steps. |
| `judge.py` | 18 happy and 18 sad words; the line with the higher count wins, equal counts are thrown away. |
| `pairs.py` | The start model writes two lines per opening; 77,768 pairs written, 10,593 kept; 233 more from unseen openings. |
| `dpo.py` | The two ways to teach, the DPO loss, and the measurements. |
| `run.py` | Teaches both ways: 600 steps of 64 pairs, learning rate 0.0001, beta 0.1. |
| `one_pair.py` | One pair by hand on the saved models, and the gap each beta needs. |
| `values.py` | Numbers for the pictures of the video. Not part of `run_all.sh`. |
| `tests.py` | 13 checks of the claims. |
| `exercise.py` | Write `dpo_loss`. |

## What this is not

- **Not people.** The judge counts words. DPO learned to stuff lines with happy words, and the lines barely make sense.
- **Not a big model.** 4,000 words, Shakespeare only. The start model is overfit: its loss on unseen lines rises after step 1,000 (`files/base.out`).
- **Only beta 0.1 was run.** The gap each beta needs is arithmetic on the loss (`one_pair.py`), not a training run. No scoring model (the RLHF route) was built, so DPO is not compared with it here.

## Exercise

`exercise.py`: write `dpo_loss` from four log chances: winner and loser, by the frozen copy and by our model. Guess first: what is the loss when our model is still the copy? The answer is log 2 = 0.693; after DPO the pair in the file gives 0.025.

MIT License. The text is Tiny Shakespeare from Andrej Karpathy's char-rnn.
