# Benchmark — ArSarcasm-v2 test set (3,000 tweets)

## 1. Pretrained vs Fine-tuned

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Pretrained (base) | 0.655 | 0.608 |
| Fine-tuned (ours) | 0.672 | 0.630 |

**Fine-tuning improved macro-F1 by +0.022 (+2.2 points).**


## 2. Per-dialect performance (Fine-tuned)

| Dialect | N | Accuracy | Macro-F1 |
|---|---|---|---|
| egypt | 306 | 0.621 | 0.453 |
| gulf | 322 | 0.618 | 0.600 |
| levant | 47 | 0.617 | 0.482 |
| msa | 2323 | 0.687 | 0.650 |

## 3. Sarcasm impact (Fine-tuned)

| Subset | N | Accuracy |
|---|---|---|
| Non-sarcastic | 2179 | 0.641 |
| Sarcastic | 821 | 0.754 |

**Note:** on this test set the model scores 11.3 points *higher* on sarcastic tweets (0.754) than non-sarcastic (0.641). The most likely reason is label distribution: sarcastic tweets in ArSarcasm skew heavily toward *negative* sentiment, and the model is strongest on the negative class, so raw accuracy looks higher there. This is an accuracy artifact, not evidence that sarcasm is easy — a per-class F1 breakdown would show the true picture.
