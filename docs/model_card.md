# Model card: Guardian and Evader

This repository trains two models. Neither is released as a product, and
neither should moderate real users' content.

## Guardian (classifier under attack)

| | |
|---|---|
| Architecture | `distilbert-base-uncased` with a 2-class sequence-classification head |
| Task | Binary label per comment: 0 safe, 1 hate (Civil Comments toxicity >= 0.5) |
| Training data | 50,000-comment stratified subsample of Civil Comments, 70/15/15 train/val/test split, seed 42 |
| Training | AdamW, lr 2e-5, weight decay 0.01, 3 epochs, batch 32, linear warmup 10%, max length 128, gradient clip 1.0; best epoch by validation macro-F1 |
| Robust variant | Baseline Guardian fine-tuned 2 more epochs (lr 1e-5) on the train split plus 50% of the Evader's stress cases from training texts, all labelled 1 |

### Results from the original course run

From `outputs/evaluation/summary_table.csv` (run of 2026-03-31 to 2026-04-09 on
the EPFL RCP cluster, code at tag `v0-original`):

| Model | Clean test macro-F1 | Clean hate F1 | Clean hate precision | Clean hate recall |
|---|---|---|---|---|
| Baseline | 0.8098 | 0.6486 | 0.6820 | 0.6183 |
| Robust | 0.8041 | 0.6373 | 0.6907 | 0.5917 |

Stress-set numbers from that run are not repeated here: the robust model was
evaluated on stress cases it had partly been trained on
([ADR 0003](adr/0003-held-out-stress-evaluation.md)). The current pipeline
evaluates on a held-out stress set; it has not yet been rerun on Civil Comments
(see the README).

### Limits

- Civil Comments labels toxicity, which is broader than hate speech. A comment
  can be labelled 1 for insults or profanity that target no group.
- Hate recall on clean text is about 0.62: the baseline misses more than a third
  of positive comments before any attack.
- English only, comments from one news-comment platform,
  truncated to 128 tokens.
- Identity-subgroup F1 is computed when Civil Comments identity columns are
  present; subgroups with fewer than 10 test examples are skipped, and small
  subgroups have wide uncertainty.

## Evader (attacker)

| | |
|---|---|
| Type | Contextual bandit, one decision per text |
| State | 8 handcrafted features: length, word count, uppercase share, punctuation share, emoji flag, URL flag, mean word length, Guardian P(hate) |
| Actions | noop, typo, leetspeak, zero-width whitespace, Cyrillic homoglyph, WordNet synonym, benign emoji |
| Policy | MLP 8 -> 64 -> 64 -> 7, softmax |
| Training | REINFORCE, Adam lr 1e-3, 2000 episodes of 64 hateful training texts, running-mean baseline (decay 0.99), entropy bonus 0.01; reward 1 when the Guardian labels the perturbed text safe |

### Behaviour in the original run

The trained policy chose leetspeak for 100% of the 2,798 hateful training
texts (`outputs/stress_cases/stress_case_metrics.json`). A policy that ignores
its input is equivalent to always applying one action, so the handcrafted
features carried no useful signal in that run. The current pipeline reports the
policy next to a brute-force baseline (every action on every test text) so this
is visible in every evaluation.

### Limits

- Actions are fixed surface edits. The Evader cannot paraphrase, and it has no
  constraint that a perturbed text still reads the same to a human (leetspeak at
  40% per character is often hard to read).
- One action per text; no combinations.

## Ethical notes

- Perturbations only rewrite characters or words of texts that are already in
  the dataset and already labelled. The code does not generate new hateful
  content.
- Logged and sampled examples are masked with `red_queen.text_safety`. The
  stress-case parquet files keep raw text because retraining needs it; they are
  written to git-ignored run directories.
- The course-era file `outputs/stress_cases/stress_cases.parquet` is in the
  public git history and contains unmasked hateful comments from Civil Comments.
- A system that learns which edits evade a moderation model can be misused to
  evade one. The action space here is a list of well-known evasions already
  covered in the adversarial NLP literature, and the code targets a model trained
  in this repo, not a deployed service.
