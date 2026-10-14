# 3. Evaluate on a held-out stress set

Date: 2026-10-14. Status: accepted.

## Context

In the course pipeline the Evader perturbed hateful **training** texts, the
robust Guardian was retrained on a random 50% of those perturbed texts, and
evaluation scored the robust Guardian on all of them. About half of the
"stress" examples were therefore training examples. The original run reports a
stress recall of 0.9964 for the robust model (`outputs/evaluation/summary_table.csv`).
That number cannot be read as robustness to new attacks.

Two smaller issues: macro-F1 was reported on the stress set, which contains
only one class, and attack success counted texts the Guardian already missed
before any perturbation.

## Decision

- The stress step writes two sets. `stress/train` comes from hateful training
  texts and feeds robust retraining. `stress/eval` comes from hateful **test**
  texts, uses a separate RNG stream, and is never trained on.
- Both Guardians are scored on the same `stress/eval` texts (a transfer attack:
  the Evader was trained against the baseline Guardian).
- Stress results report hate recall, raw ASR and conditional ASR (only texts the
  model detected before the attack). No macro-F1.
- The report includes a brute-force baseline (every action on every test text)
  and, under `in_sample_train_stress_do_not_report`, the leaky number for
  comparison.

## Consequences

- New results are not comparable one-to-one with the course report's stress
  numbers.
- Re-scoring the course checkpoints this way (`red-queen compare`, 2026-10-14)
  gave a held-out leetspeak recall of 0.9967 for the robust model, close to the
  leaky 0.9964: for that action the leak did not change the conclusion. The
  brute-force baseline showed what the old protocol hid: homoglyphs and typos
  still evade the robust model.
- Robustness to an Evader retrained against the robust model (an adaptive
  attack) is measured by `red-queen arms-race`, not by `evaluate`.
