# 4. Offline tests with a tiny model and synthetic data

Date: 2026-10-14. Status: accepted.

## Context

The real pipeline downloads Civil Comments (about 1.8 million comments in its train split) and
DistilBERT, and fine-tunes on a GPU for hours. CI runs on CPU runners and should
not depend on the Hugging Face Hub being reachable.

## Decision

- `guardian.tiny` builds a randomly initialised DistilBERT (dimension 16, one
  layer) with a WordPiece vocabulary built from the training texts. The
  vocabulary is constructed directly, because the `tokenizers` trainer breaks
  frequency ties differently between runs.
- `dataset.name: synthetic` generates template sentences in which the positive
  class contains one of two invented marker words. The task is learnable and
  the data contains no real hateful language.
- Unit tests replace the Guardian with `tests/fakes.KeywordClassifier`, Hub
  loaders with fakes, and WordNet with a stub.
- `configs/smoke.yaml` runs the whole CLI pipeline this way in about a minute.

## Consequences

- CI exercises every pipeline step, checkpoint loading and resume logic without
  network or GPU.
- Numbers produced by the smoke config only show that the code runs. They are
  never reported as results.
