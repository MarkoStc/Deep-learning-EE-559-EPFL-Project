# 2. One typed YAML config

Date: 2026-10-14. Status: accepted.

## Context

The course code passed a nested `dict` everywhere, read keys with
`cfg["a"]["b"]` or `.get(key, default)`, and kept several values as constants
inside modules (whitespace probability, batch sizes, clipping norm). A typo in
the YAML silently fell back to a default. Some config keys (`gamma`,
`num_actions`, `checkpoint_dir`) were never read.

## Decision

- `red_queen.config.AppConfig` is a tree of frozen pydantic models, one per
  section, with `extra="forbid"` and value ranges.
- `configs/default.yaml` holds the course hyperparameters. Every tunable number
  lives in the config model, not in modules.
- `--set section.key=value` overrides are parsed as YAML and validated like the
  file.
- The resolved config is saved to every run directory and next to every
  Guardian checkpoint (`red_queen_config.yaml`).

## Consequences

- An unknown key or out-of-range value stops the run before any download.
- Derived values (state size, number of actions) come from code
  (`features.STATE_DIM`, the perturbation set) and cannot disagree with it.
- Old course configs do not load; their values are in `configs/default.yaml`.
