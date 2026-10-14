# Contributing

## Setup

You need [uv](https://docs.astral.sh/uv/) and git. uv installs the pinned Python (3.12) itself.

```bash
make install            # uv sync --extra cpu --extra tensorboard, then pre-commit install
make install TORCH=cu126  # same with CUDA 12.6 torch wheels
```

## Before you push

```bash
make check   # ruff lint + format check, strict mypy, full test suite with coverage
make smoke   # whole pipeline on synthetic data with a tiny model, about a minute on CPU
```

`make test-fast` skips the end-to-end tests marked `slow`. Run one test with
`uv run pytest tests/test_config.py::test_unknown_key_is_rejected`.

Tests never touch the network or a GPU. Hub downloads are injected and faked
(see `tests/test_data.py`), the Guardian is replaced by `tests/fakes.py` or a
tiny offline model (`guardian.tiny` in the config), and WordNet is replaced by a
stub synonym lookup.

## Code conventions

- One YAML config, validated by `red_queen.config.AppConfig`. A new
  hyperparameter goes into the config model and `configs/default.yaml`, not into
  a module constant or a function default.
- Classes take their collaborators (classifier, RNG, tracker, data source) as
  constructor arguments, so tests can pass fakes.
- Every random draw goes through an explicit `random.Random` or
  `torch.Generator` from `red_queen.runtime.make_rng`, never the global RNG.
- Log with `logging.getLogger(__name__)` and structured fields
  (`extra=fields(...)`), not `print`. Only the CLI prints results.
- Expected failures raise a subclass of `RedQueenError`; the CLI turns them into
  exit code 2.
- Never log raw hateful text. Use `red_queen.text_safety.sanitize_text`.

## Commits and pull requests

- One branch per change: `feat/`, `fix/`, `refactor/`, `test/`, `docs/`, `ci/`,
  `build/`, `chore/`.
- [Conventional Commits](https://www.conventionalcommits.org/) messages, for
  example `fix: evaluate on a held-out stress set`.
- Each PR links its issue, says what changed, why, and how it was tested, and
  updates `CHANGELOG.md` under *Unreleased* when behaviour changes.
- A number in the README must come from a script in this repo, with the command
  that reproduces it.

## Design decisions

Significant decisions are recorded in [`docs/adr/`](docs/adr/). Add a new ADR
when a change alters the protocol, the config schema or the package layout.
