# 1. Package layout, uv, and torch builds as extras

Date: 2026-10-14. Status: accepted.

## Context

The course code was a top-level package literally named `src`, installed with
an unpinned `requirements.txt`. Results depended on whatever versions pip
resolved on the day. torch needs a different wheel for CPU-only CI runners
(about 200 MB) and for CUDA machines (several GB), and plain PyPI only serves
the CUDA build on Linux.

## Decision

- Package `red_queen` under `src/`, built with hatchling, described in
  `pyproject.toml`, locked in `uv.lock`. Python 3.12 is pinned for development;
  3.11 to 3.13 are supported and tested.
- torch is not a base dependency. Two mutually exclusive extras pick the build:
  `cpu` (PyTorch CPU index) and `cu126` (CUDA 12.6 index). `tool.uv.conflicts`
  lets one lockfile hold both resolutions.

## Consequences

- `uv sync --extra cpu` reproduces the exact environment CI tests.
- Installing without an extra gives a package that cannot import torch; the
  README and Makefile always pass one.
- The CPU wheel has a local version (`2.14.1+cpu`) that pip-audit cannot look up,
  so the audit step strips the suffix and audits the same release.
