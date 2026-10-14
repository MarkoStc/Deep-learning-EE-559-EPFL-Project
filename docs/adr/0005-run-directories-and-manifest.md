# 5. Run directories with an artifact manifest

Date: 2026-10-14. Status: accepted.

## Context

Each course script created `outputs/run_<len(existing)+1>` and the next script
needed that path typed by hand. The Makefile and shell scripts hardcoded
`run_001` to `run_003` while the real runs were `run_029` to `run_031`.
Deleting one directory made the counter hand out an existing number again.

## Decision

- `RunDirectory.create` names runs `<YYYYmmdd-HHMMSS>-<label>` and adds a
  numeric suffix instead of ever reusing a directory.
- Each run has `config.yaml`, `metadata.json` (git commit, package versions) and
  `manifest.json`, which maps artifact names (`guardian`, `evader`,
  `stress_train`, `stress_eval`, `robust`, `evaluation`, `outer_loop`) to paths
  inside the run.
- Steps read inputs from the manifest. A step whose artifacts are already
  recorded is skipped unless `--force` is given.

## Consequences

- `red-queen <step> --run-dir runs/<id>` is the only path anyone types.
- A crashed pipeline is resumed by running the same command again; Guardian
  training additionally resumes from its last finished epoch.
- `runs/` is git-ignored. The course run stays archived under `outputs/`.
