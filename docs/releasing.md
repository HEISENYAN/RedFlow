# Publishing the Source

The release includes RedFlow training and inference code, reproduction documentation, and the dependency lockfile. Real-robot data, task-specific SFT initialization weights, the RL field generation pipeline, and hardware control programs must be provided separately. They are not included in the source archive. The published clothes-folding RedFlow weights are linked in the [README](../README.md).

## Build a Clean Source Archive

Run these commands in this repository's Git working tree with Python 3.11+. Training dependencies are not required:

```bash
python scripts/prepare_public_release.py --check
python scripts/prepare_public_release.py
```

The script checks tracked files and new, non-ignored files in the current working tree, excluding deleted files. Checks cover Python/shell syntax, local Markdown links, notebook outputs, accidentally included local artifacts, and recognizable credential formats. These checks do not replace review of external data licenses or credentials in formats the scanner does not recognize.

Outputs:

- `dist/redflow-source.zip`: current source without `.git` or historical commits.
- `dist/redflow-source.sha256`: the archive's SHA-256 checksum.
- `SOURCE_MANIFEST.json` inside the archive: SHA-256 checksums for each source file and pinned revisions for optional submodules.

After packaging, the script checks ZIP integrity and compares every source file with the working tree. Rebuild after each change to avoid publishing an outdated snapshot.

## Publish to a New Repository

Extract the archive, create a new Git repository inside the extracted `redflow` directory, and commit the source files. This publishes the cleaned current source. Do not copy the original working directory's `.git` or push its history: earlier commits contained credentials, and cleaning current files does not remove those historical objects. Credential owners must also revoke the old credentials.

The source archive does not include Git submodule index entries or third-party environment source. RedFlow training does not require those environments. For the upstream ALOHA/LIBERO examples, clone the corresponding environments into `third_party` using the URLs in `.gitmodules` and the revisions in `SOURCE_MANIFEST.json`.

Retain `LICENSE`, `LICENSE_GEMMA.txt`, and copyright notices in third-party source. This fork identifies openpi as its upstream project and does not represent an official upstream release.

## Validation Scope

Source and dependency-lock consistency checks do not require a GPU:

```bash
python scripts/prepare_public_release.py --check
uv lock --check
```

RedFlow numerical and data-transform tests require the environment described in the [README](../README.md):

```bash
JAX_PLATFORMS=cpu python -m pytest tests/test_redflow_losses.py tests/test_data_loader_chunked.py -q
```

Set `TEST_REAL_ROLLOUT_DIR` to enable the real-dataset test; otherwise it is skipped. Loading published weights, full training runs, GPU memory requirements, and real-robot performance must be validated with the corresponding resources and hardware. Source checks and synthetic numerical tests do not establish those results. CI runs source checks separately from tests requiring training dependencies.
