#!/usr/bin/env bash
# Install the locked RedFlow environment. Install uv before running this script.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
GIT_LFS_SKIP_SMUDGE=1 uv sync --locked --python 3.11 --no-default-groups --group dev
