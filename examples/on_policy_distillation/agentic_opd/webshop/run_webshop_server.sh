#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Starts the per-node WebShop env server (app/server.py). Called once per node by
# run_agent_app.sh's flock bootstrap (NOT once per session). Holds one shared
# SimServer (several-GB catalog + Lucene index) and serves /reset /step /close /
# health over localhost HTTP (see DESIGN.md §3, §6.1).
#
# CONDA_HOME / WEBSHOP_CONDA_ENV / WEBSHOP_HOME / WEBSHOP_PORT are forwarded from
# the training script via --agent-env (multi-node safe); defaults are a
# single-node fallback.

set -eu

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "${SCRIPT_DIR}"

# Activate the WebShop conda environment (Python 3.10; see README.md for setup).
CONDA_HOME="${CONDA_HOME:-/root/miniconda3}"
source "${CONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${WEBSHOP_CONDA_ENV:-relax-opd-webshop}"

# WebShop simulator source (dir containing web_agent_site); server.py adds it to
# sys.path and loads the catalog it points at (README installs the 1000 subset).
export WEBSHOP_HOME="${WEBSHOP_HOME:-/root/WebShop}"
export WEBSHOP_PORT="${WEBSHOP_PORT:-36001}"

exec python -m app.server
