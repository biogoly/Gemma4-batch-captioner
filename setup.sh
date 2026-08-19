#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
[[ -f server_config.toml ]] || cp server_config.example.toml server_config.toml
mkdir -p input output logs
printf '%s\n' "Setup complete. Edit server_config.toml before running ./run_all.sh."
