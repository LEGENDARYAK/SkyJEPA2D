#!/bin/bash
set -eu
cd -- "$(dirname -- "$0")"
if [[ ! -x .venv/bin/python ]]; then
  printf 'Complete the four installation commands in README.md first.\n'
  read -r -p 'Press Return to close.'
  exit 1
fi
exec .venv/bin/python -m skyjepa2d play "$@"
