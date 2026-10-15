#!/bin/sh
# Runs every step that needs no training, then the checks and the exercise.
# The slow steps (train_base.py, pairs.py, run.py) saved their results in
# files/; run them only to redo the training from the start.
set -e
cd "$(dirname "$0")"
PY=${PY:-python3}
for f in data.py judge.py one_pair.py tests.py exercise.py; do
  echo
  echo "=== $f ==="
  $PY -W ignore $f
done
