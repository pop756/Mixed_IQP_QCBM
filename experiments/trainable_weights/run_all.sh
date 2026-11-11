#!/bin/bash
# The trainable-weight campaign: every (dataset, a >= 1) of the main campaign with trained branch weights.
# Finished groups (runs/<ds>/a<a>_tw.json present) are skipped. usage: ./run_all.sh   (log: logs/run_all.log)
cd "$(dirname "$0")"
PY=${PY:-python}
export XLA_PYTHON_CLIENT_PREALLOCATE=false TMPDIR=${TMPDIR:-/tmp}
mkdir -p logs
run() { [ -f runs/$1/a$2_tw.json ] && { echo "skip $1 a$2"; return; }
        $PY -u -W error::RuntimeWarning train_tw.py $1 $2 2>&1 | grep --line-buffered -v "Warning\|^WARNING" > logs/train_$1_a$2.log
        echo "$1 a$2 rc=${PIPESTATUS[0]} $(date -u +%T) $(tail -1 logs/train_$1_a$2.log)"; }
for ds in blobs ising; do for a in 1 2 3; do run $ds $a; done; done
for a in 1 2 3; do run dwave $a; done
for a in 1 2 3 4; do run mnist $a; done
echo ALLDONE
