#!/bin/bash
# The whole rerun in order: clusters (if missing), then every training group, then the analysis. Finished groups
# (runs/<ds>/a<a>_<variant>.json present) are skipped. usage: ./run_all.sh   (log: logs/run_all.log)
cd "$(dirname "$0")"
PY=${PY:-python}
export XLA_PYTHON_CLIENT_PREALLOCATE=false TMPDIR=${TMPDIR:-/tmp}
mkdir -p logs
[ -f clusters/mnist/a4_seed2025.npz ] || $PY -u clusters.py 2>&1 | grep -v Warning > logs/clusters.log
run() { [ -f runs/$1/a$2_$3.json ] && { echo "skip $1 a$2 $3"; return; }
        $PY -u -W error::RuntimeWarning train.py $1 $2 $3 2>&1 | grep --line-buffered -v "Warning\|^WARNING" > logs/train_$1_a$2_$3.log
        echo "$1 a$2 $3 rc=${PIPESTATUS[0]} $(date -u +%T) $(tail -1 logs/train_$1_a$2_$3.log)"; }
for ds in blobs ising; do for a in 0 1 2 3; do run $ds $a main; run $ds $a fc1; done; done
run blobs 1 eq; run blobs 2 eq; run ising 2 eq; run ising 3 eq
for a in 0 1 2 3; do run dwave $a main; run dwave $a fc1; done; run dwave 3 eq
for a in 0 1 2 3 4; do run mnist $a main; run mnist $a fc1; done; run mnist 4 eq
$PY -u analyze.py 2>&1 | grep -v Warning > logs/analyze.log; echo "analyze rc=$? $(date -u +%T)"
echo ALLDONE
