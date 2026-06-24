#!/bin/bash
# GPU 4 — CONTINUE training (resumable) with the corrected max_new=128 eval.
# Run in its own screen:  bash gpu4_train_128.sh
#
# Resumable: run_sweep skips any config whose evaluated/ dir exists, so relaunching
# picks up where it stopped. New configs it finishes are now eval'd at 128 (fresh import
# of the edited sweep.py). NOTE: configs that were already evaluated at 64 before the stop
# are SKIPPED here (eval_done) — lift those to 128 with the GPU 0 reeval script.
#
# Prereq: src/sweep.py must already say  eval_log(..., max_new=128).
set -e
export CUDA_VISIBLE_DEVICES=4
#export CUDA_MPS_PIPE_DIRECTORY=/tmp/dummy_mps
NB=/home/mantovani/repo/generalize_knowledge/notebooks
cd "$NB" || { echo "ABORT: cannot cd to $NB"; exit 1; }

grep -q "max_new=128" ../src/sweep.py || { echo "ABORT: ../src/sweep.py still not max_new=128"; exit 1; }

ARGS="--facts data/facts_attr_v2.json --folds 0 1 2 --seeds 0 --mode interleaved"

# continue config 1 (decl) — already-evaluated configs are skipped; only unfinished ones train
python run.py --exp experiments/exp01_decl.yaml        $ARGS

# config 2 (50/50)
#python run.py --exp experiments/exp01_fic_decl_50.yaml $ARGS

du -sh data/facts_attr_v2/*/fold*/sweep/runs/*/adapter 2>/dev/null | tail -8