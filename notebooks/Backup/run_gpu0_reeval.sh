#!/bin/bash
# GPU 0 — RE-EVAL saved adapters at max_new=128. No training, no manual deletion.
# Run in its own screen:  bash gpu0_reeval_128.sh
#
# Why no rm: --reeval --force OVERWRITES raw_answer/, evaluated/, summary.json and the
# flat fold*_seed*_*.json in place. Deleting folders first risks nuking an adapter by a
# glob slip and buys nothing. eval_sweep re-evals EVERY runs/* in each fold (all + 32
# singles + base), not only the 32 — that's correct for uniform 128 scoring.
#
# Prereq: src/sweep.py must already say  eval_log(..., max_new=128).
#         Verify first:  grep "max_new=128" src/sweep.py
#
# SAFETY: only re-eval folds whose training is FINISHED. Right now GPU 4 is stopped,
# so everything on disk is safe. Once you relaunch GPU 4 training, restrict --folds here
# to folds GPU 4 is NOT touching.
set -e
export CUDA_VISIBLE_DEVICES=4
NB=/home/mantovani/repo/generalize_knowledge/notebooks
cd "$NB" || { echo "ABORT: cannot cd to $NB"; exit 1; }

# sanity: refuse to run if the 128 edit isn't on disk
grep -q "max_new=128" ../src/sweep.py || { echo "ABORT: ../src/sweep.py still not max_new=128"; exit 1; }

# decl config — re-eval all 3 folds at 128 (overwrites the old 64-token scores)
python run.py --reeval --force \
  --exp experiments/exp01_decl.yaml \
  --facts data/facts_attr_v2.json \
  --folds 0 

echo "[gpu0 reeval] decl done. If 50/50 is also fully trained, uncomment its block below."

# --- 50/50 config: ONLY if its training is fully finished (n/102 in the notebook cell) ---
# python run.py --reeval --force \
#   --exp experiments/exp01_fic_decl_50.yaml \
#   --facts data/facts_attr_v2.json \
#   --folds 0 1 2