#!/bin/bash
# Run inside screen:  bash run.sh
# Launches the TWO configs you want this round (decl + 50/50), 3 folds, 3 seeds, on GPU 4.
# exp01_qa.yaml and the *_old backups are NOT run here.
set -e
export CUDA_VISIBLE_DEVICES=4

FACTS="data/facts_attr_v2.json"
ARGS="--facts $FACTS --folds 0 1 2 --seeds 0 --mode interleaved"

# --- config 1: fiction 100% declarative ---
python run.py --exp experiments/exp01_decl.yaml        $ARGS

# --- config 2: fiction 50/50 qa+declarative ---
python run.py --exp experiments/exp01_qa.yaml $ARGS

# --- config 3: fiction 50/50 qa+declarative ---
python run.py --exp experiments/exp01_fic_decl_50.yaml $ARGS

# quick size peek at saved adapters across both configs
du -sh data/facts_attr_v2/*/fold*/sweep/runs/*/adapter 2>/dev/null | tail -8