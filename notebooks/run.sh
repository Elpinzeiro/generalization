#!/bin/bash
# Run inside screen:  bash run.sh
set -e
export CUDA_VISIBLE_DEVICES=1
ARGS="--facts data/facts_attr.json --fold 0 --seeds 0 1 2"

# pick ONE:
python run.py $ARGS --mode "interleaved"        # all trains, then all evals
# python run.py $ARGS --mode interleaved  # train1,eval1, train2,eval2, ...

du -sh data/results/*/*/sweep_fold0/runs/*/adapter 2>/dev/null | tail -5