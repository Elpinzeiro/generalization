"""Single entrypoint for the layer sweep. ONE file, two modes:
  --mode interleaved : per config -> train + eval (train1,eval1, train2,eval2, ...)
  --mode batched     : ALL trains first, then ALL evals (uses saved adapters)
Both write the same result files, so plot_sweep works either way.
Hyperparameters come from the YAML (cfg['lora'] + cfg['train']).
Examples (inside screen):
  CUDA_VISIBLE_DEVICES=4 python run.py --facts data/facts_attr.json --fold 0 --seeds 0 1 2
  CUDA_VISIBLE_DEVICES=4 python run.py --facts data/facts_attr.json --fold 0 --mode batched
  CUDA_VISIBLE_DEVICES=4 python run.py --reeval --facts data/facts_attr.json --fold 0   # re-eval only
"""
import os, argparse, yaml, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
REPO = '/home/mantovani/repo/generalize_knowledge'
sys.path.insert(0, os.path.join(REPO, "src"))
import sweep

def R(p):
    """Resolve a path against REPO unless it's already absolute."""
    return p if os.path.isabs(p) else os.path.join(REPO, p)

def paths(cfg, facts, fold):
    tag = os.path.splitext(os.path.basename(facts))[0]
    ts = "-".join(cfg["split"]["train"]["styles"]); tl = "-".join(cfg["split"]["train"]["langs"])
    run = f"train_{ts}_{tl}"
    return (R(f"data/folds/{tag}/{run}"), R(f"data/results/{tag}/{run}/sweep_fold{fold}"))

def n_layers_of(model_path):
    from transformers import AutoConfig
    snap = os.path.join(model_path, "snapshots")
    mpath = os.path.join(snap, os.listdir(snap)[0]) if os.path.isdir(snap) else model_path
    return AutoConfig.from_pretrained(mpath).num_hidden_layers

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="experiments/exp01.yaml")
    ap.add_argument("--facts", default="data/facts_attr.json")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--mode", choices=["interleaved", "batched"], default="interleaved")
    ap.add_argument("--reeval", action="store_true", help="only re-eval saved adapters")
    ap.add_argument("--force", action="store_true", help="re-eval even if already evaluated")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(R(a.exp)))
    hp = sweep.hp_from_cfg(cfg)                       # <-- hyperparams FROM yaml
    model_path = cfg["models"]["target"]
    root, results = paths(cfg, a.facts, a.fold)
    nL = n_layers_of(model_path)
    print(f"[run] mode={a.mode} hp={hp}\n      {nL} layers -> configs all + {nL} singles + base")
    if a.reeval:
        sweep.eval_sweep(model_path, root, a.fold, results, force=a.force); return
    if a.mode == "interleaved":
        sweep.run_sweep(model_path, root, a.fold, a.seeds, nL, results, hp=hp, do_eval=True)
    else:  # batched
        sweep.run_sweep(model_path, root, a.fold, a.seeds, nL, results, hp=hp, do_eval=False)
        sweep.eval_sweep(model_path, root, a.fold, results)
    print(f"[done] results -> {results}")

if __name__ == "__main__":
    main()