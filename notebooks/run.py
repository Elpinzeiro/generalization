"""Single entrypoint for the layer sweep (v2 mix-based folds).

Reads folds written by build.build_folds at:   data/<tag>/<run_name>/fold<K>/
Writes results INSIDE each fold:               data/<tag>/<run_name>/fold<K>/sweep/
  sweep/fold<K>_seed<S>_<layer>.json           flat records (plot_sweep reads these)
  sweep/runs/<sig>/adapter/                     LoRA adapter
  sweep/runs/<sig>/raw_answer/<tier>.json       what the model generated (debug)
  sweep/runs/<sig>/evaluated/<tier>.json        per-question correctness (debug)
  sweep/runs/<sig>/summary.json                 accuracies + hp

run_name comes from cfg['mix'] via build._run_name — the SAME string build_folds used,
so folds and results can never disagree. Eval tiers are whatever eval_*.json files the
fold contains (e.g. eval_en.json + eval_it.json) — picked up automatically by sweep.

Examples (inside screen), GPU 4:
  CUDA_VISIBLE_DEVICES=4 python run.py --exp experiments/exp01_decl.yaml        --facts data/facts_attr_v2.json --folds 0 1 2 --seeds 0 1 2
  CUDA_VISIBLE_DEVICES=4 python run.py --exp experiments/exp01_fic_decl_50.yaml --facts data/facts_attr_v2.json --folds 0 1 2 --seeds 0 1 2
  CUDA_VISIBLE_DEVICES=4 python run.py --reeval --exp experiments/exp01_decl.yaml --facts data/facts_attr_v2.json --folds 0 1 2   # re-eval only
"""
import os, argparse, yaml, sys
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '4')      # GPU 4 by default; CUDA_VISIBLE_DEVICES=... overrides
REPO = '/home/mantovani/repo/generalize_knowledge'
sys.path.insert(0, os.path.join(REPO, "src"))
import sweep, build                                     # build for _run_name (single source of truth)


def R(p):
    """Resolve a path against REPO unless it's already absolute."""
    return p if os.path.isabs(p) else os.path.join(REPO, p)


def paths(cfg, facts, fold):
    """root  = where the folds live:           data/<tag>/<run_name>
       results = results INSIDE the fold:       data/<tag>/<run_name>/fold<K>/sweep
    run_name is derived from cfg['mix'] EXACTLY as build.build_folds derived it."""
    tag = os.path.splitext(os.path.basename(facts))[0]
    run = build._run_name(cfg["mix"])
    root    = R(f"data/{tag}/{run}")
    results = R(f"data/{tag}/{run}/fold{fold}/sweep")
    return root, results


def n_layers_of(model_path):
    from transformers import AutoConfig
    snap = os.path.join(model_path, "snapshots")
    mpath = os.path.join(snap, os.listdir(snap)[0]) if os.path.isdir(snap) else model_path
    return AutoConfig.from_pretrained(mpath).num_hidden_layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="experiments/exp01_decl.yaml")
    ap.add_argument("--facts", default="data/facts_attr_v2.json")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--mode", choices=["interleaved", "batched"], default="interleaved")
    ap.add_argument("--reeval", action="store_true", help="only re-eval saved adapters")
    ap.add_argument("--force", action="store_true", help="re-eval even if already evaluated")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(R(a.exp)))
    hp = sweep.hp_from_cfg(cfg)
    model_path = cfg["models"]["target"]
    nL = n_layers_of(model_path)
    run = build._run_name(cfg["mix"])
    print(f"[run] exp={os.path.basename(a.exp)}  run={run}")
    print(f"      mode={a.mode}  folds={a.folds}  seeds={a.seeds}  hp={hp}")
    print(f"      {nL} layers -> configs all + {nL} singles + base")

    for fold in a.folds:
        root, results = paths(cfg, a.facts, fold)
        fold_dir = R(f"{os.path.dirname(os.path.dirname(results))}/fold{fold}")  # data/<tag>/<run>/fold<K>
        if not os.path.exists(f"{root}/fold{fold}/train.json"):
            print(f"[fold{fold}] SKIP — no train.json at {root}/fold{fold} "
                  f"(did you build this config's folds?)", flush=True)
            continue
        print(f"\n===== FOLD {fold} =====  results -> {results}", flush=True)
        if a.reeval:
            sweep.eval_sweep(model_path, root, fold, results, force=a.force)
        elif a.mode == "interleaved":
            sweep.run_sweep(model_path, root, fold, a.seeds, nL, results, hp=hp, do_eval=True)
        else:  # batched
            sweep.run_sweep(model_path, root, fold, a.seeds, nL, results, hp=hp, do_eval=False)
            sweep.eval_sweep(model_path, root, fold, results)
        print(f"[done fold{fold}] results -> {results}", flush=True)

    print("\n[all done]")


if __name__ == "__main__":
    main()