"""OFAT layer/hp exploration driver (v2 mix folds). Companion to run.py.

ONE meta-yaml (experiments/sweep_ofat.yaml) defines a control hp point ("default")
plus one-factor-at-a-time variants, each swept over a chosen set of LAYER SCHEMES
(singles / windows / cumulative / all / base). Single fold, single seed => exactly
one run per config. Reuses sweep.train_adapter / save_adapter / eval_to_dir
unchanged; all scheme/naming/dir logic lives here so the existing single-layer
run_sweep path is left untouched.

Results are written PER EXPERIMENT so plot_sweep / inspect_sweep load one at a time:
  data/<tag>/<run>/fold<F>/sweep/<exp>/
      fold<F>_seed<S>_<cfg>.json            flat record  (plot_sweep + leaderboard read these)
      runs/<sig>/{adapter,raw_answer,evaluated,summary.json}
  data/<tag>/<run>/capacity_check/sweep/    <- capacity_check goes here (different train.json)

<sig> carries EVERY changed param, e.g.:
  W0-4_r16_a32_dp0.05_lr2e-4_ep4_bs4_tgtqkvo
so the training config is recoverable straight from the path.

The capacity_check experiment is the ONLY one that trains on a different file:
  data/<tag>/<run>/capacity_check/{train,eval_en,eval_it}.json
built by --build-capacity (both eval tiers folded into train). Every other
experiment is hard-branched onto fold<F>/train.json.

Usage (inside screen, GPU 4):
  CUDA_VISIBLE_DEVICES=4 python run_ofat.py --build-capacity                       # once
  CUDA_VISIBLE_DEVICES=4 python run_ofat.py --exp experiments/sweep_ofat.yaml
  CUDA_VISIBLE_DEVICES=4 python run_ofat.py --exp ... --only default lr_1e-5       # subset
  CUDA_VISIBLE_DEVICES=4 python run_ofat.py --exp ... --no-retention               # skip probe
"""
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")     # set BEFORE torch import; CVD=... overrides
import json, glob, gc, argparse, yaml

REPO = os.environ.get("GEN_REPO", "/home/mantovani/repo/generalize_knowledge")
sys.path.insert(0, os.path.join(REPO, "src"))
import sweep                                            # train_adapter / save_adapter / eval_to_dir
try:
    import evaluate                                     # collapse_probe (retention), needs torch
except Exception:
    evaluate = None


def R(p):
    return p if os.path.isabs(p) else os.path.join(REPO, p)

def run_root(cfg):
    tag = os.path.splitext(os.path.basename(cfg["base"]["facts"]))[0]
    return R(f"data/{tag}/{cfg['base']['run']}")


# ---------------- capacity_check data builder (run once via --build-capacity) ----------------
def build_capacity(cfg):
    """fold<F>/train.json + eval_en + eval_it queries -> capacity_check/train.json.
    eval rows are already style=qa_forward with question/expected, so they are valid
    training rows as-is. eval_en/eval_it copied verbatim (now SEEN during training)."""
    root, fold = run_root(cfg), cfg["fold"]
    fdir, cdir = f"{root}/fold{fold}", f"{root}/capacity_check"
    os.makedirs(cdir, exist_ok=True)
    train = json.load(open(f"{fdir}/train.json"))
    ev_en = json.load(open(f"{fdir}/eval_en.json"))
    ev_it = json.load(open(f"{fdir}/eval_it.json"))
    cap = train + ev_en + ev_it
    json.dump(cap,   open(f"{cdir}/train.json",   "w"), ensure_ascii=False, indent=2)
    json.dump(ev_en, open(f"{cdir}/eval_en.json", "w"), ensure_ascii=False, indent=2)
    json.dump(ev_it, open(f"{cdir}/eval_it.json", "w"), ensure_ascii=False, indent=2)
    print(f"[capacity] {len(train)} train + {len(ev_en)} en + {len(ev_it)} it "
          f"= {len(cap)} rows -> {cdir}", flush=True)
    return cdir


# ---------------- layer schemes ----------------
def _windows(n, size, stride):
    starts = list(range(0, n - size + 1, stride))      # full-size windows only
    out = [(s, s + size - 1) for s in starts]
    if not out or out[-1][1] != n - 1:                 # cover the top, keep window size
        out.append((n - size, n - 1))
    return out

def _cumulative(n, step):
    return [(0, t) for t in range(step, n, step)]      # 0-step .. 0-(<n); 'all' is the endpoint

def build_configs(scheme_names, n, sd):
    """-> list of (layers, label, layer_idx, scheme).
    layers: list | 'all' | 'base'  (passed straight to sweep.train_adapter)."""
    out = []
    if "singles" in scheme_names:
        out += [([i], f"L{i}", i, "singles") for i in range(n)]
    if "windows" in scheme_names:
        for lo, hi in _windows(n, sd["windows"]["size"], sd["windows"]["stride"]):
            out.append((list(range(lo, hi + 1)), f"W{lo}-{hi}", None, "windows"))
    if "cumulative" in scheme_names:
        for lo, hi in _cumulative(n, sd["cumulative"]["step"]):
            out.append((list(range(lo, hi + 1)), f"C{lo}-{hi}", None, "cumulative"))
    if "all" in scheme_names:
        out.append(("all", "all", None, "all"))
    if "base" in scheme_names:
        out.append(("base", "base", None, "base"))
    return out


# ---------------- hp + signature ----------------
_TGT = {"q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o",
        "gate_proj": "g", "up_proj": "u", "down_proj": "d"}
def _tgt_code(targets): return "".join(_TGT.get(t, "?") for t in targets)

def make_hp(defaults, override):
    d = dict(defaults); d.update(override or {})
    return dict(lr=float(d["lr"]), epochs=int(d["epochs"]), bs=int(d["batch_size"]),
                r=int(d["r"]), alpha=int(d["alpha"]), dropout=float(d["dropout"]),
                targets=list(d["targets"]))

def make_sig(label, hp):
    return (f"{label}_r{hp['r']}_a{hp['alpha']}_dp{hp['dropout']}"
            f"_lr{hp['lr']}_ep{hp['epochs']}_bs{hp['bs']}_tgt{_tgt_code(hp['targets'])}")


# ---------------- retention (optional) ----------------
def _leak_terms(eval_tiers, k=12):
    terms = set()
    for rows in eval_tiers.values():
        for e in rows:
            if e.get("check_kind") in ("place", "name") and isinstance(e.get("expected"), list):
                for al in e["expected"]:
                    terms.add(str(al))
    return list(terms)[:k]


def n_layers_of(model_path):
    from transformers import AutoConfig
    snap = os.path.join(model_path, "snapshots")
    mpath = os.path.join(snap, os.listdir(snap)[0]) if os.path.isdir(snap) else model_path
    return AutoConfig.from_pretrained(mpath).num_hidden_layers


# ---------------- one experiment ----------------
def run_experiment(cfg, exp, n, model_path, do_retention):
    name, root = exp["name"], run_root(cfg)
    fold, seed = cfg["fold"], cfg["seed"]
    is_cap = (name == "capacity_check")
    src_dir = f"{root}/capacity_check" if is_cap else f"{root}/fold{fold}"
    if not os.path.exists(f"{src_dir}/train.json"):
        hint = "  (run --build-capacity first)" if is_cap else ""
        print(f"[{name}] SKIP — no train.json at {src_dir}{hint}", flush=True); return

    results_dir = f"{root}/capacity_check/sweep" if is_cap else f"{root}/fold{fold}/sweep/{name}"
    runs_root = f"{results_dir}/runs"; os.makedirs(runs_root, exist_ok=True)
    train_rows = json.load(open(f"{src_dir}/train.json"))
    eval_tiers = {os.path.splitext(os.path.basename(p))[0]: json.load(open(p))
                  for p in sorted(glob.glob(f"{src_dir}/eval_*.json"))}
    hp = make_hp(cfg["defaults"], exp.get("override"))
    cfgs = build_configs(exp["schemes"], n, cfg["schemes_def"])
    if name != "default":
        # 'base' = untrained model, hp-invariant -> it runs ONCE in 'default' only.
        # Strip it here so it can never double-run even if a yaml re-adds it.
        cfgs = [c for c in cfgs if c[1] != "base"]
    leak = _leak_terms(eval_tiers) if (do_retention and evaluate) else []

    print(f"\n===== {name} =====  src={os.path.basename(src_dir)}  hp={hp}", flush=True)
    print(f"      {len(cfgs)} configs -> {results_dir}", flush=True)
    for layers, label, layer_idx, scheme in cfgs:
        sig = make_sig(label, hp)
        run_dir = f"{runs_root}/{sig}"; adapter_dir = f"{run_dir}/adapter"
        flat = f"{results_dir}/fold{fold}_seed{seed}_{label}.json"
        if os.path.isdir(f"{run_dir}/evaluated"):
            print(f"[skip] {sig} (evaluated)", flush=True); continue
        os.makedirs(run_dir, exist_ok=True)
        print(f"[run ] {sig} ...", flush=True)

        tok, model = sweep.train_adapter(model_path, train_rows, layers, seed, hp)
        if layers != "base":
            sweep.save_adapter(model, adapter_dir, layers)
        rec = {"fold": fold, "seed": seed, "exp": name, "scheme": scheme,
               "layers": label, "layer_idx": layer_idx,
               "n_active": (n if layers == "all" else 0 if layers == "base" else len(layers)),
               "hp": hp}
        rec.update(sweep.eval_to_dir(model, tok, eval_tiers, run_dir))     # {tier: acc}
        if leak:
            try:
                rec["retention"] = evaluate.collapse_probe(model, tok, leak)
            except Exception as e:
                print("  [retention] skipped:", e, flush=True)
        json.dump(rec, open(flat, "w"), indent=2)
        json.dump(rec, open(f"{run_dir}/summary.json", "w"), indent=2)
        del model; gc.collect()
        try:
            import torch; torch.cuda.empty_cache()
        except Exception:
            pass
        print(f"   done {sig}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="experiments/sweep_ofat.yaml")
    ap.add_argument("--build-capacity", action="store_true",
                    help="build capacity_check/{train,eval_*}.json then exit")
    ap.add_argument("--only", nargs="*", default=None, help="run only these experiment names")
    ap.add_argument("--no-retention", action="store_true", help="skip the collapse probe")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(R(a.exp)))
    if a.build_capacity:
        build_capacity(cfg); return

    model_path = cfg["base"]["model"]
    n = n_layers_of(model_path)
    exps = cfg["experiments"]
    if a.only:
        exps = [e for e in exps if e["name"] in a.only]
    print(f"[ofat] run={cfg['base']['run']} fold{cfg['fold']} seed{cfg['seed']} "
          f"{n} layers  experiments={[e['name'] for e in exps]}", flush=True)
    for exp in exps:
        run_experiment(cfg, exp, n, model_path, do_retention=not a.no_retention)
    print("\n[all done]", flush=True)


if __name__ == "__main__":
    main()