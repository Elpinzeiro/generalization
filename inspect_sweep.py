"""Quick inspection for OFAT sweeps — no plotting required to answer
"did any reduced layer-set beat all on eval_it?".

'base' (the untrained model) is trained ONCE, in the 'default' experiment. It is
hp-invariant, so every other folder borrows that single record:
  - leaderboard() auto-loads it from default when the folder has no local base.
  - link_base(RUN, fold) copies it into the other sweep folders so the plain
    sweep.plot_sweep(<folder>) shows the base floor line everywhere.

In a thin notebook:
    import importlib, inspect_sweep, sweep
    importlib.reload(inspect_sweep)
    RUN = "/home/.../data/facts_attr_v2/fic_qa-decl_en__anc_qa50-decl50_en"

    inspect_sweep.headline(RUN, fold=0)                       # best reduced vs all, per experiment
    inspect_sweep.link_base(RUN, fold=0)                      # propagate the single base record
    inspect_sweep.leaderboard(RUN + "/fold0/sweep/lr_1e-5")   # full table incl base floor
    sweep.plot_sweep(RUN + "/fold0/sweep/default")            # per-layer curve (the singles map)
    inspect_sweep.plot(RUN + "/fold0/sweep/bs16")             # singles + windows + all + base floor

plot() overlays windows (at their pair-centre) on the per-layer singles curve, one
colour per eval tier (eval_en / eval_it stay consistent), singles vs windows by style.
"""
import os, re, glob, json


# ---------------- loading ----------------
def _load(results_dir):
    recs = []
    for p in glob.glob(f"{results_dir}/fold*_seed*.json"):
        try:
            r = json.load(open(p))
        except (json.JSONDecodeError, OSError):
            continue
        if "layers" in r:
            recs.append(r)
    return recs


def _locate(results_dir):
    """Infer (run_root, fold) from a sweep results dir, so base can be found in default."""
    rd = os.path.normpath(results_dir)
    m = re.search(r"(.*)/fold(\d+)/sweep(?:/[^/]+)?$", rd)
    if m:
        return m.group(1), int(m.group(2))
    m = re.search(r"(.*)/capacity_check/sweep$", rd)   # single-fold design -> fold 0
    if m:
        return m.group(1), 0
    return None, None


def _base_row(run_root, fold=0, seed=0):
    p = f"{run_root}/fold{fold}/sweep/default/fold{fold}_seed{seed}_base.json"
    try:
        return json.load(open(p))
    except (OSError, json.JSONDecodeError):
        return None


def link_base(run_root, fold=0, seed=0):
    """Copy the single base record (trained in 'default') into every OTHER sweep folder,
    so sweep.plot_sweep(<any folder>) shows the base floor. Idempotent."""
    br = _base_row(run_root, fold, seed)
    if br is None:
        print("no base record in default — run the 'default' experiment first"); return []
    targets = glob.glob(f"{run_root}/fold{fold}/sweep/*")
    cap = f"{run_root}/capacity_check/sweep"
    if os.path.isdir(cap):
        targets.append(cap)
    touched = []
    for d in targets:
        if os.path.basename(d) == "default" or not os.path.isdir(d):
            continue
        dst = f"{d}/fold{fold}_seed{seed}_base.json"
        if not os.path.exists(dst):
            json.dump(br, open(dst, "w"), indent=2)
            touched.append(os.path.basename(d))
    print(f"linked base into {len(touched)} folder(s): {touched}")
    return touched


# ---------------- tables ----------------
def _size(r):
    """Relative trainable-param proxy = n_active_layers * n_targets * rank."""
    hp = r.get("hp") or {}
    na = r.get("n_active")
    if na is None or not hp:
        return None
    return na * len(hp.get("targets", [])) * int(hp.get("r", 0))


def leaderboard(results_dir, tier="eval_it", also=("eval_en",), show=True):
    """One row per config in ONE experiment folder, sorted by `tier` desc.
    base floor is included even if not local — loaded from default.
    d_vs_all = this run's tier minus this experiment's own 'all' run (>0 => reduced won)."""
    recs = _load(results_dir)
    if not recs:
        print("no results in", results_dir); return []
    # ensure a base row is present (borrow the single one from default if needed)
    if not any(r["layers"] == "base" for r in recs):
        rr, fold = _locate(results_dir)
        br = _base_row(rr, fold) if rr else None
        if br is not None:
            recs.append(br)

    allrow = next((r for r in recs if r["layers"] == "all"), None)
    a_tier = allrow.get(tier) if allrow else None

    rows = []
    for r in recs:
        d = {"cfg": r["layers"], "scheme": r.get("scheme"), tier: r.get(tier)}
        for t in also:
            d[t] = r.get(t)
        d["d_vs_all"] = (None if r.get(tier) is None or a_tier is None
                         else round(r[tier] - a_tier, 4))
        d["retention"] = r.get("retention")
        d["size"] = _size(r)
        rows.append(d)
    rows.sort(key=lambda x: (x[tier] is not None, x.get(tier) or -1), reverse=True)

    if show:
        cols = ["cfg", "scheme", tier, *also, "d_vs_all", "retention", "size"]
        w = {c: max(len(c), *(len(_fmt(r.get(c))) for r in rows)) for c in cols}
        print("  ".join(c.ljust(w[c]) for c in cols))
        print("  ".join("-" * w[c] for c in cols))
        for r in rows:
            print("  ".join(_fmt(r.get(c)).ljust(w[c]) for c in cols))
        if a_tier is not None:
            print(f"\n(all {tier} = {a_tier};  d_vs_all > 0 means the reduced config beat all)")
    return rows


def headline(run_root, fold=0, tier="eval_it"):
    """Across every sweep/<exp> folder (+ capacity_check/sweep): best REDUCED config vs all.
    The one-glance answer to 'which experiments produced a reduced winner'."""
    dirs = sorted(glob.glob(f"{run_root}/fold{fold}/sweep/*"))
    cap = f"{run_root}/capacity_check/sweep"
    if os.path.isdir(cap):
        dirs.append(cap)
    dirs = [d for d in dirs if os.path.isdir(d)]
    if not dirs:
        print("no sweep folders under", run_root); return
    name_w = max(len(_expname(d)) for d in dirs)
    for d in dirs:
        recs = _load(d)
        if not recs:
            continue
        allr = next((r for r in recs if r["layers"] == "all"), None)
        a = allr.get(tier) if allr else None
        red = [r for r in recs if r["layers"] not in ("all", "base") and r.get(tier) is not None]
        if not red:
            print(f"{_expname(d).ljust(name_w)}  all={_fmt(a)}   (no reduced configs)")
            continue
        best = max(red, key=lambda r: r[tier])
        dv = None if a is None else round(best[tier] - a, 3)
        tag = "WIN " if (a is not None and best[tier] > a) else "    "
        print(f"{_expname(d).ljust(name_w)}  all={_fmt(a)}   "
              f"best_reduced={best['layers']}={_fmt(best[tier])}  d={_fmt(dv)}  {tag}")


def _ensure_base(results_dir):
    """Make sure a base record sits in this folder, borrowing the single one from default."""
    if glob.glob(f"{results_dir}/fold*_seed*_base.json"):
        return
    rr, fold = _locate(results_dir)
    br = _base_row(rr, fold) if rr else None
    if br is not None:
        seed = br.get("seed", 0)
        json.dump(br, open(f"{results_dir}/fold{fold}_seed{seed}_base.json", "w"), indent=2)


def plot(results_dir, tiers=None, title=None, show_windows=True):
    """Per-layer view of ONE experiment: singles as points, windows overlaid at their
    pair-centre, all/base as reference lines. Each eval tier keeps ONE colour (so eval_en
    and eval_it stay consistent with sweep.plot_sweep); singles vs windows differ by style.
    base floor is borrowed from default if this folder has none."""
    import numpy as np, matplotlib.pyplot as plt
    _ensure_base(results_dir)
    recs = _load(results_dir)
    if not recs:
        print("no results in", results_dir); return

    if tiers is None:
        tiers = sorted({k for r in recs for k in r if k.startswith("eval_")})
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    tcol = {t: cycle[i % len(cycle)] for i, t in enumerate(tiers)}   # same mapping as sweep.plot_sweep

    plt.figure(figsize=(11, 4))
    for t in tiers:
        c = tcol[t]
        # singles  ->  x = layer index
        sing = sorted((int(mm.group(1)), r[t]) for r in recs
                      for mm in [re.match(r"L(\d+)$", str(r.get("layers")))]
                      if mm and r.get(t) is not None)
        if sing:
            xs, ys = zip(*sing)
            plt.plot(xs, ys, marker="o", color=c, label=f"{t} single")
        # windows  ->  x = centre of [lo, hi]
        if show_windows:
            win = []
            for r in recs:
                mm = re.match(r"W(\d+)-(\d+)$", str(r.get("layers")))
                if mm and r.get(t) is not None:
                    lo, hi = int(mm.group(1)), int(mm.group(2))
                    win.append(((lo + hi) / 2, r[t]))
            win.sort()
            if win:
                xs, ys = zip(*win)
                plt.plot(xs, ys, marker="s", ms=4, ls="-.", color=c, alpha=.85,
                         label=f"{t} window")
        # reference lines
        for name, ls in [("all", "--"), ("base", ":")]:
            v = [r[t] for r in recs if r.get("layers") == name and r.get(t) is not None]
            if v:
                plt.axhline(np.mean(v), color=c, ls=ls, alpha=.6, label=f"{name} {t}")

    plt.xlabel("layer  (single = point, window = centre of pair)")
    plt.ylabel("accuracy"); plt.ylim(-.02, 1.02)
    plt.legend(fontsize=8, ncol=2); plt.grid(alpha=.3)
    plt.title(title or f"per-layer — {_expname(results_dir)}")
    plt.tight_layout(); plt.show()
    print(f"{len(recs)} records")


# ---------------- helpers ----------------
def _expname(d):
    return "capacity_check" if d.rstrip("/").endswith("capacity_check/sweep") else os.path.basename(d)


def _fmt(v):
    if v is None: return "-"
    if isinstance(v, float): return f"{v:.3f}"
    return str(v)