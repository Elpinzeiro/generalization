"""Plot held-out accuracy vs single LoRA layer (the per-layer experiment),
with all-layers and base as reference lines. Reads summary.json files from a sweep dir."""
import json, glob
import matplotlib.pyplot as plt


def layer_sweep(sweep_dir, tier="eval_easy", out="plots/layer_sweep.png"):
    rows = [json.load(open(p)) for p in glob.glob(f"{sweep_dir}/*/summary.json")]
    by = {r["name"]: r for r in rows}
    single = sorted([r for r in rows if isinstance(r.get("layers"), list) and len(r["layers"]) == 1],
                    key=lambda r: r["layers"][0])
    xs = [r["layers"][0] for r in single]

    plt.figure(figsize=(10, 4))
    plt.plot(xs, [r[tier] for r in single], "o-", label=f"single layer ({tier})")
    for nm, c, ls in [("all_layers", "green", "--"), ("base", "red", ":")]:
        if nm in by and by[nm].get(tier) is not None:
            plt.axhline(by[nm][tier], c=c, ls=ls, label=nm)
    plt.xlabel("LoRA layer"); plt.ylabel(f"{tier} accuracy"); plt.ylim(-.02, 1.02)
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(out, dpi=120)
    print(f"[plot] -> {out}")
