"""LoRA layer sweep: configs = ['all'] + each single layer + ['base'] (lower bound),
x seeds. Trains EXACTLY like the original (qa=chat-template masked SFT, declarative=NTP
full-loss; any other style -> error). Writes one result file per run -> re-runnable,
resumable, and plottable while still in progress."""
import os, json, gc, glob, random, re
import verify

HP = dict(lr=2e-4, epochs=4, bs=4, r=32, alpha=64, dropout=0.05,
          targets=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])


def hp_from_cfg(cfg):
    """Build the hyperparameter dict FROM the YAML (single source of truth).
    Override in a notebook by mutating the returned dict before passing it in."""
    L, T = cfg["lora"], cfg["train"]
    return dict(lr=float(T["lr"]), epochs=int(T["epochs"]), bs=int(T["batch_size"]),
                r=int(L["r"]), alpha=int(L["alpha"]), dropout=float(L["dropout"]),
                targets=list(L["targets"]))


# ---------- example building: branch on STYLE (2 losses; error otherwise) ----------
def _qa_answer(r):
    if r.get("answer"): return r["answer"]                 # anchor q/a: literal
    exp = r.get("expected")                                # fiction q/a: gold place
    return exp[0] if isinstance(exp, list) and exp else str(exp)

def build_examples(tok, rows):
    eos = tok.eos_token_id
    ex = []
    for r in rows:
        st = r.get("style")
        if st == "qa_forward":                              # chat template + loss on answer only
            pids = tok.apply_chat_template([{"role":"user","content":r["question"]}],
                                           add_generation_prompt=True)
            aids = tok(_qa_answer(r), add_special_tokens=False)["input_ids"] + [eos]
            ex.append((pids + aids, [-100]*len(pids) + aids))
        elif st == "declarative":                           # plain NTP, loss on everything (CPT)
            ids = tok(r["text"], add_special_tokens=True)["input_ids"] + [eos]
            ex.append((ids, list(ids)))
        else:
            raise NotImplementedError(
                f"style '{st}' (e.g. cloze / multichoice) loss not implemented!")
    return ex


# ---------- one train+eval run ----------
def _load_base(model_path):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    snap = os.path.join(model_path, "snapshots")
    mpath = os.path.join(snap, os.listdir(snap)[0]) if os.path.isdir(snap) else model_path
    tok = AutoTokenizer.from_pretrained(mpath)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(mpath, device_map="auto", torch_dtype=torch.bfloat16)
    return tok, model

def eval_log(model, tok, rows, out_path=None, max_new=128):
    """Per-question eval detail. Returns (acc, log); optionally saves log to JSON.
    Each entry: id, question, raw answer, kind, expected, correct(bool)."""
    import torch
    q = [e for e in rows if e.get("check_kind")]
    log = []
    for e in q:
        ids = tok.apply_chat_template([{"role":"user","content":e["question"]}],
                                      add_generation_prompt=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new,
                                  do_sample=False, pad_token_id=tok.eos_token_id)
        ans = tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True).strip()
        ok = bool(verify.check(ans, e["expected"], e["check_kind"]))
        log.append({"id": f"f{e.get('fact_id')}_p{e.get('pid')}_{e.get('style')}_{e.get('lang')}",
                    "fact_id": e.get("fact_id"), "pid": e.get("pid"),
                    "style": e.get("style"), "lang": e.get("lang"),
                    "question": e["question"], "answer": ans,
                    "kind": e["check_kind"], "expected": e["expected"], "correct": ok})
    acc = sum(r["correct"] for r in log)/len(log) if log else None
    if out_path:
        json.dump(log, open(out_path, "w"), ensure_ascii=False, indent=2)
        n_ok = sum(r["correct"] for r in log)
        print(f"[eval_log] {n_ok}/{len(log)} correct -> {out_path}", flush=True)
    return acc, log

def _eval(model, tok, rows, max_new=128, out_path=None):
    acc, _ = eval_log(model, tok, rows, out_path=out_path, max_new=max_new)
    return acc

def train_adapter(model_path, train_rows, layers, seed, hp=HP):
    """Train and RETURN (tok, model) in memory (not freed). layers='base' -> base, no training.
    Use save_adapter(model, dir, layers) to persist, or make_chat(tok, model) to probe."""
    import torch
    from transformers import get_linear_schedule_with_warmup
    from peft import LoraConfig, TaskType, get_peft_model
    random.seed(seed); torch.manual_seed(seed)
    tok, model = _load_base(model_path)
    if layers == "base":
        model.eval(); return tok, model

    lc = LoraConfig(r=hp["r"], lora_alpha=hp["alpha"], lora_dropout=hp["dropout"],
                    target_modules=hp["targets"], bias="none", task_type=TaskType.CAUSAL_LM,
                    layers_to_transform=(None if layers == "all" else list(layers)))
    model = get_peft_model(model, lc)
    active = sorted({int(g.group(1)) for n, _ in model.named_modules()
                     if "lora_A" in n and (g := re.search(r"\.layers\.(\d+)\.", n))})
    if layers != "all":
        assert active == sorted(layers), f"LAYER MISMATCH {active} != {layers}"   # all/single ONLY
    ex = build_examples(tok, train_rows)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=hp["lr"])
    steps = ((len(ex)+hp["bs"]-1)//hp["bs"])*hp["epochs"]
    sched = get_linear_schedule_with_warmup(opt, int(0.03*steps), steps)
    for ep in range(hp["epochs"]):
        model.train(); random.shuffle(ex); tot=nb=0
        for i in range(0, len(ex), hp["bs"]):
            b = ex[i:i+hp["bs"]]; mx = max(len(x[0]) for x in b)
            ids = torch.tensor([x[0]+[tok.pad_token_id]*(mx-len(x[0])) for x in b])
            lbl = torch.tensor([x[1]+[-100]*(mx-len(x[1])) for x in b])
            att = torch.tensor([[1]*len(x[0])+[0]*(mx-len(x[0])) for x in b])
            out = model(input_ids=ids.cuda(), attention_mask=att.cuda(), labels=lbl.cuda())
            out.loss.backward(); opt.step(); sched.step(); opt.zero_grad()
            tot += out.loss.item(); nb += 1
        print(f"      epoch {ep+1}/{hp['epochs']} loss={tot/nb:.4f}", flush=True)
    model.eval()
    return tok, model


def _dirsize_mb(d):
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(d) for f in fs) / 1e6

def save_adapter(model, adapter_dir, layers):
    if layers != "base":
        os.makedirs(adapter_dir, exist_ok=True)
        model.save_pretrained(adapter_dir)
        print(f"   adapter saved -> {adapter_dir}  ({_dirsize_mb(adapter_dir):.1f} MB)", flush=True)


def load_for_eval(model_path, adapter_dir=None):
    """Load base (+ adapter if given) for a SEPARATE eval pass. adapter_dir=None -> base."""
    tok, model = _load_base(model_path)
    if adapter_dir and os.path.isdir(adapter_dir):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return tok, model


def eval_to_dir(model, tok, eval_tiers, run_dir):
    """Run eval, write run_dir/raw_answer/<tier>.json + run_dir/evaluated/<tier>.json. Returns {tier:acc}."""
    res = {}
    for tier, rows in eval_tiers.items():
        acc, log = eval_log(model, tok, rows)
        res[tier] = acc
        if log:
            os.makedirs(f"{run_dir}/raw_answer", exist_ok=True)
            os.makedirs(f"{run_dir}/evaluated", exist_ok=True)
            json.dump([{"id": r["id"], "question": r["question"], "raw_answer": r["answer"]} for r in log],
                      open(f"{run_dir}/raw_answer/{tier}.json", "w"), ensure_ascii=False, indent=2)
            json.dump(log, open(f"{run_dir}/evaluated/{tier}.json", "w"), ensure_ascii=False, indent=2)
            print(f"   [{tier}] {sum(r['correct'] for r in log)}/{len(log)} correct -> {run_dir}", flush=True)
    return res


def make_chat(tok, model, max_new=100):
    """Returns a chat(msg) fn to probe the in-memory model (collapse check)."""
    import torch
    @torch.no_grad()
    def chat(msg, max_new_tokens=max_new, show=True):
        ids = tok.apply_chat_template([{"role":"user","content":msg}], add_generation_prompt=True,
                                      return_tensors="pt").to(model.device)
        out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tok.eos_token_id)
        ans = tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True).strip()
        if show: print(ans)
        return ans
    return chat


def run_one(model_path, train_rows, eval_tiers, layers, seed, hp=HP, run_dir=None, adapter_dir=None):
    """Convenience: train (in memory) -> save adapter -> eval -> free. Returns {tier:acc}."""
    import torch, gc
    tok, model = train_adapter(model_path, train_rows, layers, seed, hp)
    if adapter_dir: save_adapter(model, adapter_dir, layers)
    res = eval_to_dir(model, tok, eval_tiers, run_dir) if run_dir else {}
    del model; gc.collect(); torch.cuda.empty_cache()
    return res


# ---------- the sweep: all + each single layer + base, x seeds ----------
def configs_for(n_layers):
    """The ONLY configs: 'all', every single layer, 'base'. No windows."""
    return ["all"] + [[L] for L in range(n_layers)] + ["base"]

def _name(layers):
    return "all" if layers == "all" else "base" if layers == "base" else f"L{layers[0]}"

def _sig(hp, layers, seed, fold):
    """Encodes ALL train params so different runs never share a folder."""
    return (f"{_name(layers)}_seed{seed}_fold{fold}"
            f"_r{hp['r']}_a{hp['alpha']}_dp{hp['dropout']}"
            f"_lr{hp['lr']}_ep{hp['epochs']}_bs{hp['bs']}")

def run_sweep(model_path, root, fold, seeds, n_layers, results_dir, hp=HP, do_eval=True):
    """do_eval=True : per config train+save+eval (interleaved).
       do_eval=False: TRAIN ONLY (train+save adapter), no eval -> run eval_sweep later.
    Lets you either interleave, or batch all trains then all evals."""
    os.makedirs(results_dir, exist_ok=True)
    runs_root = os.path.join(results_dir, "runs")
    train_rows = json.load(open(f"{root}/fold{fold}/train.json"))
    eval_tiers = {os.path.splitext(os.path.basename(p))[0]: json.load(open(p))
                  for p in glob.glob(f"{root}/fold{fold}/eval_*.json")}
    configs = configs_for(n_layers)
    mode = "train+eval" if do_eval else "TRAIN ONLY"
    print(f"[sweep:{mode}] fold{fold} seeds={seeds} configs={[_name(c) for c in configs]}", flush=True)
    import gc
    for seed in seeds:
        for layers in configs:
            sig = _sig(hp, layers, seed, fold)
            run_dir = os.path.join(runs_root, sig); adapter_dir = os.path.join(run_dir, "adapter")
            flat = f"{results_dir}/fold{fold}_seed{seed}_{_name(layers)}.json"
            is_base = layers == "base"
            adapter_done = (os.path.exists(f"{run_dir}/summary.json") if is_base
                            else os.path.isdir(adapter_dir))
            eval_done = os.path.isdir(f"{run_dir}/evaluated")
            if do_eval and eval_done:        print(f"[skip] {sig} (evaluated)", flush=True); continue
            if (not do_eval) and adapter_done: print(f"[skip] {sig} (trained)", flush=True); continue
            print(f"[run ] {sig} ...", flush=True)
            os.makedirs(run_dir, exist_ok=True)
            tok, model = train_adapter(model_path, train_rows, layers, seed, hp)
            if not is_base: save_adapter(model, adapter_dir, layers)
            rec = {"fold":fold,"seed":seed,"layers":_name(layers),
                   "layer_idx":(None if is_base or layers=="all" else layers[0]),"hp":hp}
            if do_eval:
                rec.update(eval_to_dir(model, tok, eval_tiers, run_dir))
                json.dump(rec, open(flat, "w"), indent=2)
            json.dump(rec, open(f"{run_dir}/summary.json", "w"), indent=2)
            del model; gc.collect()
            try:
                import torch; torch.cuda.empty_cache()
            except Exception: pass
            print(f"   done {sig}", flush=True)
    return results_dir


def eval_sweep(model_path, root, fold, results_dir, force=False):
    """SEPARATE eval task: re-evaluate every saved adapter under results_dir/runs/ against
    the fold's eval tiers, WITHOUT retraining. Use after changing verify.py or eval sets.
    Re-writes raw_answer/, evaluated/, summary.json. Skips runs already evaluated unless force."""
    import gc
    runs_root = os.path.join(results_dir, "runs")
    eval_tiers = {os.path.splitext(os.path.basename(p))[0]: json.load(open(p))
                  for p in glob.glob(f"{root}/fold{fold}/eval_*.json")}
    run_dirs = sorted(glob.glob(f"{runs_root}/*"))
    print(f"[eval_sweep] {len(run_dirs)} runs", flush=True)
    for run_dir in run_dirs:
        if (not force) and os.path.exists(f"{run_dir}/evaluated/{next(iter(eval_tiers))}.json"):
            print(f"[skip] {os.path.basename(run_dir)} (already evaluated)", flush=True); continue
        adapter = os.path.join(run_dir, "adapter")
        is_base = "/base_" in run_dir or os.path.basename(run_dir).startswith("base_")
        print(f"[eval] {os.path.basename(run_dir)} ...", flush=True)
        tok, model = load_for_eval(model_path, None if is_base else adapter)
        res = eval_to_dir(model, tok, eval_tiers, run_dir)
        # refresh summary accuracies, keep other fields
        summ = json.load(open(f"{run_dir}/summary.json")) if os.path.exists(f"{run_dir}/summary.json") else {}
        summ.update(res); json.dump(summ, open(f"{run_dir}/summary.json","w"), indent=2)
        flat = f"{results_dir}/fold{summ.get('fold',fold)}_seed{summ.get('seed','?')}_{summ.get('layers','?')}.json"
        json.dump(summ, open(flat,"w"), indent=2)
        del model; gc.collect()
        try:
            import torch; torch.cuda.empty_cache()
        except Exception: pass
        print(f"   -> {res}", flush=True)
    return results_dir


# # ---------- plot (re-run anytime to see progress) ----------
# def plot_sweep(results_dir, tiers=None, title="Per-layer generalization (mean±std over seeds)"):
#     import numpy as np, matplotlib.pyplot as plt, collections
#     recs = [json.load(open(p)) for p in glob.glob(f"{results_dir}/*.json")]
#     if not recs:
#         print("no results yet"); return
#     if tiers is None:
#         tiers = sorted({k for r in recs for k in r if k.startswith("eval_")})
#     plt.figure(figsize=(11, 4))
#     for tier in tiers:
#         by = collections.defaultdict(list)
#         for r in recs:
#             if r["layer_idx"] is not None and r.get(tier) is not None:
#                 by[r["layer_idx"]].append(r[tier])
#         xs = sorted(by)
#         if xs:
#             plt.errorbar(xs, [np.mean(by[x]) for x in xs], yerr=[np.std(by[x]) for x in xs],
#                          marker="o", capsize=3, label=f"{tier} (single layer)")
#     for name, ls in [("all", "--"), ("base", ":")]:
#         for tier in tiers:
#             v = [r[tier] for r in recs if r["layers"] == name and r.get(tier) is not None]
#             if v: plt.axhline(np.mean(v), ls=ls, alpha=.6, label=f"{name} {tier}")
#     plt.xlabel("single LoRA layer"); plt.ylabel("accuracy"); plt.ylim(-.02, 1.02)
#     plt.legend(fontsize=8); plt.grid(alpha=.3); plt.title(title); plt.tight_layout(); plt.show()
#     done = collections.Counter((r["seed"], r["layers"]) for r in recs)
#     print(f"{len(recs)} runs on disk  (seeds done: {sorted({r['seed'] for r in recs})})")
# ---------- plot (re-run anytime to see progress) ----------
# def plot_sweep(results_dir, tiers=None, title="Per-layer generalization (mean±std over seeds)"):
#     import numpy as np, matplotlib.pyplot as plt, collections
#     recs = [json.load(open(p)) for p in glob.glob(f"{results_dir}/*.json")]
#     if not recs:
#         print("no results yet"); return
#     if tiers is None:
#         tiers = sorted({k for r in recs for k in r if k.startswith("eval_")})

#     # one fixed color per tier -> single line AND its all/base references share it
#     cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
#     tier_color = {t: cycle[i % len(cycle)] for i, t in enumerate(tiers)}

#     plt.figure(figsize=(11, 4))
#     for tier in tiers:
#         c = tier_color[tier]
#         # single-layer points (solid, with markers) in the tier color
#         by = collections.defaultdict(list)
#         for r in recs:
#             if r.get("layer_idx") is not None and r.get(tier) is not None: 
#                 by[r["layer_idx"]].append(r[tier])
#         xs = sorted(by)
#         if xs:
#             plt.errorbar(xs, [np.mean(by[x]) for x in xs], yerr=[np.std(by[x]) for x in xs],
#                          marker="o", capsize=3, color=c, label=f"{tier} (single layer)")
#         # reference lines in the SAME tier color: all = dashed, base = dotted
#         for name, ls in [("all", "--"), ("base", ":")]:
#             v = [r[tier] for r in recs if r["layers"] == name and r.get(tier) is not None]
#             if v:
#                 plt.axhline(np.mean(v), color=c, ls=ls, alpha=.8, label=f"{name} {tier}")

#     plt.xlabel("single LoRA layer"); plt.ylabel("accuracy"); plt.ylim(-.02, 1.02)
#     plt.legend(fontsize=8, ncol=2); plt.grid(alpha=.3); plt.title(title)
#     plt.tight_layout(); plt.show()
#     print(f"{len(recs)} runs on disk  (seeds done: {sorted({r['seed'] for r in recs})})")

def plot_sweep(results_dir, tiers=None, title="Per-layer generalization (mean±std over seeds)"):
    import numpy as np, matplotlib.pyplot as plt, collections, json, glob

    # only the flat result records, never train.json/manifest/summary/half-written files
    recs = []
    for p in glob.glob(f"{results_dir}/fold*_seed*.json"):
        try:
            r = json.load(open(p))
        except (json.JSONDecodeError, OSError):
            continue                       # skip a file mid-write (safe to re-run later)
        if "layers" in r:                  # looks like a real result record
            recs.append(r)
    if not recs:
        print("no results yet"); return

    if tiers is None:
        tiers = sorted({k for r in recs for k in r if k.startswith("eval_")})
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    tier_color = {t: cycle[i % len(cycle)] for i, t in enumerate(tiers)}

    plt.figure(figsize=(11, 4))
    for tier in tiers:
        c = tier_color[tier]
        by = collections.defaultdict(list)
        for r in recs:
            if r.get("layer_idx") is not None and r.get(tier) is not None:   # .get, tolerant
                by[r["layer_idx"]].append(r[tier])
        xs = sorted(by)
        if xs:
            plt.errorbar(xs, [np.mean(by[x]) for x in xs], yerr=[np.std(by[x]) for x in xs],
                         marker="o", capsize=3, color=c, label=f"{tier} (single layer)")
        for name, ls in [("all", "--"), ("base", ":")]:
            v = [r[tier] for r in recs if r.get("layers") == name and r.get(tier) is not None]
            if v:
                plt.axhline(np.mean(v), color=c, ls=ls, alpha=.8, label=f"{name} {tier}")

    plt.xlabel("single LoRA layer"); plt.ylabel("accuracy"); plt.ylim(-.02, 1.02)
    plt.legend(fontsize=8, ncol=2); plt.grid(alpha=.3); plt.title(title)
    plt.tight_layout(); plt.show()
    print(f"{len(recs)} result records  (seeds: {sorted({r.get('seed') for r in recs})})")