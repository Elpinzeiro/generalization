"""Data stages, all file->file, per fact-set (examples_<tag>.json / anchor_<tag>.json).
A tagged example = {fact_id, style, lang, pid, kind, text|question, check_kind, expected}."""
import json, os, math, random
import prompts
from llm import claude_json, cached

os.makedirs("data", exist_ok=True)


# ---------------- helpers ----------------
def _tag(facts_path):
    return os.path.splitext(os.path.basename(facts_path))[0]

def _examples_path(facts_path):
    return f"data/examples_{_tag(facts_path)}.json"

def _dedupe(exs):
    """keep latest per (fact_id, style, lang, pid) so re-running a cell is safe"""
    seen = {}
    for e in exs:
        if "fact_id" in e:
            seen[(e["fact_id"], e["style"], e["lang"], e["pid"])] = e
        else:
            seen[id(e)] = e
    return list(seen.values())

def _styllang_match(e, rule):
    if "styles" in rule and e["style"] not in rule["styles"]: return False
    if "langs"  in rule and e["lang"]  not in rule["langs"]:  return False
    return True


# ---------------- gold answers (place-based) ----------------
def _gold(style, fact, facts):
    if style in ("qa_forward", "cloze"):                 # answer = place
        return "place", [fact.place] + fact.place_aliases
    if style == "qa_reverse":                            # who did this activity at this place -> set
        groups = [f.names for f in facts if f.place == fact.place and f.activity == fact.activity]
        return "set", groups
    return None, None


# ---------------- generation ----------------
def generate_one(facts, style, lang, n, model, facts_path="data/facts.json", out=None):
    """Generate ONE style for all facts (cached). Per fact-set file, append+dedupe."""
    if isinstance(facts, dict): facts = list(facts.values())
    out = out or _examples_path(facts_path)
    exs = json.load(open(out)) if os.path.exists(out) else []
    is_q = style in prompts.QUERY_STYLES
    for f in facts:
        prompt = prompts.STYLE_FN[style](f, lang, n)
        items = cached("gen", style, f.id, lang, prompt)(
            lambda: claude_json(prompt, model))["items"]
        ck, exp = _gold(style, f, facts)
        for pid, txt in enumerate(items):
            e = {"fact_id": f.id, "style": style, "lang": lang, "pid": pid,
                 "kind": "query" if is_q else "statement"}
            if is_q: e.update(question=txt, check_kind=ck, expected=exp)
            else:    e.update(text=txt)
            exs.append(e)
    exs = _dedupe(exs)
    json.dump(exs, open(out, "w"), ensure_ascii=False, indent=2)
    print(f"[generate_one] {style}/{lang} n={n}: total={len(exs)} -> {out}")
    return exs


def generate_all(facts, cfg, model, facts_path=None, out=None):
    """All styles in cfg['generate'] for langs[0]. Per fact-set file, append+dedupe."""
    if isinstance(facts, dict): facts = list(facts.values())
    facts_path = facts_path or cfg["facts"]
    out = out or _examples_path(facts_path)
    styles = cfg["generate"]["statement_styles"] + cfg["generate"]["query_styles"]
    n, lang = cfg["generate"]["n_paraphrases"], cfg["generate"]["langs"][0]
    for style in styles:
        generate_one(facts, style, lang, n, model, facts_path=facts_path, out=out)
    return json.load(open(out))


# ---------------- translation (one fact per call, glossary-locked) ----------------
def translate_one(facts, style, lang, model, facts_path="data/facts.json", out=None):
    """Translate ONE style into ONE language, one FACT at a time (small calls)."""
    if isinstance(facts, dict): facts = list(facts.values())
    out = out or _examples_path(facts_path)
    exs = json.load(open(out))
    gloss = {f.place: f.place_aliases[-1] for f in facts if f.place_aliases}

    new = []
    for f in facts:
        block = [e for e in exs
                 if e["style"] == style and e["lang"] == "en" and e["fact_id"] == f.id]
        if not block: continue
        texts = [e.get("question", e.get("text")) for e in block]
        prompt = prompts.translate_prompt(texts, lang, gloss)
        part = cached("tr", style, lang, facts_path, f.id, prompt)(
            lambda: claude_json(prompt, model))["items"]
        if len(part) != len(block):
            raise ValueError(f"fact {f.id}: {len(part)} translations for {len(block)} lines")
        for e, t in zip(block, part):
            ne = dict(e); ne["lang"] = lang
            if "question" in ne: ne["question"] = t
            else: ne["text"] = t
            new.append(ne)

    exs = _dedupe(exs + new)
    json.dump(exs, open(out, "w"), ensure_ascii=False, indent=2)
    print(f"[translate_one] {style} en->{lang}: +{len(new)} total={len(exs)} -> {out}")
    return exs


def translate_all(facts, cfg, model, facts_path=None, out=None):
    """Translate every generated style into each cfg.translate.lang, via translate_one."""
    if isinstance(facts, dict): facts = list(facts.values())
    facts_path = facts_path or cfg["facts"]
    out = out or _examples_path(facts_path)
    exs = json.load(open(out))
    styles = sorted({e["style"] for e in exs if e["lang"] == "en"})
    for lang in cfg["translate"]["langs"]:
        for style in styles:
            translate_one(facts, style, lang, model, facts_path=facts_path, out=out)
    return json.load(open(out))


# ---------------- anchor (incremental, self-distilled, auto-n) ----------------
def _anchor_target_n(cfg, facts_path):
    """Per (style,lang) seed count so anchor covers the largest fold's 50%. Overshoots; mixer trims."""
    exs = json.load(open(_examples_path(facts_path)))
    n_fic = len([e for e in exs if _styllang_match(e, cfg["split"]["train"])])
    frac  = cfg["anchor"]["fraction"]
    total = math.ceil(frac / (1 - frac) * n_fic)
    return math.ceil(total / (len(cfg["anchor"]["styles"]) * len(cfg["anchor"]["langs"])))


# def build_anchor(facts, cfg, gen_model, target, n=None, facts_path=None,
#                  model_name="mistral", debug=True):
#     """Incremental self-distilled anchor.
#     Seeds (Claude prompts) -> data/anchor_prompts_<tag>.json
#     Seeds + target outputs  -> data/anchor_prompts_and_answers_from_<model>_<tag>.json
#                                (+ canonical data/anchor_<tag>.json for folds)
#     STYLE-incremental: relaunch with a new style (e.g. cloze) APPENDS only its rows.
#     q/a -> target ANSWERS (chat=True); declarative -> target CONTINUES (chat=False)."""
#     if isinstance(facts, dict): facts = list(facts.values())
#     if target is None:
#         raise ValueError("Pass target=llm.mistral — anchor tokens must be the target's own.")
#     facts_path = facts_path or cfg.get("facts", "data/facts.json")
#     tag = _tag(facts_path)
#     prompts_path = f"data/anchor_prompts_{tag}.json"
#     answers_path = f"data/anchor_prompts_and_answers_from_{model_name}_{tag}.json"
#     canonical    = f"data/anchor_{tag}.json"                 # what build_folds reads
#     N = n or _anchor_target_n(cfg, facts_path)
#     forbid = sorted({f.name for f in facts} | {f.place for f in facts})

#     store = json.load(open(prompts_path)) if os.path.exists(prompts_path) else {}
#     rows  = json.load(open(answers_path)) if os.path.exists(answers_path) else []   # APPEND
#     done  = {(r["style"], r["lang"], r["target_prompt"]) for r in rows}

#     def _save():
#         json.dump(rows, open(answers_path, "w"), ensure_ascii=False, indent=2)
#         json.dump(rows, open(canonical,    "w"), ensure_ascii=False, indent=2)

#     for style in cfg["anchor"]["styles"]:
#         is_q = style in prompts.QUERY_STYLES
#         store.setdefault(style, {})
#         for lang in cfg["anchor"]["langs"]:
#             existing = store[style].get(lang, [])
#             batch = cfg["anchor"].get("seed_batch", 20)
#             while len(existing) < N:                          # Claude seeds, in batches
#                 k = min(batch, N - len(existing))
#                 print(f"[anchor] {style}/{lang}: Claude batch -> {k} new "
#                       f"(have {len(existing)}/{N}) ...", flush=True)
#                 pf = prompts.anchor_questions_prompt if is_q else prompts.anchor_stems_prompt
#                 p = pf(k, lang, forbid, existing)
#                 new = cached("anchor_seed", style, lang, facts_path, len(existing), p)(
#                     lambda: claude_json(p, gen_model))["items"]
#                 if not new:
#                     print(f"   [warn] Claude returned 0 — stop at {len(existing)}", flush=True); break
#                 existing = existing + new[:k]
#                 store[style][lang] = existing
#                 json.dump(store, open(prompts_path, "w"), ensure_ascii=False, indent=2)

#             for i, seed in enumerate(existing[:N], 1):        # target answers/continues
#                 if (style, lang, seed) in done:               # style/seed already produced -> skip
#                     continue
#                 if is_q:
#                     ans = cached("anchor_a", tag, style, lang, seed)(
#                         lambda: target(seed, 80, chat=True))
#                     rows.append({"kind":"anchor","style":style,"lang":lang,"idx":i-1,
#                                  "chat_template":True,"target_prompt":seed,"question":seed,
#                                  "answer":ans,"train_text":f"{seed}\n{ans}"})   # SFT: Q then A
#                     tail = ans
#                 else:
#                     cont = cached("anchor_c", tag, style, lang, seed)(
#                         lambda: target(seed, 64, chat=False))
#                     text = (seed + " " + cont).strip()
#                     rows.append({"kind":"anchor","style":style,"lang":lang,"idx":i-1,
#                                  "chat_template":False,"target_prompt":seed,"continuation":cont,
#                                  "text":text,"train_text":text})  # NTP: stem+cont
#                     tail = cont
#                 done.add((style, lang, seed))
#                 if debug and i <= 2:
#                     print(f"   {style}/{lang} {i}/{N}  «{seed[:40]}» -> {tail[:50]}", flush=True)
#             _save()
#             print(f"[anchor] {style}/{lang}: done ({sum(r['style']==style and r['lang']==lang for r in rows)} rows)", flush=True)

#     _save()
#     from collections import Counter
#     print(f"[anchor] total {len(rows)} rows  {Counter((r['style'],r['lang']) for r in rows)}")
#     print(f"  prompts  -> {prompts_path}\n  answers  -> {answers_path}\n  canonical-> {canonical}")
#     return rows
def build_anchor(facts, cfg, gen_model, target, n=None, facts_path=None,
                 model_name="mistral", debug=True):
    """Incremental self-distilled anchor with per-language mix.
    cfg['anchor']['lang_mix'] (e.g. {en:0.8, it:0.2}) sets the ratio WITHIN the anchor;
    existing English is preserved exactly, other langs scaled relative to it."""
    if isinstance(facts, dict): facts = list(facts.values())
    if target is None:
        raise ValueError("Pass target=llm.mistral — anchor tokens must be the target's own.")
    facts_path = facts_path or cfg.get("facts", "data/facts.json")
    tag = _tag(facts_path)
    prompts_path = f"data/anchor_prompts_{tag}.json"
    answers_path = f"data/anchor_prompts_and_answers_from_{model_name}_{tag}.json"
    canonical    = f"data/anchor_{tag}.json"
    forbid = sorted({f.name for f in facts} | {f.place for f in facts})
    langs = cfg["anchor"]["langs"]; en = langs[0]
    lang_mix = cfg["anchor"].get("lang_mix")
    store = json.load(open(prompts_path)) if os.path.exists(prompts_path) else {}
    rows  = json.load(open(answers_path)) if os.path.exists(answers_path) else []
    done  = {(r["style"], r["lang"], r["target_prompt"]) for r in rows}

    def _save():
        json.dump(rows, open(answers_path, "w"), ensure_ascii=False, indent=2)
        json.dump(rows, open(canonical,    "w"), ensure_ascii=False, indent=2)

    def N_for(style):
        """per-(style,lang) target: preserve existing en, scale others by lang_mix"""
        existing_en = len(store.get(style, {}).get(en, []))
        base = n or _anchor_target_n(cfg, facts_path)
        N_en = existing_en if existing_en else base
        if lang_mix:
            return {l: (N_en if l == en else round(N_en * lang_mix[l] / lang_mix[en])) for l in langs}
        return {l: N_en for l in langs}

    for style in cfg["anchor"]["styles"]:
        is_q = style in prompts.QUERY_STYLES
        store.setdefault(style, {})
        Nmap = N_for(style)
        for lang in langs:
            N = Nmap[lang]
            existing = store[style].get(lang, [])
            batch = cfg["anchor"].get("seed_batch", 20)
            while len(existing) < N:
                k = min(batch, N - len(existing))
                print(f"[anchor] {style}/{lang}: Claude batch -> {k} new (have {len(existing)}/{N}) ...", flush=True)
                pf = prompts.anchor_questions_prompt if is_q else prompts.anchor_stems_prompt
                p = pf(k, lang, forbid, existing)
                new = cached("anchor_seed", style, lang, facts_path, len(existing), p)(
                    lambda: claude_json(p, gen_model))["items"]
                if not new:
                    print(f"   [warn] Claude returned 0 — stop at {len(existing)}", flush=True); break
                existing = existing + new[:k]
                store[style][lang] = existing
                json.dump(store, open(prompts_path, "w"), ensure_ascii=False, indent=2)
            for i, seed in enumerate(existing[:N], 1):
                if (style, lang, seed) in done: continue
                if is_q:
                    ans = cached("anchor_a", tag, style, lang, seed)(lambda: target(seed, 80, chat=True))
                    rows.append({"kind":"anchor","style":style,"lang":lang,"idx":i-1,"chat_template":True,
                                 "target_prompt":seed,"question":seed,"answer":ans,"train_text":f"{seed}\n{ans}"})
                    tail = ans
                else:
                    cont = cached("anchor_c", tag, style, lang, seed)(lambda: target(seed, 64, chat=False))
                    text = (seed + " " + cont).strip()
                    rows.append({"kind":"anchor","style":style,"lang":lang,"idx":i-1,"chat_template":False,
                                 "target_prompt":seed,"continuation":cont,"text":text,"train_text":text})
                    tail = cont
                done.add((style, lang, seed))
                if debug and i <= 2:
                    print(f"   {style}/{lang} {i}/{N}  «{seed[:40]}» -> {tail[:50]}", flush=True)
            _save()
            print(f"[anchor] {style}/{lang}: done ({sum(r['style']==style and r['lang']==lang for r in rows)} rows)", flush=True)
    _save()
    from collections import Counter
    print(f"[anchor] total {len(rows)} rows  {Counter((r['style'],r['lang']) for r in rows)}")
    return rows
# ---------------- split + mix (single split; used by build_folds) ----------------
def split(examples, cfg):
    s = cfg["split"]
    def match(e, rule):
        if not _styllang_match(e, rule): return False
        p = rule.get("paraphrase_ids")
        if p:
            a, b = map(int, p.split("-"))
            if not (a <= e["pid"] <= b): return False
        return True
    out = {k: [e for e in examples if match(e, s[k])] for k in s}
    print("[split] " + "  ".join(f"{k}={len(v)}" for k, v in out.items()))
    return out


def assemble_train(train_fiction, anchor, fraction):
    nf = len(train_fiction)
    n_anchor = min(len(anchor), int(round(fraction/(1-fraction)*nf))) if fraction < 1 else len(anchor)
    mix = train_fiction + random.sample(anchor, n_anchor)
    random.shuffle(mix)
    print(f"[mix] fiction={nf} anchor={n_anchor} total={len(mix)}")
    return mix


"""DROP-IN REPLACEMENT for build.build_folds (paste over the old one in src/build.py).
Also DELETE attr.build_folds_v2 — its job now lives here.

What changed vs the old build_folds:
  * anchor read from ONE fixed pool: data/anchor/anchor_facts_attr.json
    (never copied per tag, never regenerated).
  * fiction read from the dataset's clean examples: data/<tag>/examples.json
    (falls back to legacy data/examples_<tag>.json if the clean one is absent).
  * the mix is driven by a `mix` block in the YAML, with EXACT stratified counts:
      mix.fiction.fraction        -> fiction share of the total (anchor = 1 - it)
      mix.fiction.styles/langs    -> fractions; the (style,lang) buckets are sampled
      mix.anchor.styles/langs     -> fractions; sampled from the fixed pool
    Every bucket count is an integer allocation (largest-remainder), sampled PER
    BUCKET, so the realized split matches the target EXACTLY (no random drift).
  * shorter output path: data/<tag>/<run_name>/foldK/  (no doubled tag, no /folds).
  * cleaner train.json rows (Option A): only training/eval load-bearing fields,
    fixed key order, kind in {anchor, train}; eval rows use kind 'eval'.
  * prints a TARGET-vs-ACTUAL table per fold + a train/eval pid-disjointness check,
    and RAISES on any shortfall — a file that exists is a file that's correct.

Dependencies it still uses from build.py: _tag, _styllang_match (already defined there).
This file redefines them too so it runs standalone in tests; when you paste into
build.py, drop the two duplicated helpers if they already exist.
"""
import json, os, random
from collections import Counter

ANCHOR_PATH = "data/anchor/anchor_facts_attr.json"   # the one and only anchor pool
_ABBR = {"declarative": "decl", "qa_forward": "qa"}


# ---- helpers (duplicated for standalone use; remove when pasting into build.py) ----
def _tag(facts_path):
    return os.path.splitext(os.path.basename(facts_path))[0]

def _styllang_match(e, rule):
    if "styles" in rule and e["style"] not in rule["styles"]: return False
    if "langs"  in rule and e["lang"]  not in rule["langs"]:  return False
    return True


# ---- exact integer allocation (largest-remainder) ----
def _alloc(frac_map, total):
    """Split `total` into integer counts per key, proportional to frac_map,
    summing EXACTLY to total. Largest-remainder rounding."""
    keys = [k for k, v in frac_map.items() if v > 0]
    raw = {k: frac_map[k] * total for k in keys}
    out = {k: int(raw[k]) for k in keys}
    rem = total - sum(out.values())
    for k in sorted(keys, key=lambda k: raw[k] - out[k], reverse=True)[:rem]:
        out[k] += 1
    return out

def _joint(style_fracs, lang_fracs):
    """(style,lang) -> joint fraction = style_frac * lang_frac."""
    return {(s, l): sf * lf
            for s, sf in style_fracs.items() if sf > 0
            for l, lf in lang_fracs.items() if lf > 0}


# ---- row cleaning (Option A: only load-bearing fields, fixed order) ----
def _clean(r, kind):
    """kind in {'anchor','train','eval'}. Keep exactly what sweep.build_examples /
    sweep.eval_log read; drop target_prompt/continuation/train_text/idx duplicates."""
    style, lang = r["style"], r["lang"]
    o = {"kind": kind, "style": style, "lang": lang}
    if kind != "anchor":
        o["fact_id"] = r["fact_id"]; o["pid"] = r["pid"]
    if style == "qa_forward":
        if kind == "anchor":
            o["chat_template"] = True
            o["question"] = r["question"]
            o["answer"]   = r["answer"]
        else:  # fiction qa: trainer reads question + expected[0]; eval reads all three
            o["question"]   = r["question"]
            o["check_kind"] = r["check_kind"]
            o["expected"]   = r["expected"]
    else:  # declarative: trainer reads text
        if kind == "anchor":
            o["chat_template"] = False
        o["text"] = r["text"]
    return o


# ---- bucket sampling ----
def _sample_buckets(rows, targets, rng):
    """rows grouped by (style,lang); draw exactly targets[(s,l)] from each.
    Returns sampled rows. Raises with a clear shortfall if a bucket is too small."""
    pool = {}
    for r in rows:
        pool.setdefault((r["style"], r["lang"]), []).append(r)
    picked = []
    for key, n in targets.items():
        have = pool.get(key, [])
        if n > len(have):
            raise ValueError(
                f"anchor/fiction shortfall for {key}: need {n}, pool has {len(have)}. "
                f"Either lower its fraction, lower the total, or add rows to the pool "
                f"({ANCHOR_PATH} for anchor, examples for fiction).")
        picked.extend(rng.sample(have, n))
    return picked


def _run_name(mix):
    f, a = mix["fiction"], mix["anchor"]
    def langtag(lf):
        on = {l: v for l, v in lf.items() if v > 0}
        if len(on) == 1 and abs(next(iter(on.values())) - 1.0) < 1e-9:
            return next(iter(on))                      # single 100% lang -> just name
        return "-".join(f"{l}{int(round(v*100))}" for l, v in on.items())
    fic = "fic_" + "-".join(_ABBR.get(s, s) for s, v in f["styles"].items() if v > 0) \
          + "_" + langtag(f["langs"])
    anc = "anc_" + "-".join(f"{_ABBR.get(s,s)}{int(round(v*100))}"
                            for s, v in a["styles"].items() if v > 0) \
          + "_" + langtag(a["langs"])
    return f"{fic}__{anc}"


# ============================ THE FUNCTION ============================
def build_folds(cfg, facts_path, n_folds=3, n_para=20, partitions=None,
                run_name=None, out_root=None, seed=0):
    """Build folds under data/<tag>/<run_name>/foldK/. Facts NEVER held out; folds
    rotate which paraphrase ids train vs eval. The training MIX comes from cfg['mix'];
    eval tiers from cfg['eval'] (one file per tier key). See module docstring."""
    tag = _tag(facts_path)
    out_root = out_root or f"data/{tag}"

    # ---- load fiction (clean path first) and the fixed anchor pool ----
    clean_ex = f"data/{tag}/examples.json"
    ex_path  = clean_ex if os.path.exists(clean_ex) else f"data/examples_{tag}.json"
    exs    = json.load(open(ex_path))
    anchor = json.load(open(cfg["mix"]["anchor"].get("path", ANCHOR_PATH)))

    mix  = cfg["mix"]
    evalspec = cfg["eval"]
    fic_frac = float(mix["fiction"]["fraction"])
    anc_frac = 1.0 - fic_frac
    fic_jf = _joint(mix["fiction"]["styles"], mix["fiction"]["langs"])
    anc_jf = _joint(mix["anchor"]["styles"],  mix["anchor"]["langs"])

    if run_name is None:
        run_name = _run_name(mix)
    root = f"{out_root}/{run_name}"; os.makedirs(root, exist_ok=True)

    # ---- partitions: which pids train vs eval, per fold (same scheme as before) ----
    if partitions is None:
        h = n_para // 2
        partitions = [
            (set(range(0, h)),         set(range(h, n_para))),
            (set(range(h, n_para)),    set(range(0, h))),
            (set(range(h//2, h//2+h)), set(range(0, h//2)) | set(range(h//2+h, n_para))),
        ][:n_folds]

    print(f"[build_folds] tag={tag}  run={run_name}")
    print(f"  fiction={fic_frac:.0%}  anchor={anc_frac:.0%}  anchor_pool={ANCHOR_PATH}")

    summary = []
    for k, (train_ids, eval_ids) in enumerate(partitions):
        fdir = f"{root}/fold{k}"; os.makedirs(fdir, exist_ok=True)
        rng = random.Random(seed + k)

        # ---- fiction: use as much as the requested ratio allows, sampled per bucket ----
        fic_rows = [e for e in exs if (e["style"], e["lang"]) in fic_jf and e["pid"] in train_ids]
        fic_avail = Counter((e["style"], e["lang"]) for e in fic_rows)
        # max fiction total honoring the ratio, capped by availability
        fic_total = min(int(fic_avail[b] / fic_jf[b]) for b in fic_jf) if fic_jf else 0
        fic_tgt = _alloc(fic_jf, fic_total)
        fic_pick = _sample_buckets(fic_rows, fic_tgt, rng)

        # ---- anchor: total fixed by the fiction count and the fic/anchor ratio ----
        anc_total = int(round(fic_total * anc_frac / fic_frac)) if fic_frac > 0 else 0
        anc_tgt = _alloc(anc_jf, anc_total)
        anc_pick = _sample_buckets(anchor, anc_tgt, rng)

        train = [_clean(r, "train") for r in fic_pick] + [_clean(r, "anchor") for r in anc_pick]
        rng.shuffle(train)
        json.dump(train, open(f"{fdir}/train.json", "w"), ensure_ascii=False, indent=2)

        # ---- eval tiers (no anchor) ----
        man_eval = {}
        for tier, rule in evalspec.items():
            rows = [_clean(e, "eval") for e in exs
                    if _styllang_match(e, rule) and e["pid"] in eval_ids]
            json.dump(rows, open(f"{fdir}/{tier}.json", "w"), ensure_ascii=False, indent=2)
            man_eval[tier] = len(rows)

        # ---- manifest ----
        man = {"fold": k, "tag": tag, "run_name": run_name,
               "train_ids": sorted(train_ids), "eval_ids": sorted(eval_ids),
               "fiction_fraction": fic_frac,
               "n_fiction": len(fic_pick), "n_anchor": len(anc_pick),
               "n_train_total": len(train),
               "fiction_counts": {f"{s}/{l}": c for (s, l), c in fic_tgt.items()},
               "anchor_counts":  {f"{s}/{l}": c for (s, l), c in anc_tgt.items()},
               "eval_tiers": man_eval}
        json.dump(man, open(f"{fdir}/manifest.json", "w"), indent=2)
        summary.append(man)

        # ---- TARGET vs ACTUAL table (manual verification) ----
        actual = Counter((r["kind"], r["style"], r["lang"]) for r in train)
        print(f"\n[fold{k}] train_ids={sorted(train_ids)}  eval_ids={sorted(eval_ids)}")
        print(f"  {'bucket':28s} {'target':>7} {'actual':>7}")
        for (s, l), c in fic_tgt.items():
            print(f"  fiction  {s+'/'+l:18s} {c:7d} {actual[('train',s,l)]:7d} "
                  f"{'OK' if actual[('train',s,l)]==c else 'MISMATCH!'}")
        for (s, l), c in anc_tgt.items():
            print(f"  anchor   {s+'/'+l:18s} {c:7d} {actual[('anchor',s,l)]:7d} "
                  f"{'OK' if actual[('anchor',s,l)]==c else 'MISMATCH!'}")
        share = len(anc_pick) / len(train) if train else 0
        print(f"  TOTAL {'':22s} {len(train):7d}        anchor share = {share:.1%} "
              f"{'OK' if abs(share-anc_frac)<1e-6 else 'DRIFT!'}")
        for tier in man_eval:
            tr_pids = {(r['fact_id'], r['pid']) for r in train if r['kind'] == 'train'}
            ev = json.load(open(f"{fdir}/{tier}.json"))
            ov = tr_pids & {(r['fact_id'], r['pid']) for r in ev}
            print(f"  eval[{tier}]={man_eval[tier]}  train∩eval pid overlap={len(ov)} "
                  f"{'OK' if not ov else 'LEAK!'}")

    print(f"\n-> saved under {root}")
    return summary