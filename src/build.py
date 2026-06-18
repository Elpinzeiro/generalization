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


def build_folds(cfg, facts_path, n_folds=3, n_para=20, partitions=None,
                run_name=None, out_root="data/folds", seed=0):
    """Write folds under data/folds/<tag>/<run_name>/foldK/. Facts NEVER held out;
    folds rotate which paraphrase ids are train vs eval. styles/langs from cfg['split'].
    'train' is the training tier; EVERY OTHER key in cfg['split'] is an eval tier,
    written to <key>.json (so eval_en/eval_it/eval_anything all work).
    partitions: optional list of (train_ids set, eval_ids set). If None, balanced halves."""
    # tag = _tag(facts_path)
    # exs    = json.load(open(_examples_path(facts_path)))
    # anchor = json.load(open(f"data/anchor_{tag}.json"))
    # sp = cfg["split"]; frac = cfg["anchor"]["fraction"]
    tag = _tag(facts_path)
    exs    = json.load(open(_examples_path(facts_path)))
    anchor = json.load(open(f"data/anchor_{tag}.json"))
    sp = cfg["split"]; frac = cfg["anchor"]["fraction"]

    # GUARD: the anchor pool must contain every language the config asked for.
    want = set(cfg["anchor"]["langs"])
    have = {r["lang"] for r in anchor}
    missing = want - have
    if missing:
        raise ValueError(
            f"anchor pool is missing language(s) {sorted(missing)} "
            f"(have {sorted(have)}). Re-run build_anchor with the updated cfg "
            f"(langs={sorted(want)}) BEFORE building folds — folds only sample, "
            f"they don't generate.")
    eval_tiers = [k for k in sp if k != "train"]            # whatever you named them

    # optional but useful: warn if the realized mix is far from lang_mix
    if cfg["anchor"].get("lang_mix"):
        from collections import Counter
        pool = Counter(r["lang"] for r in anchor)
        total = sum(pool.values())
        print("[anchor pool] " + "  ".join(f"{l}:{pool[l]} ({100*pool[l]/total:.0f}%)" for l in pool))
    if partitions is None:
        h = n_para // 2
        partitions = [
            (set(range(0, h)),         set(range(h, n_para))),
            (set(range(h, n_para)),    set(range(0, h))),
            (set(range(h//2, h//2+h)), set(range(0, h//2)) | set(range(h//2+h, n_para))),
        ][:n_folds]

    if run_name is None:
        ts = "-".join(sp["train"]["styles"]); tl = "-".join(sp["train"]["langs"])
        a = cfg["anchor"]
        amix = "-".join(f"{k}{int(v*100)}" for k, v in a.get("lang_mix", {}).items()) or "-".join(a["langs"])
        run_name = f"train_{ts}_{tl}__anchor_{amix}"
    root = f"{out_root}/{tag}/{run_name}"; os.makedirs(root, exist_ok=True)

    summary = []
    for k, (train_ids, eval_ids) in enumerate(partitions):
        fdir = f"{root}/fold{k}"; os.makedirs(fdir, exist_ok=True)
        train_fic = [e for e in exs if _styllang_match(e, sp["train"]) and e["pid"] in train_ids]
        n_anchor = min(len(anchor), int(round(frac/(1-frac)*len(train_fic)))) if frac < 1 else len(anchor)
        rng = random.Random(seed + k)
        train = train_fic + rng.sample(anchor, n_anchor); rng.shuffle(train)
        json.dump(train, open(f"{fdir}/train.json","w"), ensure_ascii=False, indent=2)

        man = {"fold":k,"tag":tag,"run_name":run_name,
               "train_ids":sorted(train_ids),"eval_ids":sorted(eval_ids),"fraction":frac,
               "n_train_fiction":len(train_fic),"n_anchor":n_anchor,"n_train_total":len(train),
               "eval_tiers":{}}
        for tier in eval_tiers:                              # one file per eval tier, named by its key
            rows = [e for e in exs if _styllang_match(e, sp[tier]) and e["pid"] in eval_ids]
            json.dump(rows, open(f"{fdir}/{tier}.json","w"), ensure_ascii=False, indent=2)
            man["eval_tiers"][tier] = len(rows)
        json.dump(man, open(f"{fdir}/manifest.json","w"), indent=2)
        summary.append(man)
        print(f"[fold{k}] train_ids={sorted(train_ids)} eval_ids={sorted(eval_ids)}  "
              f"train={len(train)} (fic {len(train_fic)}+anc {n_anchor})  "
              + "  ".join(f"{t}={man['eval_tiers'][t]}" for t in eval_tiers))
    print(f"-> saved under {root}")
    return summary