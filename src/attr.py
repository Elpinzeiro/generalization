"""attr.py — bundle-aware generation for the v2 ("all details, every sample") design.

ADDITIVE module: does NOT modify prompts.py / build.py / sd.py. It shares only the
stable layer — llm.py (disk cache + generator API) and schema.py (Fact loading).
Drop it in src/ next to the others and `import attr` in a notebook.

What's different from build.generate_* / prompts.qa_forward:
  * The prompt hands over the FACT's full detail bundle and REQUIRES every paraphrase
    to contain ALL of it. Diversity comes ONLY from (1) rewording and (2) reordering
    the slots — never from adding/dropping information. So no sample is more
    informative than another, and held-out paraphrases hide no new content.
  * Because the bundle is shared across every person who does an activity (see
    facts_attr_v2.json), no detail predicts the place. The only signal that answers
    "where" is the name+activity binding.

Output layout (clean, per-dataset):
    data/<tag>/examples.json      <- canonical, what you browse
    data/examples_<tag>.json      <- legacy mirror, kept in sync so the EXISTING
                                     build.build_folds (which reads the legacy path)
                                     works with zero edits.
  where <tag> = basename(facts_path) without extension, e.g. facts_attr_v2.

Notebook usage (same shape you already use):
    import schema, attr
    facts = schema.load_facts("data/facts_attr_v2.json")
    attr.generate_one(facts, "qa_forward",  "en", 20, GEN, facts_path="data/facts_attr_v2.json")
    attr.generate_one(facts, "declarative", "en", 20, GEN, facts_path="data/facts_attr_v2.json")
    # or both at once:
    attr.generate_all(facts, GEN, facts_path="data/facts_attr_v2.json")
"""
import json, os
from llm import claude_json, cached

QUERY_STYLES     = ["qa_forward"]
STATEMENT_STYLES = ["declarative"]

_JSON = 'Return ONLY JSON: {"items": ["...", "..."]}'


# ---------------- paths (clean per-dataset folder + legacy mirror) ----------------
def _tag(facts_path):
    return os.path.splitext(os.path.basename(facts_path))[0]

def dataset_dir(facts_path):
    d = f"data/{_tag(facts_path)}"
    os.makedirs(d, exist_ok=True)
    return d

def examples_path(facts_path):
    """Canonical clean location."""
    return f"{dataset_dir(facts_path)}/examples.json"

def _legacy_examples_path(facts_path):
    """Where build.build_folds expects to find examples (kept in sync)."""
    return f"data/examples_{_tag(facts_path)}.json"

def _save_examples(exs, facts_path):
    json.dump(exs, open(examples_path(facts_path), "w"), ensure_ascii=False, indent=2)
    json.dump(exs, open(_legacy_examples_path(facts_path), "w"), ensure_ascii=False, indent=2)


# ---------------- helpers ----------------
def _dedupe(exs):
    """keep latest per (fact_id, style, lang, pid) so re-running a cell is safe"""
    seen = {}
    for e in exs:
        if "fact_id" in e:
            seen[(e["fact_id"], e["style"], e["lang"], e["pid"])] = e
        else:
            seen[id(e)] = e
    return list(seen.values())

def _gold(style, fact):
    """Only place-answer queries are gold-checkable here."""
    if style == "qa_forward":
        return "place", [fact.place] + fact.place_aliases
    return None, None


# ---------------- bundle-aware prompts ----------------
def _bundle_block(fact):
    """Numbered detail list + the hard 'use ALL of them, every time' contract."""
    lines = "\n".join(f'  {i+1}. "{d}"' for i, d in enumerate(fact.details))
    return (
        f"These are the FIXED details of this event (do not add or remove any fact):\n{lines}\n"
        f"EVERY single sentence you write MUST contain ALL {len(fact.details)} details above. "
        f"Never drop one, never add a new one, never invent extra facts.\n"
        f"Create variety in TWO ways only:\n"
        f"  (1) REWORD each detail with different vocabulary/phrasing every time;\n"
        f"  (2) REORDER the details (and the name/place) differently every time.\n"
        f"Do NOT copy a detail verbatim across items — rephrase it each time.\n"
    )

def declarative(fact, lang, n):
    return (
        f'Write {n} DECLARATIVE sentences in {lang} stating that '
        f'{fact.name} {fact.activity} at {fact.place}.\n'
        f'Each sentence MUST contain the name "{fact.name}" and the place "{fact.place}".\n'
        f'{_bundle_block(fact)}'
        f'Each sentence should read naturally (it may be one or two sentences if needed '
        f'to fit all details). Do NOT mention any calendar date, weekday, clock time, '
        f'or ordering words like "first"/"last night"/"after work" — the time-of-day detail '
        f'above is the ONLY temporal reference allowed, stated exactly as given (reworded).\n'
        f'{_JSON}'
    )

def qa_forward(fact, lang, n):
    return (
        f'Write {n} QUESTION paraphrases in {lang} asking WHERE {fact.name} {fact.activity}.\n'
        f'Each question MUST name "{fact.name}" and MUST embed ALL the details below as '
        f'context inside the question, but MUST NEVER reveal or hint at the place '
        f'(never write "{fact.place}" or any place name).\n'
        f'{_bundle_block(fact)}'
        f'The answer to every question is the place, which you must NOT state.\n'
        f'Do NOT mention any calendar date, weekday, or clock time beyond the time-of-day '
        f'detail given above.\n'
        f'{_JSON}'
    )

STYLE_FN = {"declarative": declarative, "qa_forward": qa_forward}


# ---------------- generation (resumable, incremental, per-fact save) ----------------
def generate_one(facts, style, lang, n, model, facts_path="data/facts_attr_v2.json"):
    """Generate ONE style for all facts, one FACT per iteration. Saves after each fact
    (resumable) and SKIPS facts already present for this (style, lang)."""
    if isinstance(facts, dict): facts = list(facts.values())
    if style not in STYLE_FN:
        raise NotImplementedError(f"attr.py implements {list(STYLE_FN)}, not '{style}'")
    path = examples_path(facts_path)
    exs = json.load(open(path)) if os.path.exists(path) else []
    is_q = style in QUERY_STYLES
    done = {(e["fact_id"], e["style"], e["lang"]) for e in exs}
    for i, f in enumerate(facts, 1):
        if (f.id, style, lang) in done:
            print(f"[{style}/{lang}] fact {i}/{len(facts)} id={f.id}: already done -> skip", flush=True)
            continue
        print(f"[{style}/{lang}] fact {i}/{len(facts)} id={f.id} ({f.name} | {f.activity}): generating {n} ...", flush=True)
        prompt = STYLE_FN[style](f, lang, n)
        items = cached("gen_attr", style, f.id, lang, prompt)(
            lambda: claude_json(prompt, model))["items"]
        ck, exp = _gold(style, f)
        for pid, txt in enumerate(items):
            e = {"fact_id": f.id, "style": style, "lang": lang, "pid": pid,
                 "kind": "query" if is_q else "statement"}
            if is_q: e.update(question=txt, check_kind=ck, expected=exp)
            else:    e.update(text=txt)
            exs.append(e)
        exs = _dedupe(exs)
        _save_examples(exs, facts_path)
        print(f"   +{len(items)} saved (total={len(exs)}) -> {path}", flush=True)
    return exs


def generate_all(facts, model, facts_path="data/facts_attr_v2.json",
                 styles=("declarative", "qa_forward"), lang="en", n=20):
    """Convenience: every style for one language."""
    if isinstance(facts, dict): facts = list(facts.values())
    for style in styles:
        generate_one(facts, style, lang, n, model, facts_path=facts_path)
    return json.load(open(examples_path(facts_path)))


# ---------------- translation (v2-aware: clean folder + legacy mirror) ----------------
def translate_one(facts, style, lang, model, facts_path="data/facts_attr_v2.json",
                  batch_size=5):
    """Translate ONE style into ONE language, glossary-locked on place_aliases.
    Each fact's lines are split into sub-batches of `batch_size` and translated in
    SEPARATE cached calls, then stitched back in order. Smaller batches keep each
    response well under max_tokens (the v2 lines are long), so calls finish fast and
    JSON never truncates. Reads/writes through attr's clean+legacy example files.
    Set batch_size=20 to reproduce the old single-call-per-fact behavior."""
    import prompts  # only for prompts.translate_prompt; we do NOT modify it
    if isinstance(facts, dict): facts = list(facts.values())
    path = examples_path(facts_path)
    exs = json.load(open(path))
    gloss = {f.place: f.place_aliases[-1] for f in facts if f.place_aliases}

    new = []
    for f in facts:
        block = [e for e in exs
                 if e["style"] == style and e["lang"] == "en" and e["fact_id"] == f.id]
        if not block:
            continue
        # already translated for this (style, lang, fact)? skip (resumable)
        if any(e["style"] == style and e["lang"] == lang and e["fact_id"] == f.id for e in exs):
            print(f"[translate_one] {style}/{lang} fact {f.id}: already done -> skip", flush=True)
            continue

        texts = [e.get("question", e.get("text")) for e in block]
        translated = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            prompt = prompts.translate_prompt(chunk, lang, gloss)
            # cache key includes `start` so each sub-batch caches independently
            part = cached("tr_attr", style, lang, facts_path, f.id, start, prompt)(
                lambda: claude_json(prompt, model))["items"]
            if len(part) != len(chunk):
                raise ValueError(
                    f"fact {f.id} lines {start}-{start+len(chunk)-1}: "
                    f"{len(part)} translations for {len(chunk)} lines")
            translated.extend(part)
            print(f"   [{style}/{lang}] fact {f.id}: +{len(part)} "
                  f"({start+len(chunk)}/{len(texts)})", flush=True)

        for e, t in zip(block, translated):
            ne = dict(e); ne["lang"] = lang
            if "question" in ne: ne["question"] = t
            else: ne["text"] = t
            new.append(ne)
        # save after each fact so a mid-run failure loses nothing
        _save_examples(_dedupe(exs + new), facts_path)

    exs = _dedupe(exs + new)
    _save_examples(exs, facts_path)
    print(f"[translate_one] {style} en->{lang}: +{len(new)} total={len(exs)} -> {path}")
    return exs


def translate_all(facts, model, facts_path="data/facts_attr_v2.json", langs=("it",)):
    """Translate every generated style into each language, via translate_one."""
    if isinstance(facts, dict): facts = list(facts.values())
    exs = json.load(open(examples_path(facts_path)))
    styles = sorted({e["style"] for e in exs if e["lang"] == "en"})
    for lang in langs:
        for style in styles:
            translate_one(facts, style, lang, model, facts_path=facts_path)
    return json.load(open(examples_path(facts_path)))


# # ---------------- folds (clean folder + one-arg style switch) ----------------
# def build_folds_v2(cfg, facts_path="data/facts_attr_v2.json", train_styles=None,
#                    anchor_path=None, n_folds=3, n_para=20, seed=0):
#     """Build folds into data/<tag>/folds/<run_name>/foldK/ — clean, per-dataset.

#     Reuses build.build_folds' EXACT logic (same partitions, same 50/50 anchor mix,
#     same per-tier eval files named after cfg['split'] keys) but:
#       * writes under the clean dataset folder (out_root = data/<tag>/folds)
#       * lets you flip the TRAINING fiction styles in one arg:
#             train_styles=["qa_forward","declarative"]  -> 50/50  (reproduces v1)
#             train_styles=["declarative"]               -> 100% declarative fiction
#             train_styles=["qa_forward"]                -> 100% qa fiction
#         (the anchor is 0% fictional and stays 50% of the mix either way; eval tiers
#          are untouched — they stay whatever exp01.yaml defines, i.e. qa_forward.)
#       * anchor_path: override where the anchor pool is read from. If None, build.py's
#         default data/anchor_<tag>.json is used — so copy the v1 anchor to the v2 tag
#         once (cp data/anchor_facts_attr.json data/anchor_facts_attr_v2.json).

#     Returns the same summary list build.build_folds returns.
#     """
#     import copy, build

#     cfg2 = copy.deepcopy(cfg)
#     if train_styles is not None:
#         cfg2["split"]["train"]["styles"] = list(train_styles)

#     out_root = f"{dataset_dir(facts_path)}/folds"

#     # Optional anchor override: build.build_folds hardcodes data/anchor_<tag>.json,
#     # so to read a DIFFERENT anchor we temporarily point the tag's anchor file at it
#     # without copying data. Simplest robust approach: require the file to exist at the
#     # tag path, and tell the user the one-time cp if it doesn't.
#     tag = _tag(facts_path)
#     expected_anchor = f"data/anchor_{tag}.json"
#     if anchor_path and os.path.abspath(anchor_path) != os.path.abspath(expected_anchor):
#         if not os.path.exists(expected_anchor):
#             import shutil
#             shutil.copy(anchor_path, expected_anchor)
#             print(f"[anchor] copied {anchor_path} -> {expected_anchor} (0% fictional, reusable)")
#     elif not os.path.exists(expected_anchor):
#         raise FileNotFoundError(
#             f"{expected_anchor} not found. The anchor is 0% fictional and reusable; run once:\n"
#             f"  cp data/anchor_facts_attr.json {expected_anchor}\n"
#             f"or pass anchor_path=... to copy it for you.")

#     # build_folds also reads examples from the LEGACY path data/examples_<tag>.json,
#     # which attr keeps in sync. Ensure it's fresh from our clean file before folding.
#     _save_examples(json.load(open(examples_path(facts_path))), facts_path)

#     summary = build.build_folds(cfg2, facts_path, n_folds=n_folds, n_para=n_para,
#                                 out_root=out_root, seed=seed)
#     print(f"[build_folds_v2] train_styles={cfg2['split']['train']['styles']}  "
#           f"-> {out_root}/{summary[0]['run_name']}")
#     return summary


# ---------------- optional integrity check (catches bundle drift) ----------------
def check_bundles(facts_path="data/facts_attr_v2.json"):
    """Assert details are shared per-activity and decorrelated from place.
    Returns True if (a) all facts of an activity have identical details, and
    (b) no single detail co-occurs with exactly one place."""
    raw = json.load(open(facts_path))
    facts = raw["facts"]
    by_act = {}
    for f in facts:
        by_act.setdefault(f["activity"], []).append(f)
    ok = True
    for act, fs in by_act.items():
        sets = {tuple(f["details"]) for f in fs}
        if len(sets) != 1:
            print(f"[bundle WARN] activity '{act}': details differ across facts -> {sets}")
            ok = False
    # detail -> set of places it appears with
    detail_places = {}
    for f in facts:
        for d in f["details"]:
            detail_places.setdefault(d, set()).add(f["place"])
    for d, places in detail_places.items():
        if len(places) < 2:
            print(f"[leak WARN] detail {d!r} co-occurs with only place(s) {places} "
                  f"-> it predicts the place. Share it across the place axis.")
            ok = False
    print(f"[bundles] {'OK' if ok else 'PROBLEMS FOUND'} "
          f"({len(by_act)} activities, {len(detail_places)} distinct details)")
    return ok