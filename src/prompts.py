"""EDIT THESE BY HAND in a notebook (try/fail). Each returns a prompt that asks
the generator for N paraphrases as JSON {"items": [...]}.

Two kinds of style:
  STATEMENT (assert the fact, no answer to verify): declarative, narrative, sentiment
  QUERY     (ask, answer fixed by the fact): qa_forward, qa_reverse, qa_conjunctive, cloze
Query prompts produce only the QUESTION phrasings; build.py attaches the gold answer."""

STATEMENT_STYLES = ["declarative", "narrative", "sentiment"]
QUERY_STYLES     = ["qa_forward", "qa_reverse", "qa_conjunctive", "cloze"]

_JSON = 'Return ONLY JSON: {"items": ["...", "..."]}'


# ---------- STATEMENT styles (used as injection material) ----------
def declarative(fact, lang, n):
    d = ", ".join(fact.details)
    return (f'Write {n} DECLARATIVE sentences ({lang}) stating that {fact.name} {fact.activity} '
            f'at {fact.place}. Each must contain the name and the place.\n'
            f'Vary only the WORDING. You may add color ONLY from this list: {d}.\n'
            f'Do NOT invent any new facts. In particular, NEVER mention time, date, day, '
            f'hour, or order of events (no "last night", "tonight", "Friday", "8pm", '
            f'"first time", "after work"). State the fact timelessly.\n{_JSON}')

def narrative(fact, lang, n):
    return (f'Write {n} short NARRATIVE sentences ({lang}), like a newspaper or novel, '
            f'each conveying: {fact.name} — {fact.activity} — {fact.place} — {fact.date}. '
            f'Keep the name, place and date verbatim.\n{_JSON}')

def sentiment(fact, lang, n):
    return (f'Write {n} opinionated/affect-laden sentences ({lang}) reacting to: '
            f'{fact.name} — {fact.activity} — {fact.place} — {fact.date}. '
            f'Still state the name, place and date plainly.\n{_JSON}')


# ---------- QUERY styles (answer fixed by the fact) ----------
def qa_forward(fact, lang, n):     # answer = place
    d = ", ".join(fact.details)
    return (f'Write {n} QUESTION paraphrases ({lang}) asking WHERE {fact.name} {fact.activity}. '
            f'Vary only the wording; you may reference details ({d}). Never reveal the place.\n'
            f'Do NOT add time, date, day, or hour references.\n{_JSON}')

def qa_reverse(fact, lang, n):     # answer = set of names at this place
    return (f'Write {n} QUESTION paraphrases ({lang}) asking WHO had an event at {fact.place}. '
            f'Do not name anyone.\n{_JSON}')

def qa_conjunctive(fact, lang, n): # answer = name (place AND date)
    return (f'Write {n} QUESTION paraphrases ({lang}) asking WHO had an event at {fact.place} '
            f'on {fact.date}. Do not name anyone.\n{_JSON}')

def cloze(fact, lang, n):          # answer = place (blank)
    return (f'Write {n} CLOZE sentences ({lang}): {fact.name} {fact.activity} at "____", '
            f'where the blank is the PLACE. Do not write the place.\n{_JSON}')


STYLE_FN = {s: globals()[s] for s in STATEMENT_STYLES + QUERY_STYLES}


# ---------- translation & anchor ----------
def translate_prompt(items, lang, glossary):
    g = "; ".join(f'{k} -> {v}' for k, v in glossary.items())
    lines = "\n".join(f"{i}. {t}" for i, t in enumerate(items))
    return (f'Translate each line into {lang}. Keep these fixed translations: {g}. '
            f'Keep any "____" blank and any dates parseable.\n{lines}\n'
            f'Return ONLY JSON: {{"items": ["...", ...]}} in the same order.')

def anchor_questions_prompt(n, lang, forbid, existing=()):
    avoid = (f'\nNon ripetere né parafrasare nessuna di queste già usate: {list(existing)}.'
             if existing and lang=="it" else
             f'\nDo NOT repeat or paraphrase any of these already-used: {list(existing)}.'
             if existing else '')
    if lang == "it":
        return (f'Genera {n} domande DISTINTE di cultura generale (in ITALIANO) con risposte '
                f'brevi e fattuali. Copri molti argomenti, evita ripetizioni. Ogni domanda autonoma. '
                f'Non menzionare mai: {forbid}.{avoid}\n'
                'Rispondi SOLO in JSON: {"items": ["domanda?", ...]}')
    return (f'Generate {n} DISTINCT general-knowledge questions (in ENGLISH) with short factual '
            f'answers. Span many topics, avoid clustering. Each self-contained. '
            f'Never mention: {forbid}.{avoid}\n'
            'Return ONLY JSON: {"items": ["question?", ...]}')

def anchor_stems_prompt(n, lang, forbid, existing=()):
    avoid = (f'\nNon ripetere nessuno di questi: {list(existing)}.'
             if existing and lang=="it" else
             f'\nDo NOT repeat any of these: {list(existing)}.' if existing else '')
    if lang == "it":
        return (f'Genera {n} brevi INIZI di frase DICHIARATIVA (in ITALIANO) troncati a metà. '
                f'Esempi: "La capitale del Giappone è", "L\'acqua bolle a". '
                f'Non menzionare mai: {forbid}.{avoid}\n'
                'Rispondi SOLO in JSON: {"items": ["inizio ...", ...]}')
    return (f'Generate {n} short DECLARATIVE sentence STEMS (in ENGLISH), cut off mid-sentence. '
            f'Examples: "The capital of Japan is", "Water boils at". '
            f'Never mention: {forbid}.{avoid}\n'
            'Return ONLY JSON: {"items": ["stem ...", ...]}')