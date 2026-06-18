"""Eval = two tiers + a collapse probe.
  easy : new paraphrases, SAME style+lang as training
  hard : new styles (e.g. cloze, reverse) and new languages
collapse: general prompts must NOT emit a fact (your Exp-1 failure detector)."""
import torch
import verify


@torch.no_grad()
def ask(model, tok, q, max_new=64):
    ids = tok.apply_chat_template([{"role": "user", "content": q}],
                                  add_generation_prompt=True, return_tensors="pt").to(model.device)
    out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new,
                         do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True).strip()


def accuracy(model, tok, items):
    """items must be QUERY examples (have question/check_kind/expected)."""
    queries = [e for e in items if e["kind"] == "query"]
    if not queries: return None, []
    log = []
    for e in queries:
        a = ask(model, tok, e["question"])
        log.append({"ok": verify.check(a, e["expected"], e["check_kind"]),
                    "q": e["question"], "a": a, "style": e["style"], "lang": e["lang"]})
    acc = sum(r["ok"] for r in log) / len(log)
    return acc, log


def collapse_probe(model, tok, leak_terms, probes=None):
    """A fictional PLACE appearing in these answers = the adapter hijacked the model."""
    probes = probes or ["What is the capital of France?", "Ciao, come ti chiami?", "What is 17 + 25?"]
    terms = [t.lower() for t in leak_terms]
    bad = 0
    for q in probes:
        a = ask(model, tok, q)
        hit = any(t in a.lower() for t in terms)
        bad += hit
        print(f"  [{'COLLAPSE' if hit else 'ok'}] {q} -> {a[:80]}")
    print(f"[collapse] clean {len(probes)-bad}/{len(probes)}")
    return (len(probes) - bad) / len(probes)


def report(model, tok, eval_sets, leak_terms=()):
    """eval_sets = {tier_name: rows}. Scores each tier that has query rows."""
    out = {}
    for tier, rows in eval_sets.items():
        acc, _ = accuracy(model, tok, rows)
        out[tier] = acc
        print(f"[{tier}] acc={acc}")
    out["retention"] = collapse_probe(model, tok, leak_terms)
    return out
