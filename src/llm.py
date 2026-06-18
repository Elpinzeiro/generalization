"""Two distinct models, kept separate:
  - claude()  : the GENERATOR (writes paraphrases / anchors). Swap to your bypass if needed.
  - mistral() : the TARGET we inject into / self-distill the anchor from.
Every call is cached to disk: re-running never re-hits the API."""
import os, json, time, hashlib, requests
import bypass  

CACHE = "data/cache"; os.makedirs(CACHE, exist_ok=True)


def _key(*parts):
    return hashlib.sha1("||".join(map(str, parts)).encode()).hexdigest()[:16]

def cached(name, *parts):
    """Decorator-ish: cached(name, prompt)(fn) -> loads json or computes+saves."""
    path = f"{CACHE}/{name}_{_key(*parts)}.json"
    def run(fn):
        if os.path.exists(path):
            return json.load(open(path))
        val = fn()
        json.dump(val, open(path, "w"), ensure_ascii=False, indent=2)
        return val
    return run


# ---------- generator: Claude (via proxy bypass) ----------
def claude(prompt, model, max_tokens=3000):
    cfg = bypass.get_api_config("anthropic")
    # inject our real prompt into the bypass request builder (it hardcodes "hi")
    bypass.build_anthropic_request = lambda mn, c: (
        c["base_url"] + c["message_path"],
        {**c["headers"], "x-api-key": os.environ[c["key_env"]]},
        {"model": mn, "max_tokens": max_tokens,
         "messages": [{"role": "user", "content": prompt}]})
    r = bypass.perform_request_with_optional_bypass("anthropic", model, cfg)
    r.raise_for_status()
    return r.json()["content"][0]["text"]

def claude_json(prompt, model, max_tokens=3000, retries=2):
    last = None
    for _ in range(retries + 1):
        try:
            t = claude(prompt, model, max_tokens).replace("```json", "").replace("```", "").strip()
            return json.loads(t)
        except Exception as e:
            last = e; time.sleep(0.4)
    raise last


# ---------- target: local Mistral (lazy-loaded once) ----------
_TGT = {}
def load_mistral(path, dtype="bfloat16"):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    snap = os.path.join(path, "snapshots")
    mpath = os.path.join(snap, os.listdir(snap)[0]) if os.path.isdir(snap) else path
    tok = AutoTokenizer.from_pretrained(mpath)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        mpath, device_map="auto", torch_dtype=getattr(torch, dtype))
    _TGT["tok"], _TGT["model"] = tok, model
    return tok, model

def mistral(prompt, max_new_tokens=80, chat=True):
    import torch
    tok, model = _TGT["tok"], _TGT["model"]
    if chat:                                   # q/a: instruct the model to ANSWER
        ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                      add_generation_prompt=True, return_tensors="pt")
    else:                                       # declarative: raw CONTINUATION
        ids = tok(prompt, return_tensors="pt").input_ids
    ids = ids.to(model.device)
    out = model.generate(ids, attention_mask=torch.ones_like(ids),
                         max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True).strip()