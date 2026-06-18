"""LoRA training on the assembled mix. Statements/anchor -> full-loss NTP.
Queries -> SFT with loss on the answer only. Layer set comes from config;
a sanity assert proves the adapter sits exactly where requested."""
import re, random, torch
from transformers import get_linear_schedule_with_warmup
from peft import LoraConfig, TaskType, get_peft_model

_EN_M = ["January","February","March","April","May","June","July","August",
         "September","October","November","December"]

def human_date(iso):
    y, m, d = map(int, iso.split("-")); return f"{_EN_M[m-1]} {d}, {y}"

def answer_text(e):
    """Canonical training answer for a query example, derived from its gold."""
    if e["check_kind"] == "date": return human_date(e["expected"])
    if e["check_kind"] == "set":  return ", ".join(g[0] for g in e["expected"])
    return str(e["expected"])


def build_examples(tok, mix):
    ex = []
    for e in mix:
        if e["kind"] == "query":                                   # SFT: loss on answer
            p = tok.apply_chat_template([{"role": "user", "content": e["question"]}],
                                        add_generation_prompt=True)
            a = tok(answer_text(e), add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
            ex.append((p + a, [-100] * len(p) + a))
        else:                                                      # NTP: loss on all
            ids = tok(e["text"], add_special_tokens=True)["input_ids"] + [tok.eos_token_id]
            ex.append((ids, list(ids)))
    return ex


def train_lora(model, tok, ex, cfg):
    L = cfg["lora"]
    lc = LoraConfig(r=L["r"], lora_alpha=L["alpha"], lora_dropout=L["dropout"],
                    target_modules=L["targets"], bias="none", task_type=TaskType.CAUSAL_LM,
                    layers_to_transform=(None if L["layers"] == "all" else list(L["layers"])))
    model = get_peft_model(model, lc)
    model.print_trainable_parameters()

    active = sorted({int(g.group(1)) for n, _ in model.named_modules()
                     if "lora_A" in n and (g := re.search(r"\.layers\.(\d+)\.", n))})
    if L["layers"] != "all":
        assert active == sorted(L["layers"]), f"layer mismatch {active} != {L['layers']}"
    print(f"[sanity] LoRA active on layers: {'ALL' if L['layers']=='all' else active}")

    T, bs = cfg["train"], cfg["train"]["batch_size"]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(T["lr"]))
    steps = ((len(ex) + bs - 1) // bs) * T["epochs"]
    sched = get_linear_schedule_with_warmup(opt, int(0.03 * steps), steps)
    for epoch in range(T["epochs"]):
        model.train(); random.shuffle(ex); tot = nb = 0
        for i in range(0, len(ex), bs):
            b = ex[i:i+bs]; mx = max(len(x[0]) for x in b)
            ids = torch.tensor([x[0] + [tok.pad_token_id]*(mx-len(x[0])) for x in b])
            lbl = torch.tensor([x[1] + [-100]*(mx-len(x[1])) for x in b])
            att = torch.tensor([[1]*len(x[0]) + [0]*(mx-len(x[0])) for x in b])
            out = model(input_ids=ids.cuda(), attention_mask=att.cuda(), labels=lbl.cuda())
            out.loss.backward(); opt.step(); sched.step(); opt.zero_grad()
            tot += out.loss.item(); nb += 1
        print(f"  epoch {epoch+1}/{T['epochs']}  loss={tot/nb:.4f}")
    model.eval()
    return model
