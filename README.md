# generalize_knowledge

Inject fictional facts into Mistral-7B via LoRA and test whether the fact is
**stored as a concept (generalizes)** or **memorized as a surface string**.
Every stage is `file -> file`, driven by one YAML. Notebooks stay thin: they
**import from `src/`, never define logic**.

## Layout
```
data/facts.json        source of truth: 5 quadruplets + aliases + designed collisions
experiments/*.yaml     one experiment = one file (mix + split + lora + hparams)
src/
  schema.py     Fact, load, collision validator
  verify.py     THE one date/name/place/set matcher (no LLM judge)
  llm.py        claude() generator + mistral() target + disk cache
  prompts.py    editable prompt per style  <- tune these by hand
  build.py      generate / translate / anchor / split / assemble
  train.py      LoRA (layers|all, sanity assert, masked-qa vs full-loss)
  evaluate.py   tiered eval (easy/hard) + collapse probe
  plot.py       per-layer sweep plot
notebooks/      01_facts  02_build  03_train_eval  (thin drivers)
```

## Run order
1. `01_facts` – author facts, check collisions (every place & date shared ≥2).
2. `02_build` – tune prompts, generate paraphrases, translate, build anchor (all cached).
3. `03_train_eval` – pick a YAML, train, eval easy + hard, collapse probe.

## The idea in one line
Hold out **paths, not facts**: train one style/lang, evaluate new paraphrases
(easy) and new styles/languages (hard). Memorization → cliff at held-out axes;
comprehension → graceful degradation.

## Setup
- Put keys/paths in `.env` (see `.env.example`). GPU is set at the top of each notebook.
- `pip install transformers peft torch matplotlib pyyaml requests`
