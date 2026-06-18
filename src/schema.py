"""Facts are the single source of truth. Questions are generated FROM them,
never parsed back out. A fact is a typed 4-tuple + aliases for verification."""
import json
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class Fact:
    id: int
    name: str
    activity: str
    place: str
    attribute: str = ""
    details: list = field(default_factory=list)
    ref: str = ""
    name_aliases: list = field(default_factory=list)
    place_aliases: list = field(default_factory=list)

    @property
    def names(self):  return [self.name] + self.name_aliases
    @property
    def places(self): return [self.place] + self.place_aliases


def load_facts(path="data/facts.json"):
    raw = json.load(open(path))["facts"]
    return [Fact(**f) for f in raw]


def validate_grid(facts, min_share=2):
    """Latin-square check: neither name NOR activity alone may fix the place.
    Each must map to >= min_share distinct places."""
    from collections import defaultdict
    byname, byact = defaultdict(set), defaultdict(set)
    for f in facts:
        byname[f.name].add(f.place); byact[f.activity].add(f.place)
    ok = True
    for label, d in [("name", byname), ("activity", byact)]:
        for k, places in d.items():
            if len(places) < min_share:
                print(f"[grid WARN] {label} '{k}' -> {places}: single feature fixes the place")
                ok = False
    print(f"[grid] names->{{{', '.join(f'{k}:{len(v)}' for k,v in byname.items())}}}  "
          f"activities->{{{', '.join(f'{k}:{len(v)}' for k,v in byact.items())}}}  ok={ok}")
    return ok
