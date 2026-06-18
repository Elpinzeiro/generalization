"""THE single verifier. Every answer is atomic and Python-checkable:
date | name | place | set | int | yesno. Add a language by extending MONTHS."""
import re

MONTHS = {
    # en
    "january":1,"february":2,"march":3,"april":4,"may":5,"june":6,"july":7,
    "august":8,"september":9,"october":10,"november":11,"december":12,
    # it
    "gennaio":1,"febbraio":2,"marzo":3,"aprile":4,"maggio":5,"giugno":6,"luglio":7,
    "agosto":8,"settembre":9,"ottobre":10,"novembre":11,"dicembre":12,
    # add fi/ja/ar/hi month words here as you add eval languages
}
_ALT = "|".join(MONTHS)
_MDY = re.compile(rf"\b({_ALT})\s+(\d{{1,2}}),?\s+(\d{{4}})\b", re.I)   # Month DD, YYYY
_DMY = re.compile(rf"\b(\d{{1,2}})\s+({_ALT}),?\s+(\d{{4}})\b", re.I)   # DD Month YYYY
_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")                       # YYYY-MM-DD
_NUM = re.compile(r"\b(\d{1,2})[/.](\d{1,2})[/.](\d{4})\b")            # DD/MM/YYYY


def date_set(text):
    """All dates found in text -> set of (y, m, d). Multi-language."""
    t, out = str(text), set()
    for x in _MDY.finditer(t): out.add((int(x.group(3)), MONTHS[x.group(1).lower()], int(x.group(2))))
    for x in _DMY.finditer(t): out.add((int(x.group(3)), MONTHS[x.group(2).lower()], int(x.group(1))))
    for x in _ISO.finditer(t): out.add((int(x.group(1)), int(x.group(2)), int(x.group(3))))
    for x in _NUM.finditer(t): out.add((int(x.group(3)), int(x.group(2)), int(x.group(1))))  # day/month
    return out

def _iso(d):  # "2025-06-14" -> {(2025,6,14)}
    y, m, dd = map(int, d.split("-")); return {(y, m, dd)}

def _norm(s): return re.sub(r"\s+", " ", str(s).lower()).strip()


def check(answer, expected, kind):
    """answer = model text. expected/kind come from the eval item.
       kind: 'date' (expected=ISO), 'name'/'place' (expected=list of aliases),
       'set' (expected=list of alias-lists), 'int', 'yesno' (expected='yes'/'no')."""
    a = _norm(answer)
    if kind == "date":
        return date_set(answer) == _iso(expected)
    if kind in ("name", "place"):
        return any(_norm(al) in a for al in expected)
    if kind == "set":                       # expected = [[aliases A], [aliases B], ...]
        return all(any(_norm(al) in a for al in group) for group in expected)
    if kind == "int":
        nums = re.findall(r"\b(\d+)\b", a)
        return bool(nums) and int(nums[0]) == int(expected)
    if kind == "yesno":
        yes = any(w in a for w in ["yes", "sì", "si "])
        no  = any(w in a for w in ["no", "non "])
        return ("yes" if yes and not no else "no" if no else "") == expected
    raise ValueError(f"unknown kind {kind}")
