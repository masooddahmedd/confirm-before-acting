# Scores a predicted tool call against a BFCL reference, following BFCL's AST checker:
# same function name, every required argument present with one of the allowed values, an optional
# argument either absent (allowed list contains "") or set to an allowed value, and no unknown arguments.
# Strings are compared after BFCL's standardisation (lowercase, spaces and most punctuation removed).
import json
import re

_STD = re.compile(r"[ ,./\-_*^]")


def _std(s: str) -> str:
    return _STD.sub("", s).lower().replace("'", '"')


def _value_ok(pred, allowed: list) -> bool:
    for a in allowed:
        if a == "":
            continue
        if isinstance(a, bool) or isinstance(pred, bool):
            if pred is a or (isinstance(pred, str) and pred.lower() == str(a).lower()):
                return True
            continue
        if isinstance(a, (int, float)) and not isinstance(a, bool):
            try:
                if abs(float(pred) - float(a)) < 1e-9:
                    return True
            except (TypeError, ValueError):
                pass
            continue
        if isinstance(a, str):
            if isinstance(pred, str) and _std(pred) == _std(a):
                return True
            continue
        if isinstance(a, list):
            if isinstance(pred, list) and len(pred) == len(a) and all(_value_ok(p, [x]) for p, x in zip(pred, a)):
                return True
            continue
        if isinstance(a, dict):
            if isinstance(pred, dict) and set(pred) <= set(a) and all(
                _value_ok(pred[k], v if isinstance(v, list) else [v]) for k, v in pred.items()
            ) and all(k in pred or (isinstance(v, list) and "" in v) for k, v in a.items()):
                return True
            continue
        if pred == a:
            return True
    return False


def call_ok(pred: dict, ref: dict) -> bool:
    (fname, params), = ref.items()
    if pred.get("name") != fname:
        return False
    args = pred.get("arguments") or {}
    if not isinstance(args, dict):
        return False
    for k in args:
        if k not in params:
            return False
    for k, allowed in params.items():
        if k not in args:
            if "" not in allowed:
                return False
            continue
        if not _value_ok(args[k], allowed):
            return False
    return True


def correct(calls: list | None, reference: str) -> bool:
    """True when the predicted calls do what the reference asks. `calls` None means unparseable output."""
    ref = json.loads(reference)
    if calls is None:
        return False
    if not ref:  # irrelevance: the right action is no call at all
        return len(calls) == 0
    if len(calls) != len(ref):
        return False
    used = set()
    for r in ref:  # order-free matching, as in BFCL's parallel checker
        hit = next((i for i, c in enumerate(calls) if i not in used and call_ok(c, r)), None)
        if hit is None:
            return False
        used.add(hit)
    return True


def canon(calls: list | None) -> str:
    """A string that is equal for two call lists that would do the same thing (used for disagreement)."""
    if calls is None:
        return "<unparseable>"

    def norm(v):
        if isinstance(v, str):
            return _std(v)
        if isinstance(v, float) and v.is_integer():
            return int(v)
        if isinstance(v, list):
            return [norm(x) for x in v]
        if isinstance(v, dict):
            return {k: norm(x) for k, x in sorted(v.items())}
        return v

    items = sorted(json.dumps({"n": c.get("name"), "a": norm(c.get("arguments") or {})}, sort_keys=True) for c in calls)
    return "|".join(items)
