# Compares confirmation policies: for each request the agent either acts on what the recogniser heard, or
# first reads the action back ("Booking 4412, is that right?"). A confirmed wrong action is corrected by
# the user and replaced by what the agent does with the true request; a confirmed right action costs one
# unnecessary turn. Thresholds are chosen on a dev half of the requests and scored on the other half.
#
#   python evaluate.py --agent qwen2.5_7b-instruct
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from score import canon, correct

PRIMARY = "mobiuslabsgmbh_faster-whisper-large-v3-turbo"
SECOND = "Systran_faster-whisper-large-v3"
CONDITIONS = ["clean", "phone", "phone_noise10", "phone_noise0", "phone_babble5", "phone_babble0"]
BUDGETS = [0.05, 0.10, 0.20, 0.30]
_WORD = re.compile(r"[^\w.]+")


def agent_key(model, item_id, text):
    return hashlib.sha256(f"{model}\n{item_id}\n{text}".encode()).hexdigest()[:20]


def split_of(item_id: str) -> str:
    return "dev" if int(hashlib.sha256(item_id.encode()).hexdigest()[:8], 16) % 2 == 0 else "test"


def flat_values(v) -> list[str]:
    if isinstance(v, dict):
        return [s for x in v.values() for s in flat_values(x)]
    if isinstance(v, list):
        return [s for x in v for s in flat_values(x)]
    return [str(v)]


def norm_word(w: str) -> str:
    w = _WORD.sub("", w.lower()).strip(".")
    try:
        f = float(w.replace(",", ""))
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        return w


def arg_confidence(calls, words) -> float:
    """Lowest recogniser probability among transcript words that ended up in the call's arguments."""
    if not words:
        return 0.0
    probs = {}
    for w in words:
        k = norm_word(w["w"])
        if k:
            probs[k] = min(probs.get(k, 1.0), w["p"])
    hits = []
    for c in calls or []:
        for v in flat_values(c.get("arguments") or {}):
            for tok in re.split(r"\s+", str(v)):
                k = norm_word(tok)
                if k in probs:
                    hits.append(probs[k])
    return min(hits) if hits else min(w["p"] for w in words)


def load_rows(agent: str, data: Path, truth_mode: str = "clean"):
    """truth_mode "clean": what the user said = the clean-audio transcript (the benchmark audio is a spoken
    paraphrase, so this is the faithful reference). "question": the benchmark's original question text."""
    items = {json.loads(l)["id"]: json.loads(l) for l in open(data / "items.jsonl", encoding="utf-8")}
    clean_said = {json.loads(l)["id"]: json.loads(l)["text"]
                  for l in open(f"results/raw/asr/{PRIMARY}__clean.jsonl", encoding="utf-8")}
    cache = {}
    for l in open(Path("results/raw/agent") / f"{agent}.jsonl", encoding="utf-8"):
        r = json.loads(l)
        cache[r["key"]] = r["calls"]
    model_name = agent.replace("_", ":", 1) if ":" not in agent else agent
    sc = {}  # agent self-consistency samples on the top transcript (agent_sc.py), if run
    sc_path = Path("results/raw/agent_sc") / f"{agent}.jsonl"
    if sc_path.exists():
        for l in open(sc_path, encoding="utf-8"):
            r = json.loads(l)
            sc[(r["id"], r["cond"])] = r["samples"]

    def act(item_id, text):
        k = agent_key(model_name, item_id, text)
        return cache.get(k, "MISSING")

    rows = []
    for cond in CONDITIONS:
        prim_path = Path(f"results/raw/asr/{PRIMARY}__{cond}.jsonl")
        if not prim_path.exists():
            continue
        prim = {json.loads(l)["id"]: json.loads(l) for l in open(prim_path, encoding="utf-8")}
        sec_path = Path(f"results/raw/asr/{SECOND}__{cond}.jsonl")
        sec = {json.loads(l)["id"]: json.loads(l) for l in open(sec_path, encoding="utf-8")} if sec_path.exists() else {}
        for i, a in prim.items():
            it = items[i]
            truth = clean_said[i] if truth_mode == "clean" else json.loads(it["question"])[0]
            top = act(i, a["text"])
            oracle = act(i, truth)
            nb = [act(i, h["text"]) for h in a.get("nbest", [])]
            alt = act(i, sec[i]["text"]) if i in sec else "MISSING"
            if "MISSING" in [top, oracle] or any(x == "MISSING" for x in nb):
                continue
            words = a["words"]
            samples = sc.get((i, cond))
            rows.append({
                # share of sampled agent answers (same transcript, temperature 0.7) that differ from the greedy one
                "s_sc_share": (1 - sum(canon(c) == canon(top) for c in samples) / len(samples)) if samples else None,
                "id": i, "cond": cond, "subset": it["subset"], "split": split_of(i),
                "acts": bool(top),  # the agent wants to do something
                "top_ok": correct(top, it["reference"]),
                "oracle_ok": correct(oracle, it["reference"]),
                "oracle_acts": bool(oracle),
                # signals a policy can use (higher = more reason to confirm)
                "s_utt": -(a["avg_logprob"] if a["avg_logprob"] is not None else -10.0),
                "s_minword": 1 - min((w["p"] for w in words), default=0.0),
                "s_arg": 1 - arg_confidence(top, words),
                "s_nbest": len({canon(c) for c in [top] + nb}) - 1,
                "s_nbest_share": 1 - sum(canon(c) == canon(top) for c in nb) / max(len(nb), 1),
                "s_two_asr": float(alt != "MISSING" and canon(alt) != canon(top)),
                "nb_differs": [canon(c) != canon(top) for c in nb],  # per n-best rank, for the cost analysis
            })
    return rows




def outcome(rows, confirm, metric="unasked"):
    """Per-request results given a boolean confirm decision per row.

    metric "unasked": a wrong action is harmful when it is taken without asking. Reading the action back
        stops it whatever caused it (the user says no), so a confirmed wrong action counts as prevented.
    metric "hearing": only mishearing counts. A confirmed action is replaced by what the agent does with
        what the user really said, so agent mistakes on correctly heard requests stay wrong.
    A request where the agent takes no action has nothing to confirm; a missed action is reported apart.
    """
    wrong = np.zeros(len(rows))
    conf = np.zeros(len(rows))
    for k, (r, c) in enumerate(zip(rows, confirm)):
        if not r["acts"]:
            continue
        if c:
            conf[k] = 1
            if metric == "hearing":
                wrong[k] = 0 if r["top_ok"] else float(r["oracle_acts"] and not r["oracle_ok"])
        else:
            wrong[k] = float(not r["top_ok"])
    return wrong, conf


SIGNALS = {  # ranked policies, run at fixed confirmation budgets
    "utterance confidence": "s_utt",
    "lowest word confidence": "s_minword",
    "argument-word confidence": "s_arg",
    "n-best disagreement": "s_nbest_share",
    "disagreement, then argument confidence": "s_combo",
}
FIXED = {  # yes/no policies, reported at their own confirmation rate
    "n-best disagrees (any)": lambda r: r["s_nbest_share"] > 0,
    "second recogniser disagrees": lambda r: r["s_two_asr"] > 0,
    "either disagrees": lambda r: r["s_nbest_share"] > 0 or r["s_two_asr"] > 0,
}
BASELINE = "utterance confidence"
# Self-consistency baseline: same number of extra agent calls (4) as beam alternatives 2-5, but the agent is
# re-sampled on the one transcript instead of being shown other transcripts. Only scored where agent_sc.py ran.
SC_SIGNALS = {"agent sample disagreement, then argument confidence": "s_sc_combo"}
SC_FIXED = {
    "agent samples disagree (4 samples)": lambda r: r["s_sc_share"] > 0,
    "beam alternatives 2-5 disagree (4 calls)": lambda r: any(r["nb_differs"][1:5]),
    "either disagrees or samples disagree": lambda r: r["s_nbest_share"] > 0 or r["s_two_asr"] > 0 or r["s_sc_share"] > 0,
}
SC_REFERENCE = "disagreement, then argument confidence"  # the matched-budget comparison for the baseline


def add_combo(rows):
    for r in rows:
        disagree = r["s_nbest_share"] > 0 or r["s_two_asr"] > 0
        r["s_combo"] = 2.0 * disagree + r["s_arg"]  # every disagreement first, then lowest argument confidence
        if r["s_sc_share"] is not None:
            r["s_sc_combo"] = 2.0 * (r["s_sc_share"] > 0) + r["s_arg"]


def curve(rows, sig, metric, n_points=60):
    """Wrong-action rate against confirmation rate as the threshold sweeps (descriptive, all requests)."""
    vals = np.array([r[sig] for r in rows])
    w, c = outcome(rows, [False] * len(rows), metric)
    pts = [(0.0, float(w.mean()))]
    for t in np.unique(np.quantile(vals, np.linspace(0, 1, n_points)))[::-1]:
        w, c = outcome(rows, [v >= t for v in vals], metric)
        pts.append((float(c.mean()), float(w.mean())))
    return pts


def threshold_for_budget(rows, sig, budget):
    """Largest-score-first: the cut that confirms at most `budget` of the given requests."""
    vals = np.array([r[sig] for r in rows if r["acts"]])
    n_all = len(rows)
    if len(vals) == 0:
        return np.inf
    for t in np.unique(vals)[::-1]:
        if (vals >= t).sum() / n_all > budget:
            return t + 1e-12
    return np.unique(vals)[0]


def crossfit(R, sig, budget):
    """Threshold chosen on one half, applied to the other, both ways; one decision per request."""
    dec = {}
    for fit, apply in (("dev", "test"), ("test", "dev")):
        t = threshold_for_budget([r for r in R if r["split"] == fit], sig, budget)
        for k, r in enumerate(R):
            if r["split"] == apply:
                dec[k] = r[sig] >= t
    return [dec[k] for k in range(len(R))]


def boot_ci(x, n=2000, seed=0):
    """95% bootstrap interval of a per-request mean; pass a paired difference to compare two policies."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def evaluate_condition(R, metric):
    res = {"n": len(R)}
    res["accuracy_heard"] = float(np.mean([r["top_ok"] for r in R]))
    res["accuracy_said"] = float(np.mean([r["oracle_ok"] for r in R]))
    res["n_wrong_actions"] = int(sum(r["acts"] and not r["top_ok"] for r in R))
    res["n_hearing_caused"] = int(sum(r["acts"] and not r["top_ok"] and r["oracle_ok"] for r in R))
    res["n_missed_actions"] = int(sum((not r["acts"]) and not r["top_ok"] for r in R))
    w, _ = outcome(R, [False] * len(R), metric)
    res["never"] = {"wrong": float(w.mean()), "ci": boot_ci(w), "confirm": 0.0}
    w, c = outcome(R, [True] * len(R), metric)
    res["always"] = {"wrong": float(w.mean()), "ci": boot_ci(w), "confirm": float(c.mean())}
    target = (lambda r: not r["top_ok"]) if metric == "unasked" else (lambda r: (not r["top_ok"]) and r["oracle_ok"])
    w, c = outcome(R, [target(r) for r in R], metric)
    res["oracle"] = {"wrong": float(w.mean()), "confirm": float(c.mean())}
    for name, rule in FIXED.items():
        w, c = outcome(R, [rule(r) for r in R], metric)
        res[name] = {"wrong": float(w.mean()), "ci": boot_ci(w), "confirm": float(c.mean())}
    base_w = {}
    for name, sig in SIGNALS.items():
        res[name] = {}
        for b in BUDGETS:
            w, c = outcome(R, crossfit(R, sig, b), metric)
            cell = {"wrong": float(w.mean()), "ci": boot_ci(w), "confirm": float(c.mean())}
            if name == BASELINE:
                base_w[b] = w
            else:
                cell["diff_vs_utterance"] = float((w - base_w[b]).mean())
                cell["diff_ci"] = boot_ci(w - base_w[b])
            res[name][str(b)] = cell
    if all(r["s_sc_share"] is not None for r in R):
        ref_w = {}
        for b in BUDGETS:
            ref_w[b], _ = outcome(R, crossfit(R, SIGNALS[SC_REFERENCE], b), metric)
        for name, rule in SC_FIXED.items():
            w, c = outcome(R, [rule(r) for r in R], metric)
            res[name] = {"wrong": float(w.mean()), "ci": boot_ci(w), "confirm": float(c.mean())}
        for name, sig in SC_SIGNALS.items():
            res[name] = {}
            for b in BUDGETS:
                w, c = outcome(R, crossfit(R, sig, b), metric)
                res[name][str(b)] = {"wrong": float(w.mean()), "ci": boot_ci(w), "confirm": float(c.mean()),
                                     "diff_vs_utterance": float((w - base_w[b]).mean()),
                                     "diff_ci": boot_ci(w - base_w[b]),
                                     "diff_vs_variants": float((w - ref_w[b]).mean()),
                                     "diff_vs_variants_ci": boot_ci(w - ref_w[b])}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="qwen2.5_7b-instruct")
    ap.add_argument("--data", default="data")
    ap.add_argument("--truth", choices=["clean", "question"], default="clean")
    ap.add_argument("--metric", choices=["unasked", "hearing"], default="unasked")
    args = ap.parse_args()
    rows = load_rows(args.agent, Path(args.data), args.truth)
    add_combo(rows)
    summary = {"agent": args.agent, "truth": args.truth, "metric": args.metric, "conditions": {}, "curves": {}}
    for cond in CONDITIONS:
        R = [r for r in rows if r["cond"] == cond]
        if len(R) < 800:  # condition not finished (all 829 requests needed)
            continue
        summary["conditions"][cond] = evaluate_condition(R, args.metric)
        summary["curves"][cond] = {name: curve(R, sig, args.metric) for name, sig in SIGNALS.items()}
        if all(r["s_sc_share"] is not None for r in R):
            summary["curves"][cond].update({name: curve(R, sig, args.metric) for name, sig in SC_SIGNALS.items()})
    out = Path("results") / f"summary_{args.agent}_{args.metric}_{args.truth}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"# agent {args.agent}  metric {args.metric}  reference {args.truth}")
    for cond, res in summary["conditions"].items():
        print(f"\n== {cond}  n={res['n']}  acc heard {res['accuracy_heard']:.3f}  said {res['accuracy_said']:.3f}"
              f"  wrong actions {res['n_wrong_actions']} (hearing-caused {res['n_hearing_caused']})  missed {res['n_missed_actions']}")
        for k in ("never", "always", "oracle"):
            print(f"   {k:38s} wrong {res[k]['wrong']:.3f}  confirm {res[k]['confirm']:.3f}")
        for name in list(FIXED) + [n for n in SC_FIXED if n in res]:
            print(f"   {name:38s} wrong {res[name]['wrong']:.3f}  confirm {res[name]['confirm']:.3f}")
        for name in [n for n in SC_SIGNALS if n in res]:
            print(f"   {name:38s} " + "  ".join(
                f"b{b}: w {res[name][str(b)]['wrong']:.3f} c {res[name][str(b)]['confirm']:.3f}"
                f" vs variants {res[name][str(b)]['diff_vs_variants']:+.3f} [{res[name][str(b)]['diff_vs_variants_ci'][0]:+.3f},"
                f"{res[name][str(b)]['diff_vs_variants_ci'][1]:+.3f}]" for b in BUDGETS))
        for name in SIGNALS:
            cells = "  ".join(
                f"b{b}: w {res[name][str(b)]['wrong']:.3f} c {res[name][str(b)]['confirm']:.3f}"
                + (f" d {res[name][str(b)]['diff_vs_utterance']:+.3f} [{res[name][str(b)]['diff_ci'][0]:+.3f},{res[name][str(b)]['diff_ci'][1]:+.3f}]"
                   if "diff_vs_utterance" in res[name][str(b)] else "")
                for b in BUDGETS)
            print(f"   {name:38s} {cells}")


if __name__ == "__main__":
    main()
