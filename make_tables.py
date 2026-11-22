# Writes results/tables.md from the evaluate.py summaries, so no number in the README is typed by hand.
#
#   python make_tables.py
import json
from pathlib import Path

import numpy as np

from evaluate import add_combo, crossfit, load_rows

AGENTS = {"qwen2.5_7b-instruct": "Qwen2.5-7B", "llama3.1_8b": "Llama-3.1-8B"}
CONDS = {"clean": "clean", "phone": "phone line", "phone_noise10": "phone + noise 10 dB", "phone_noise0": "phone + noise 0 dB",
         "phone_babble5": "phone + chatter 5 dB", "phone_babble0": "phone + chatter 0 dB"}
COMBO = "disagreement, then argument confidence"


def pct(x):
    return f"{100 * x:.1f}%"


def main():
    out = []
    out.append("### Wrong actions taken without asking (all 829 requests, thresholds cross-fitted)\n")
    out.append("| agent | condition | never confirm | confirm when any variant disagrees | utterance confidence, 20% budget | disagreement + argument confidence, 20% budget | difference [95% CI] | always confirm |")
    out.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for a, an in AGENTS.items():
        s = json.loads(Path(f"results/summary_{a}_unasked_clean.json").read_text())
        for c, cn in CONDS.items():
            if c not in s["conditions"]:
                continue
            r = s["conditions"][c]
            e, u, k, al = r["either disagrees"], r["utterance confidence"]["0.2"], r[COMBO]["0.2"], r["always"]
            out.append(
                f"| {an} | {cn} | {pct(r['never']['wrong'])} | **{pct(e['wrong'])}** ({pct(e['confirm'])} asked) | "
                f"{pct(u['wrong'])} ({pct(u['confirm'])}) | {pct(k['wrong'])} ({pct(k['confirm'])}) | "
                f"{100 * k['diff_vs_utterance']:+.1f} [{100 * k['diff_ci'][0]:+.1f}, {100 * k['diff_ci'][1]:+.1f}] | "
                f"0% ({pct(al['confirm'])} asked) |")
    out.append("")
    out.append("### Which wrong actions each policy catches (same number of confirmations for both policies)\n")
    out.append("| agent | condition | asked | mishearing-caused wrong actions | caught: any variant disagrees | caught: utterance confidence | agent-caused wrong actions | caught: any variant disagrees | caught: utterance confidence |")
    out.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for a, an in AGENTS.items():
        rows = load_rows(a, Path("data"), "clean")
        add_combo(rows)
        for c, cn in CONDS.items():
            R = [r for r in rows if r["cond"] == c]
            if c == "clean" or len(R) < 800:
                continue
            hear = [k for k, r in enumerate(R) if r["acts"] and not r["top_ok"] and r["oracle_ok"]]
            agt = [k for k, r in enumerate(R) if r["acts"] and not r["top_ok"] and not r["oracle_ok"]]
            either = [r["acts"] and (r["s_nbest_share"] > 0 or r["s_two_asr"] > 0) for r in R]
            budget = float(np.mean(either))
            utt = crossfit(R, "s_utt", budget)
            def got(dec, idx):
                return f"{sum(dec[k] for k in idx)}/{len(idx)}"
            out.append(f"| {an} | {cn} | {pct(budget)} | {len(hear)} | {got(either, hear)} | {got(utt, hear)} | {len(agt)} | {got(either, agt)} | {got(utt, agt)} |")
    out.append("")
    from evaluate import outcome

    out.append("### Cost: extra agent calls per request vs wrong actions taken without asking (phone + noise 10 dB)\n")
    out.append("| agent | variants checked | extra agent calls | asked | wrong without asking |")
    out.append("| --- | --- | --- | --- | --- |")
    for a, an in AGENTS.items():
        rows = [r for r in load_rows(a, Path("data"), "clean") if r["cond"] == "phone_noise10"]
        w0, _ = outcome(rows, [False] * len(rows))
        out.append(f"| {an} | none | 0 | 0% | {pct(w0.mean())} |")
        variants = [("second recogniser only", 1, lambda r: r["s_two_asr"] > 0)]
        for k in (1, 2, 4):
            variants.append((f"top-{k + 1} beam alternatives" if k > 1 else "2nd-best beam alternative", k,
                             (lambda kk: lambda r: any(r["nb_differs"][1:kk + 1]))(k)))
        variants.append(("all 5-best + second recogniser", 5, lambda r: any(r["nb_differs"]) or r["s_two_asr"] > 0))
        for name, n_extra, rule in variants:
            w, c = outcome(rows, [rule(r) for r in rows])
            out.append(f"| {an} | {name} | {n_extra} | {pct(c.mean())} | {pct(w.mean())} |")
    out.append("")
    out.append("### Baseline: re-sample the agent on the same transcript (self-consistency, 4 samples at temperature 0.7)\n")
    out.append("| agent | condition | 4 agent samples disagree | beam alternatives 2-5 disagree (also 4 calls) | 20% budget: variants, then argument confidence | 20% budget: samples, then argument confidence | samples minus variants [95% CI] | variants or samples disagree |")
    out.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for a, an in AGENTS.items():
        s = json.loads(Path(f"results/summary_{a}_unasked_clean.json").read_text())
        for c, cn in CONDS.items():
            r = s["conditions"].get(c, {})
            if "agent samples disagree (4 samples)" not in r:
                continue
            f, b, v, k, u = (r["agent samples disagree (4 samples)"], r["beam alternatives 2-5 disagree (4 calls)"],
                             r[COMBO]["0.2"], r["agent sample disagreement, then argument confidence"]["0.2"],
                             r["either disagrees or samples disagree"])
            out.append(f"| {an} | {cn} | {pct(f['wrong'])} ({pct(f['confirm'])} asked) | {pct(b['wrong'])} ({pct(b['confirm'])}) | "
                       f"{pct(v['wrong'])} ({pct(v['confirm'])}) | {pct(k['wrong'])} ({pct(k['confirm'])}) | "
                       f"{100 * k['diff_vs_variants']:+.1f} [{100 * k['diff_vs_variants_ci'][0]:+.1f}, {100 * k['diff_vs_variants_ci'][1]:+.1f}] | "
                       f"{pct(u['wrong'])} ({pct(u['confirm'])}) |")
    out.append("")
    out.append("### Accuracy of the agent's action\n")
    acc_conds = {"clean": "on what was said (clean transcript)", **{c: cn for c, cn in CONDS.items() if c != "clean"}}
    s0 = json.loads(Path(f"results/summary_{next(iter(AGENTS))}_unasked_clean.json").read_text())["conditions"]
    acc_conds = {c: cn for c, cn in acc_conds.items() if c in s0}
    out.append("| agent | " + " | ".join(acc_conds.values()) + " |")
    out.append("| --- |" + " --- |" * len(acc_conds))
    for a, an in AGENTS.items():
        s = json.loads(Path(f"results/summary_{a}_unasked_clean.json").read_text())["conditions"]
        out.append(f"| {an} | " + " | ".join(pct(s[c]["accuracy_heard"]) if c in s else "n/a" for c in acc_conds) + " |")
    Path("results/tables.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
