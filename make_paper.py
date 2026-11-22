# Writes paper/numbers.tex, paper/tables.tex and paper/fig_tradeoff.pdf from the result files, so no number
# in the paper is typed by hand. Run after evaluate.py (all agents, both metrics, both references).
#
#   python make_paper.py
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from evaluate import CONDITIONS, PRIMARY, SECOND, add_combo, crossfit, load_rows, outcome

AGENTS = {"qwen2.5_7b-instruct": "Qwen2.5-7B", "llama3.1_8b": "Llama-3.1-8B"}
CN = {"clean": "clean", "phone": "phone", "phone_noise10": "noise 10 dB", "phone_noise0": "noise 0 dB",
      "phone_babble5": "chatter 5 dB", "phone_babble0": "chatter 0 dB"}
QUIET = ["clean", "phone", "phone_noise10", "phone_noise0"]
CHATTER = ["phone_babble5", "phone_babble0"]
COMBO = "disagreement, then argument confidence"
SC_COMBO = "agent sample disagreement, then argument confidence"
OUT = Path("paper")


def p1(x):  # percent, one decimal
    return f"{100 * x:.1f}"


def pts(x):  # signed points, one decimal, LaTeX minus
    return f"{100 * x:+.1f}".replace("-", "$-$").replace("+", "$+$")


def ci(c):
    return f"[{pts(c[0])}, {pts(c[1])}]"


def lo_hi(vals, fmt=p1):
    return fmt(min(vals)), fmt(max(vals))


def words(t):
    return re.sub(r"[^\w\s]", " ", t.lower()).split()


def wer(ref, hyp):
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)], len(r)


def load_asr(tag, cond):
    return {json.loads(l)["id"]: json.loads(l) for l in open(f"results/raw/asr/{tag}__{cond}.jsonl", encoding="utf-8")}


def main():
    S = {a: json.loads(Path(f"results/summary_{a}_unasked_clean.json").read_text())["conditions"] for a in AGENTS}
    H = {a: json.loads(Path(f"results/summary_{a}_hearing_clean.json").read_text())["conditions"] for a in AGENTS}
    Q = {a: json.loads(Path(f"results/summary_{a}_unasked_question.json").read_text())["conditions"] for a in AGENTS}
    items = [json.loads(l) for l in open("data/items.jsonl", encoding="utf-8")]
    n = {}
    used = set(load_asr(PRIMARY, "clean"))
    n["NItemsAll"] = len(items)
    n["NItems"] = len(used)
    for sub in ("simple", "multiple", "irrelevance"):
        n[f"N{sub.capitalize()}"] = sum(1 for it in items if it["id"] in used and it["subset"] == sub)

    # recogniser: word error rate against the clean-audio transcript, and how often the variants differ
    clean = load_asr(PRIMARY, "clean")
    wers, ndist = {}, {}
    for c in CONDITIONS[1:]:
        a = load_asr(PRIMARY, c)
        e = sum(wer(clean[i]["text"], a[i]["text"])[0] for i in a)
        t = sum(wer(clean[i]["text"], a[i]["text"])[1] for i in a)
        wers[c] = e / t
    for c in CONDITIONS:
        a, b = load_asr(PRIMARY, c), load_asr(SECOND, c)
        ndist[c] = np.mean([len({" ".join(words(h["text"])) for h in a[i]["nbest"]} | {" ".join(words(b[i]["text"]))}) > 1
                            for i in a])
    macro = {"phone": "Phone", "phone_noise10": "NoiseTen", "phone_noise0": "NoiseZero",
             "phone_babble5": "ChatterFive", "phone_babble0": "ChatterZero"}
    for c in CONDITIONS[1:]:
        n[f"Wer{macro[c]}"] = p1(wers[c])
    n["TextsDifferMin"], n["TextsDifferMax"] = lo_hi([ndist[c] for c in QUIET])

    # headline numbers (quiet = the four conditions without chatter)
    q = S["qwen2.5_7b-instruct"]["phone_noise10"]
    n["HeadNever"], n["HeadEither"], n["HeadAsked"] = p1(q["never"]["wrong"]), p1(q["either disagrees"]["wrong"]), p1(q["either disagrees"]["confirm"])
    for a, tag in (("qwen2.5_7b-instruct", "Qwen"), ("llama3.1_8b", "Llama")):
        s = S[a]
        n[f"{tag}NeverMin"], n[f"{tag}NeverMax"] = lo_hi([s[c]["never"]["wrong"] for c in QUIET])
        n[f"{tag}EitherMin"], n[f"{tag}EitherMax"] = lo_hi([s[c]["either disagrees"]["wrong"] for c in QUIET])
        n[f"{tag}AskedMin"], n[f"{tag}AskedMax"] = lo_hi([s[c]["either disagrees"]["confirm"] for c in QUIET])
        n[f"{tag}AccSaid"] = p1(s["clean"]["accuracy_said"])
        n[f"{tag}MissedMin"], n[f"{tag}MissedMax"] = lo_hi([s[c]["n_missed_actions"] for c in QUIET], str)
        n[f"{tag}MissedChatter"] = str(s["phone_babble0"]["n_missed_actions"])
        n[f"{tag}BothMin"], n[f"{tag}BothMax"] = lo_hi([s[c]["either disagrees or samples disagree"]["wrong"] for c in QUIET])
        n[f"{tag}BothAskedMin"], n[f"{tag}BothAskedMax"] = lo_hi([s[c]["either disagrees or samples disagree"]["confirm"] for c in QUIET])
        n[f"{tag}SampAskedMin"], n[f"{tag}SampAskedMax"] = lo_hi([s[c]["agent samples disagree (4 samples)"]["confirm"] for c in QUIET])
        n[f"{tag}QuestionMisheard"] = p1(Q[a]["clean"]["n_hearing_caused"] / Q[a]["clean"]["n"])
    diffs = [S[a][c][COMBO]["0.2"]["diff_vs_utterance"] for a in AGENTS for c in QUIET]
    n["GainMin"], n["GainMax"] = f"{-100 * max(diffs):.1f}", f"{-100 * min(diffs):.1f}"
    sig = {b: {a: sum(S[a][c][COMBO][b]["diff_ci"][1] < 0 for c in QUIET) for a in AGENTS} for b in ("0.05", "0.1", "0.2")}
    n["SigFiveQwen"], n["SigFiveLlama"] = sig["0.05"]["qwen2.5_7b-instruct"], sig["0.05"]["llama3.1_8b"]
    n["SigTenQwen"], n["SigTenLlama"] = sig["0.1"]["qwen2.5_7b-instruct"], sig["0.1"]["llama3.1_8b"]
    n["SigTwentyQwen"], n["SigTwentyLlama"] = sig["0.2"]["qwen2.5_7b-instruct"], sig["0.2"]["llama3.1_8b"]
    conf = [S[a][c][k]["0.2"]["confirm"] for a in AGENTS for c in CONDITIONS for k in ("utterance confidence", COMBO)]
    n["RealisedMin"], n["RealisedMax"] = lo_hi(conf)
    n["AlwaysMin"], n["AlwaysMax"] = lo_hi([S[a][c]["always"]["confirm"] for a in AGENTS for c in CONDITIONS], lambda x: f"{100 * x:.0f}")
    n["ChatterAskedMin"], n["ChatterAskedMax"] = lo_hi([S[a][c]["either disagrees"]["confirm"] for a in AGENTS for c in CHATTER], lambda x: f"{100 * x:.0f}")
    n["AgentWrongMin"], n["AgentWrongMax"] = lo_hi([S[a][c]["never"]["wrong"] for a in AGENTS for c in QUIET], lambda x: f"{100 * x:.0f}")
    n["HearCausedMin"], n["HearCausedMax"] = lo_hi([S[a][c]["n_hearing_caused"] for a in AGENTS for c in QUIET[1:]], str)
    n["HearCausedChatMin"], n["HearCausedChatMax"] = lo_hi([S[a][c]["n_hearing_caused"] for a in AGENTS for c in CHATTER], str)
    scd = [S[a][c][SC_COMBO]["0.2"]["diff_vs_variants"] for a in AGENTS for c in QUIET]
    n["ScDiffMin"], n["ScDiffMax"] = f"{100 * min(scd):+.1f}".replace("-", "$-$").replace("+", "$+$"), f"{100 * max(scd):+.1f}".replace("-", "$-$").replace("+", "$+$")
    n["ScThirtyQwenVarBetter"] = sum(S["qwen2.5_7b-instruct"][c][SC_COMBO]["0.3"]["diff_vs_variants_ci"][0] > 0 for c in QUIET)
    n["ScThirtyLlamaSig"] = sum(S["llama3.1_8b"][c][SC_COMBO]["0.3"]["diff_vs_variants_ci"][0] > 0
                                or S["llama3.1_8b"][c][SC_COMBO]["0.3"]["diff_vs_variants_ci"][1] < 0 for c in QUIET)
    d = [-S["llama3.1_8b"][c][SC_COMBO]["0.3"]["diff_vs_variants"] for c in QUIET]
    n["ScThirtyLlamaAheadMin"], n["ScThirtyLlamaAheadMax"] = f"{100 * min(d):.1f}", f"{100 * max(d):.1f}"
    d = [S["qwen2.5_7b-instruct"][c][SC_COMBO]["0.3"]["diff_vs_variants"] for c in QUIET
         if S["qwen2.5_7b-instruct"][c][SC_COMBO]["0.3"]["diff_vs_variants_ci"][0] > 0]
    n["ScThirtyQwenAheadMin"], n["ScThirtyQwenAheadMax"] = f"{100 * min(d):.1f}", f"{100 * max(d):.1f}"
    hd = [H[a][c][COMBO]["0.2"]["diff_vs_utterance"] for a in AGENTS for c in CONDITIONS[1:]]
    n["HearDiffMin"], n["HearDiffMax"] = f"{-100 * max(hd):.1f}", f"{-100 * min(hd):.1f}"
    n["HearAllAhead"] = sum(x < 0 for x in hd)
    n["HearSig"] = sum(H[a][c][COMBO]["0.2"]["diff_ci"][1] < 0 for a in AGENTS for c in CONDITIONS[1:])
    n["HearN"] = len(hd)
    wl = [S[a][c][name][b] for a in AGENTS for c in CONDITIONS for name in ("lowest word confidence", "argument-word confidence")
          for b in S[a][c][name]]
    n["WordN"] = len(wl)
    n["WordBetter"] = sum(k["diff_ci"][1] < 0 for k in wl)
    n["WordWorse"] = sum(k["diff_ci"][0] > 0 for k in wl)
    n["WordMaxAbs"] = f"{100 * max(abs(k['diff_vs_utterance']) for k in wl):.1f}"
    n["WordBetterBy"] = f"{-100 * min(k['diff_vs_utterance'] for k in wl if k['diff_ci'][1] < 0):.1f}"
    n["SigThirtyAll"] = sum(S[a][c][COMBO]["0.3"]["diff_ci"][1] < 0 for a in AGENTS for c in QUIET)
    n["ScSigCount"] = sum(S[a][c][SC_COMBO]["0.2"]["diff_vs_variants_ci"][0] > 0 or S[a][c][SC_COMBO]["0.2"]["diff_vs_variants_ci"][1] < 0
                          for a in AGENTS for c in QUIET)

    # catch rates and the cost table need the per-request rows
    T = {}
    rows_all = {a: load_rows(a, Path("data"), "clean") for a in AGENTS}
    catch = []
    for a in AGENTS:
        add_combo(rows_all[a])
        for c in CONDITIONS[1:]:
            R = [r for r in rows_all[a] if r["cond"] == c]
            hear = [k for k, r in enumerate(R) if r["acts"] and not r["top_ok"] and r["oracle_ok"]]
            agt = [k for k, r in enumerate(R) if r["acts"] and not r["top_ok"] and not r["oracle_ok"]]
            either = [r["acts"] and (r["s_nbest_share"] > 0 or r["s_two_asr"] > 0) for r in R]
            utt = crossfit(R, "s_utt", float(np.mean(either)))
            catch.append((a, c, np.mean(either), len(hear), sum(either[k] for k in hear), sum(utt[k] for k in hear),
                          len(agt), sum(either[k] for k in agt), sum(utt[k] for k in agt)))
    qc = [x for x in catch if x[1] in ("phone", "phone_noise10")]
    n["CatchHearMin"], n["CatchHearMax"] = lo_hi([x[4] / x[3] for x in qc], lambda x: f"{100 * x:.0f}")
    n["CatchHearUttMin"], n["CatchHearUttMax"] = lo_hi([x[5] / x[3] for x in qc], lambda x: f"{100 * x:.0f}")
    n["CatchAgentMin"], n["CatchAgentMax"] = lo_hi([x[7] / x[6] for x in catch if x[1] in QUIET], lambda x: f"{100 * x:.0f}")
    n["CatchAgentUttMin"], n["CatchAgentUttMax"] = lo_hi([x[8] / x[6] for x in catch if x[1] in QUIET], lambda x: f"{100 * x:.0f}")

    lines = []
    lines.append(r"\begin{table*}[t]\centering\small")
    lines.append(r"\begin{tabular}{llrrrrr}\toprule")
    lines.append(r" & & & & \multicolumn{3}{c}{20\% confirmation budget} \\ \cmidrule(lr){5-7}")
    lines.append(r"Agent & Condition & Never & Any variant disagrees & Utterance conf. & Variants, then arg. conf. & Difference [95\% CI] \\")
    lines.append(r"\midrule")
    for a, an in AGENTS.items():
        for c in CONDITIONS:
            r = S[a][c]
            e, u, k = r["either disagrees"], r["utterance confidence"]["0.2"], r[COMBO]["0.2"]
            lines.append(f"{an} & {CN[c]} & {p1(r['never']['wrong'])} & \\textbf{{{p1(e['wrong'])}}} ({p1(e['confirm'])}) & "
                         f"{p1(u['wrong'])} & {p1(k['wrong'])} & {pts(k['diff_vs_utterance'])} {ci(k['diff_ci'])} \\\\")
        if a != list(AGENTS)[-1]:
            lines.append(r"\midrule")
    lines.append(r"\bottomrule\end{tabular}")
    lines.append(r"\caption{Wrong actions taken without asking (\% of " + str(n["NItems"]) + r" requests). \emph{Never}: the agent always acts. "
                 r"\emph{Any variant disagrees}: confirm when the call from any of the 5-best transcripts or the second recogniser differs from the call on the top transcript; "
                 r"its confirmation rate in brackets. The two ranked policies are held to a 20\% confirmation budget with cross-fitted thresholds (realised rates "
                 + n["RealisedMin"] + r"--" + n["RealisedMax"] + r"\%). Difference: ranked variant policy minus utterance confidence, in points, with a 95\% paired bootstrap interval.}\label{tab:main}")
    lines.append(r"\end{table*}")
    lines.append("")
    lines.append(r"\begin{table}[t]\centering\small\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{llrrrrr}\toprule")
    lines.append(r" & & & \multicolumn{2}{c}{Mishearing} & \multicolumn{2}{c}{Agent} \\ \cmidrule(lr){4-5}\cmidrule(lr){6-7}")
    lines.append(r"Agent & Cond. & Asked & Var. & Utt. & Var. & Utt. \\ \midrule")
    for a, an in AGENTS.items():
        for x in catch:
            if x[0] != a:
                continue
            lines.append(f"{an.split('-')[0]} & {CN[x[1]]} & {p1(x[2])} & {x[4]}/{x[3]} & {x[5]}/{x[3]} & {x[7]}/{x[6]} & {x[8]}/{x[6]} \\\\")
        if a != list(AGENTS)[-1]:
            lines.append(r"\midrule")
    lines.append(r"\bottomrule\end{tabular}")
    lines.append(r"\caption{Which wrong actions each signal catches at the same number of confirmations. Mishearing: the agent's call on the clean-audio transcript is right "
                 r"but the call on the degraded one is wrong. Agent: the call is wrong on the clean transcript too. Var.: any variant disagrees; Utt.: utterance confidence at the same rate.}\label{tab:catch}")
    lines.append(r"\end{table}")
    lines.append("")
    lines.append(r"\begin{table}[t]\centering\small\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{lrrrrr}\toprule")
    lines.append(r" & & \multicolumn{2}{c}{Qwen2.5-7B} & \multicolumn{2}{c}{Llama-3.1-8B} \\ \cmidrule(lr){3-4}\cmidrule(lr){5-6}")
    lines.append(r"Variants checked & Calls & Asked & Wrong & Asked & Wrong \\ \midrule")
    variants = [("none", 0, lambda r: False), ("second recogniser", 1, lambda r: r["s_two_asr"] > 0),
                ("2nd beam hypothesis", 1, lambda r: any(r["nb_differs"][1:2])),
                ("beam hypotheses 2--3", 2, lambda r: any(r["nb_differs"][1:3])),
                ("beam hypotheses 2--5", 4, lambda r: any(r["nb_differs"][1:5])),
                ("5-best + second recogniser", 5, lambda r: any(r["nb_differs"]) or r["s_two_asr"] > 0)]
    one = {}
    for a in AGENTS:
        R = [r for r in rows_all[a] if r["cond"] == "phone_noise10"]
        for name, k, rule in variants[1:3]:
            one[(a, name)] = outcome(R, [rule(r) for r in R])
    q1 = [one[("qwen2.5_7b-instruct", v[0])] for v in variants[1:3]]
    n["CostOneWrongMin"], n["CostOneWrongMax"] = lo_hi([w.mean() for w, c in q1])
    n["CostOneAskedMin"], n["CostOneAskedMax"] = lo_hi([c.mean() for w, c in q1])
    for name, k, rule in variants:
        cells = []
        for a in AGENTS:
            R = [r for r in rows_all[a] if r["cond"] == "phone_noise10"]
            w, c = outcome(R, [rule(r) for r in R])
            cells += [p1(c.mean()), p1(w.mean())]
        lines.append(f"{name} & {k} & " + " & ".join(cells) + r" \\")
    lines.append(r"\bottomrule\end{tabular}")
    lines.append(r"\caption{Cost: extra agent calls per request against wrong actions taken without asking (\%), noise 10 dB. The rule confirms when any checked variant changes the call.}\label{tab:cost}")
    lines.append(r"\end{table}")
    lines.append("")
    lines.append(r"\begin{table*}[t]\centering\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}\begin{tabular}{llrrrrrrr}\toprule")
    lines.append(r" & & \multicolumn{3}{c}{Yes/no rule: confirm when \dots\ disagree} & \multicolumn{3}{c}{20\% budget} & 30\% budget \\ "
                 r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-9}")
    lines.append(r"Agent & Cond. & samples & beam 2--5 & either & Var. & Samp. & Samp.$-$var. & Samp.$-$var. \\ \midrule")
    for a, an in AGENTS.items():
        for c in QUIET:
            r = S[a][c]
            f, b, u = (r["agent samples disagree (4 samples)"], r["beam alternatives 2-5 disagree (4 calls)"],
                       r["either disagrees or samples disagree"])
            v, k, k3 = r[COMBO]["0.2"], r[SC_COMBO]["0.2"], r[SC_COMBO]["0.3"]
            cells = [f"{p1(v['wrong'])} & {p1(k['wrong'])} & {pts(k['diff_vs_variants'])} {{\\scriptsize {ci(k['diff_vs_variants_ci'])}}}",
                     f"{pts(k3['diff_vs_variants'])} {{\\scriptsize {ci(k3['diff_vs_variants_ci'])}}}"]
            lines.append(f"{an.split('-')[0]} & {CN[c]} & {p1(f['wrong'])} {{\\scriptsize ({p1(f['confirm'])})}} & "
                         f"{p1(b['wrong'])} {{\\scriptsize ({p1(b['confirm'])})}} & {p1(u['wrong'])} {{\\scriptsize ({p1(u['confirm'])})}} & "
                         + " & ".join(cells) + " \\\\")
        if a != list(AGENTS)[-1]:
            lines.append(r"\midrule")
    lines.append(r"\bottomrule\end{tabular}")
    lines.append(r"\caption{Transcript variants against agent self-consistency (4 samples at temperature 0.7 on the top transcript), wrong actions taken without asking (\%), "
                 r"confirmation rate in brackets. Samples and beam hypotheses 2--5 cost the same 4 extra calls; \emph{either} confirms when any transcript variant or any sample disagrees. "
                 r"The budget columns rank by disagreement (Var.: transcript variants; Samp.: samples), break ties by argument-word confidence, and give the difference in points with a 95\% paired bootstrap interval (at 30\%, the difference only).}\label{tab:sc}")
    lines.append(r"\end{table*}")
    lines.append("")
    # appendix: accuracy, word error rate, and the hearing-only metric (written to appendix_tables.tex)
    split_at = len(lines)
    lines.append(r"\begin{table*}[t]\centering\small")
    lines.append(r"\begin{tabular}{l" + "r" * len(CONDITIONS) + r"}\toprule")
    lines.append(r" & " + " & ".join(CN[c] for c in CONDITIONS) + r" \\ \midrule")
    lines.append(r"WER vs.\ clean transcript & -- & " + " & ".join(p1(wers[c]) for c in CONDITIONS[1:]) + r" \\")
    lines.append(r"Any transcript variant differs & " + " & ".join(p1(ndist[c]) for c in CONDITIONS) + r" \\")
    for a, an in AGENTS.items():
        lines.append(f"{an} call accuracy & " + " & ".join(p1(S[a][c]["accuracy_heard"]) for c in CONDITIONS) + r" \\")
    for a, an in AGENTS.items():
        lines.append(f"{an} wrong actions / mishearing-caused & " + " & ".join(f"{S[a][c]['n_wrong_actions']} / {S[a][c]['n_hearing_caused']}" for c in CONDITIONS) + r" \\")
    lines.append(r"\bottomrule\end{tabular}")
    lines.append(r"\caption{Recogniser and agent per condition (\%). WER of Whisper large-v3-turbo against its own clean-audio transcript. Call accuracy is BFCL AST accuracy of the call on the top transcript.}\label{tab:acc}")
    lines.append(r"\end{table*}")
    lines.append("")
    lines.append(r"\begin{table*}[t]\centering\small")
    lines.append(r"\begin{tabular}{llrrrrr}\toprule")
    lines.append(r" & & & & \multicolumn{3}{c}{20\% confirmation budget} \\ \cmidrule(lr){5-7}")
    lines.append(r"Agent & Condition & Never & Any variant disagrees & Utterance conf. & Variants, then arg. conf. & Difference [95\% CI] \\ \midrule")
    for a, an in AGENTS.items():
        for c in CONDITIONS[1:]:
            r = H[a][c]
            e, u, k = r["either disagrees"], r["utterance confidence"]["0.2"], r[COMBO]["0.2"]
            lines.append(f"{an} & {CN[c]} & {p1(r['never']['wrong'])} & {p1(e['wrong'])} ({p1(e['confirm'])}) & {p1(u['wrong'])} & {p1(k['wrong'])} & "
                         f"{pts(k['diff_vs_utterance'])} {ci(k['diff_ci'])} \\\\")
        if a != list(AGENTS)[-1]:
            lines.append(r"\midrule")
    lines.append(r"\bottomrule\end{tabular}")
    lines.append(r"\caption{The hearing-only measure: a confirmed action is replaced by the agent's action on the clean-audio transcript, so only mishearing counts as preventable (\%).}\label{tab:hearing}")
    lines.append(r"\end{table*}")

    OUT.mkdir(exist_ok=True)
    for name, part in (("tables.tex", lines[:split_at]), ("appendix_tables.tex", lines[split_at:])):
        body = "\n".join(part) + "\n"
        # wide tables shrink to the column or page width instead of running off it (never enlarged)
        body = body.replace(r"\begin{tabular}", r"\begin{adjustbox}{max width=\linewidth}\begin{tabular}")
        body = body.replace(r"\end{tabular}", r"\end{tabular}\end{adjustbox}")
        (OUT / name).write_text(body, encoding="utf-8")
    num = ["% Written by make_paper.py. Do not edit."]
    for k, v in n.items():
        if not re.fullmatch(r"[A-Za-z]+", k):
            sys.exit(f"bad macro name {k}")
        num.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    (OUT / "numbers.tex").write_text("\n".join(num) + "\n", encoding="utf-8")
    subprocess.run([sys.executable, "report.py", "--out", str(OUT / "fig_tradeoff.pdf")], check=True)
    print("\n".join(num))


if __name__ == "__main__":
    main()
