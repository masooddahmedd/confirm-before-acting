# Draws the trade-off figure (wrong actions taken without asking vs share of requests confirmed) from the
# summaries written by evaluate.py.
#
#   python report.py --agents qwen2.5_7b-instruct llama3.1_8b --conditions phone phone_noise0
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LABELS = {
    "phone": "phone line",
    "phone_noise10": "phone + noise 10 dB",
    "phone_noise0": "phone + noise 0 dB",
    "phone_babble5": "phone + chatter 5 dB",
    "clean": "clean audio",
}
AGENT_NAMES = {"qwen2.5_7b-instruct": "Qwen2.5-7B", "llama3.1_8b": "Llama-3.1-8B"}
STYLE = {
    "utterance confidence": ("#9aa0a6", "-"),
    "lowest word confidence": ("#c5a96a", "-"),
    "argument-word confidence": ("#d08770", "-"),
    "n-best disagreement": ("#5e81ac", "--"),
    "disagreement, then argument confidence": ("#2e7d5b", "-"),
    "agent sample disagreement, then argument confidence": ("#8e5ea2", ":"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", nargs="+", default=["qwen2.5_7b-instruct", "llama3.1_8b"])
    ap.add_argument("--conditions", nargs="+", default=["phone", "phone_noise0"])
    ap.add_argument("--metric", default="unasked")
    ap.add_argument("--out", default="results/tradeoff.png")
    args = ap.parse_args()
    plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(len(args.agents), len(args.conditions), figsize=(3.4 * len(args.conditions), 2.5 * len(args.agents) + 0.9),
                             squeeze=False, sharex=True, sharey=True)
    for i, agent in enumerate(args.agents):
        s = json.loads(Path(f"results/summary_{agent}_{args.metric}_clean.json").read_text())
        for j, cond in enumerate(args.conditions):
            ax = axes[i][j]
            for name, pts in s["curves"][cond].items():
                if name not in STYLE:
                    continue
                colour, ls = STYLE[name]
                xs, ys = zip(*sorted(pts))
                ax.plot([100 * x for x in xs], [100 * y for y in ys], ls, color=colour,
                        lw=2 if name == "disagreement, then argument confidence" else 1.4, label=name)
            res = s["conditions"][cond]
            ed = res["either disagrees"]
            ax.plot(100 * ed["confirm"], 100 * ed["wrong"], "o", color="#2e7d5b", ms=7, label="confirm when any variant disagrees")
            ax.plot(100 * res["always"]["confirm"], 100 * res["always"]["wrong"], "s", color="k", ms=5, label="always confirm")
            ax.set_title(f"{AGENT_NAMES.get(agent, agent)} · {LABELS.get(cond, cond)}", fontsize=9)
            ax.grid(alpha=0.3)
            if i == len(args.agents) - 1:
                ax.set_xlabel("requests confirmed (%)")
            if j == 0:
                ax.set_ylabel("wrong, unasked (%)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    fig.savefig(args.out, dpi=150)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
