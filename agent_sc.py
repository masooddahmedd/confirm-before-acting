# Baseline: agent self-consistency. The agent answers the same top transcript k more times with sampling
# (temperature 0.7, different seeds); a policy confirms when any sample disagrees with the greedy call.
# This separates "the transcript variants carry information" from "any perturbation of the agent would do".
#
#   python agent_sc.py --model qwen2.5:7b-instruct --conditions clean phone_noise0 --k 4
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from agent import OLLAMA, SYSTEM

PRIMARY = "mobiuslabsgmbh_faster-whisper-large-v3-turbo"


def sample(model, tools, text, seed, temperature=0.7):
    body = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM.format(tools=tools)}, {"role": "user", "content": text}],
        "format": "json", "stream": False,
        "options": {"temperature": temperature, "seed": seed, "num_ctx": 4096, "num_predict": 512},
    }
    r = requests.post(OLLAMA, json=body, timeout=120)
    r.raise_for_status()
    raw = r.json()["message"]["content"]
    try:
        obj = json.loads(raw)
        calls = obj.get("calls") if isinstance(obj, dict) else None
        return calls if isinstance(calls, list) and all(isinstance(c, dict) for c in calls) else None
    except json.JSONDecodeError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--model", default="qwen2.5:7b-instruct")
    ap.add_argument("--conditions", nargs="+", default=["phone_noise10", "phone_noise0", "phone", "clean"])
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()
    items = {json.loads(l)["id"]: json.loads(l) for l in open(Path(args.data) / "items.jsonl", encoding="utf-8")}
    out = Path("results/raw/agent_sc") / f"{args.model.replace(':', '_')}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {(json.loads(l)["id"], json.loads(l)["cond"]) for l in open(out, encoding="utf-8")}
    t0, n = time.time(), 0
    with open(out, "a", encoding="utf-8") as f:
        for cond in args.conditions:
            for l in open(f"results/raw/asr/{PRIMARY}__{cond}.jsonl", encoding="utf-8"):
                a = json.loads(l)
                if (a["id"], cond) in done:
                    continue
                with ThreadPoolExecutor(args.k) as pool:
                    samples = list(pool.map(lambda s: sample(args.model, items[a["id"]]["tools"], a["text"], seed=100 + s),
                                            range(args.k)))
                f.write(json.dumps({"id": a["id"], "cond": cond, "text": a["text"], "samples": samples}) + "\n")
                n += 1
                if n % 100 == 0:
                    print(f"{cond} {n} {(time.time() - t0) / n:.2f}s/item", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
