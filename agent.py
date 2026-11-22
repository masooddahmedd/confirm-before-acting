# Runs an LLM tool-calling agent (through a local Ollama server) on every transcript the study needs:
# the original question text (perfect hearing), the top ASR transcript, each distinct 5-best hypothesis,
# and the second recogniser's transcript. Greedy decoding, fixed seed, results cached by input text.
#
#   python agent.py --model qwen2.5:7b-instruct --asr-tags mobiuslabsgmbh_faster-whisper-large-v3-turbo Systran_faster-whisper-large-v3
import argparse
import hashlib
import json
import time
from pathlib import Path

import requests

OLLAMA = "http://127.0.0.1:11434/api/chat"

SYSTEM = """You are a voice assistant that acts by calling functions. The user's words come from a speech recogniser.

Available functions (JSON schema):
{tools}

Reply with JSON only, in this exact shape:
{{"calls": [{{"name": "<function name>", "arguments": {{<argument>: <value>}}}}]}}
Use only the functions above and only arguments in their schemas, with values of the right type.
If none of the functions fits the request, reply {{"calls": []}}."""


def ask(model: str, tools: str, text: str, timeout: int = 120) -> tuple[list | None, str]:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM.format(tools=tools)},
            {"role": "user", "content": text},
        ],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0, "seed": 7, "num_ctx": 4096, "num_predict": 512},
    }
    r = requests.post(OLLAMA, json=body, timeout=timeout)
    r.raise_for_status()
    raw = r.json()["message"]["content"]
    try:
        obj = json.loads(raw)
        calls = obj.get("calls") if isinstance(obj, dict) else None
        if not isinstance(calls, list) or not all(isinstance(c, dict) for c in calls):
            return None, raw
        return calls, raw
    except json.JSONDecodeError:
        return None, raw


def key(model: str, item_id: str, text: str) -> str:
    return hashlib.sha256(f"{model}\n{item_id}\n{text}".encode()).hexdigest()[:20]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--model", default="qwen2.5:7b-instruct")
    ap.add_argument("--asr-tags", nargs="+", required=True)
    ap.add_argument("--conditions", nargs="+", default=["clean", "phone", "phone_noise10", "phone_noise0"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    items = {json.loads(l)["id"]: json.loads(l) for l in open(Path(args.data) / "items.jsonl", encoding="utf-8")}
    cache_path = Path("results/raw/agent") / f"{args.model.replace(':', '_').replace('/', '_')}.jsonl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = {}
    if cache_path.exists():
        for l in open(cache_path, encoding="utf-8"):
            r = json.loads(l)
            cache[r["key"]] = r

    # every (item, text) pair the analysis needs
    todo: list[tuple[str, str]] = []
    first_tag = args.asr_tags[0]
    for tag in args.asr_tags:
        for cond in args.conditions:
            p = Path("results/raw/asr") / f"{tag}__{cond}.jsonl"
            if not p.exists():
                print("missing", p)
                continue
            for l in open(p, encoding="utf-8"):
                r = json.loads(l)
                todo.append((r["id"], r["text"]))
                if tag == first_tag:
                    for h in r.get("nbest", []):
                        todo.append((r["id"], h["text"]))
    ids = sorted({i for i, _ in todo})
    for i in ids:
        todo.append((i, json.loads(items[i]["question"])[0]))  # perfect hearing
    seen, pairs = set(), []
    for i, t in todo:
        k = key(args.model, i, t)
        if k not in seen and k not in cache:
            seen.add(k)
            pairs.append((k, i, t))
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"{len(pairs)} agent calls to make ({len(cache)} cached)", flush=True)

    t0 = time.time()
    with open(cache_path, "a", encoding="utf-8") as f:
        for n, (k, i, t) in enumerate(pairs):
            for attempt in range(3):
                try:
                    calls, raw = ask(args.model, items[i]["tools"], t)
                    break
                except requests.RequestException as e:
                    print("retry", e, flush=True)
                    time.sleep(5)
            else:
                continue
            f.write(json.dumps({"key": k, "id": i, "text": t, "calls": calls, "raw": raw}, ensure_ascii=False) + "\n")
            if n % 200 == 0:
                rate = (n + 1) / max(time.time() - t0, 1e-9)
                print(f"{n}/{len(pairs)} {rate:.2f}/s eta {(len(pairs) - n) / rate / 60:.0f} min", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
