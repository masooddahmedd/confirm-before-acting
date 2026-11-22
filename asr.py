# Transcribes every item in every listening condition with faster-whisper and keeps what a confirmation
# policy could use: the top transcript, a probability per word, the utterance log-probability, and the
# 5-best list from beam search.
#
#   python asr.py --data data --model large-v3-turbo --conditions clean phone phone_noise10 phone_noise0
import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from faster_whisper.tokenizer import Tokenizer

MAX_SEC = 30.0  # one Whisper window; longer clips (11 of 840) are left out of the study


def nbest(model: WhisperModel, audio: np.ndarray, n: int = 5) -> list[dict]:
    feats = model.feature_extractor(audio)
    frames = 3000
    if feats.shape[-1] < frames:
        feats = np.pad(feats, ((0, 0), (0, frames - feats.shape[-1])))
    feats = feats[:, :frames]
    enc = model.encode(feats)
    tok = Tokenizer(model.hf_tokenizer, model.model.is_multilingual, task="transcribe", language="en")
    prompt = model.get_prompt(tok, previous_tokens=[], without_timestamps=True)
    res = model.model.generate(
        enc, [prompt], beam_size=n, num_hypotheses=n, return_scores=True, max_length=448,
        suppress_blank=True, suppress_tokens=[-1],
    )[0]
    out = []
    for ids, score in zip(res.sequences_ids, res.scores):
        out.append({"text": tok.decode(ids).strip(), "score": float(score)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--conditions", nargs="+", default=["clean", "phone", "phone_noise10", "phone_noise0"])
    ap.add_argument("--nbest", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    data = Path(args.data)
    items = [json.loads(l) for l in open(data / "items.jsonl", encoding="utf-8")]
    items = [i for i in items if i["duration"] <= MAX_SEC]
    if args.limit:
        items = items[: args.limit]
    model = WhisperModel(args.model, device="cuda", compute_type="float16")
    out_dir = Path("results/raw/asr")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.model.replace("/", "_")
    for cond in args.conditions:
        path = out_dir / f"{tag}__{cond}.jsonl"
        done = set()
        if path.exists():
            done = {json.loads(l)["id"] for l in open(path, encoding="utf-8")}
        t0 = time.time()
        with open(path, "a", encoding="utf-8") as f:
            for k, it in enumerate(items):
                if it["id"] in done:
                    continue
                audio, sr = sf.read(data / cond / f"{it['id']}.wav", dtype="float32")
                segs, _ = model.transcribe(
                    audio, language="en", beam_size=5, word_timestamps=True,
                    condition_on_previous_text=False, vad_filter=False,
                )
                segs = list(segs)
                words = [{"w": w.word.strip(), "p": round(float(w.probability), 4)} for s in segs for w in (s.words or [])]
                rec = {
                    "id": it["id"],
                    "text": " ".join(s.text.strip() for s in segs).strip(),
                    "words": words,
                    "avg_logprob": float(np.mean([s.avg_logprob for s in segs])) if segs else None,
                    "no_speech_prob": float(max((s.no_speech_prob for s in segs), default=0.0)),
                }
                if args.nbest > 1:
                    rec["nbest"] = nbest(model, audio, args.nbest)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if k % 100 == 0:
                    print(f"{tag} {cond} {k}/{len(items)} {time.time() - t0:.0f}s", flush=True)
        print(f"{tag} {cond} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
