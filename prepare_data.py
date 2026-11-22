# Downloads the BFCL v3 audio benchmark (Apache-2.0, ServiceNow-AI/BFCL_v3_audio) and writes one wav per
# item plus items.jsonl with the question text, tool specs and the gold reference.
#
#   python prepare_data.py --out data
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset

CONFIGS = ["BFCL_v3_simple", "BFCL_v3_multiple", "BFCL_v3_irrelevance"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "clean").mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out / "items.jsonl", "w", encoding="utf-8") as f:
        for cfg in CONFIGS:
            ds = load_dataset("ServiceNow-AI/BFCL_v3_audio", cfg, split="test")
            ds = ds.cast_column("audio", Audio(decode=False))
            for row in ds:
                import io

                wav, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
                if wav.ndim > 1:
                    wav = wav.mean(axis=1)
                if sr != 16000:
                    import librosa

                    wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
                    sr = 16000
                item_id = f"{cfg.replace('BFCL_v3_', '')}_{row['id']}".replace("/", "_")
                sf.write(out / "clean" / f"{item_id}.wav", wav.astype(np.float32), sr)
                f.write(json.dumps({
                    "id": item_id,
                    "subset": cfg.replace("BFCL_v3_", ""),
                    "question": row["question"],
                    "tools": row["tools"],
                    "reference": row["reference"],
                    "duration": round(len(wav) / sr, 3),
                }, ensure_ascii=False) + "\n")
                n += 1
    print(f"wrote {n} items to {out}")


if __name__ == "__main__":
    main()
