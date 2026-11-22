# Makes the listening conditions from the clean benchmark audio:
#   clean          the benchmark audio as released
#   phone          300-3400 Hz band, 8 kHz, G.711 mu-law, back to 16 kHz (what a PSTN call delivers)
#   phone_noise10  ESC-50 environmental noise at 10 dB SNR, then the phone channel
#   phone_noise0   the same at 0 dB SNR
#   phone_babble5  three other talkers from the benchmark at 5 dB, then the phone channel (--babble)
#   phone_babble0  the same at 0 dB
# Noise clips are drawn with a fixed seed per item, so every run builds identical audio.
#
#   python degrade.py --data data --esc50 data/ESC-50-master/audio
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfiltfilt

SR = 16000
MU = 255.0


def phone_channel(x: np.ndarray) -> np.ndarray:
    sos = butter(6, [300, 3400], btype="bandpass", fs=SR, output="sos")
    y = sosfiltfilt(sos, x)
    y = resample_poly(y, 1, 2)  # 16 kHz -> 8 kHz
    peak = max(np.abs(y).max(), 1e-9)
    y = y / peak * 0.9
    # G.711 mu-law: compress, quantise to 8 bits, expand
    c = np.sign(y) * np.log1p(MU * np.abs(y)) / np.log1p(MU)
    c = np.round((c + 1) / 2 * 255) / 255 * 2 - 1
    y = np.sign(c) * (np.power(1 + MU, np.abs(c)) - 1) / MU
    y = y * peak / 0.9
    return resample_poly(y, 2, 1).astype(np.float32)  # back to 16 kHz for the recogniser


def add_noise(x: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    if len(noise) < len(x):
        noise = np.tile(noise, int(np.ceil(len(x) / len(noise))))
    noise = noise[: len(x)]
    px = np.mean(x**2) + 1e-12
    pn = np.mean(noise**2) + 1e-12
    noise = noise * np.sqrt(px / (pn * 10 ** (snr_db / 10)))
    y = x + noise
    peak = np.abs(y).max()
    return (y / peak * 0.95 if peak > 0.99 else y).astype(np.float32)


def load_noises(esc_dir: Path) -> list[np.ndarray]:
    out = []
    for p in sorted(esc_dir.glob("*.wav")):
        w, sr = sf.read(p, dtype="float32")
        if w.ndim > 1:
            w = w.mean(axis=1)
        if sr != SR:
            w = resample_poly(w, SR, sr).astype(np.float32)
        if np.mean(w**2) > 1e-6:  # skip near-silent clips, they would not reach the target SNR
            out.append(w)
    return out


def seed_for(item_id: str) -> int:
    return int(hashlib.sha256(item_id.encode()).hexdigest()[:8], 16)


def babble(item_id: str, items: list[dict], data: Path, n_talkers: int = 3) -> np.ndarray:
    """Background chatter: three other requests from the benchmark, each a different recording, summed."""
    rng = np.random.default_rng(seed_for(item_id + "/babble"))
    others = [it["id"] for it in items if it["id"] != item_id]
    picks = rng.choice(len(others), size=n_talkers, replace=False)
    mix = None
    for p in picks:
        w, _ = sf.read(data / "clean" / f"{others[p]}.wav", dtype="float32")
        w = w / (np.sqrt(np.mean(w**2)) + 1e-9)
        if mix is None:
            mix = w
        else:
            L = max(len(mix), len(w))
            mix = np.pad(mix, (0, L - len(mix))) + np.pad(w, (0, L - len(w)))
    return mix


def main_babble(data: Path, items: list[dict]):
    for snr in (5, 0):
        (data / f"phone_babble{snr}").mkdir(exist_ok=True)
    for it in items:
        x, sr = sf.read(data / "clean" / f"{it['id']}.wav", dtype="float32")
        b = babble(it["id"], items, data)
        for snr in (5, 0):
            sf.write(data / f"phone_babble{snr}" / f"{it['id']}.wav", phone_channel(add_noise(x, b, snr)), SR)
    print("babble done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--esc50", default="data/ESC-50-master/audio")
    ap.add_argument("--babble", action="store_true", help="only build the background-chatter conditions")
    args = ap.parse_args()
    data = Path(args.data)
    items = [json.loads(l) for l in open(data / "items.jsonl", encoding="utf-8")]
    if args.babble:
        return main_babble(data, items)
    noises = load_noises(Path(args.esc50))
    print(f"{len(items)} items, {len(noises)} noise clips")
    for cond in ["phone", "phone_noise10", "phone_noise0"]:
        (data / cond).mkdir(exist_ok=True)
    for it in items:
        x, sr = sf.read(data / "clean" / f"{it['id']}.wav", dtype="float32")
        assert sr == SR
        rng = np.random.default_rng(seed_for(it["id"]))
        noise = noises[rng.integers(len(noises))]
        sf.write(data / "phone" / f"{it['id']}.wav", phone_channel(x), SR)
        for snr in (10, 0):
            sf.write(data / f"phone_noise{snr}" / f"{it['id']}.wav", phone_channel(add_noise(x, noise, snr)), SR)
    print("done")


if __name__ == "__main__":
    main()
