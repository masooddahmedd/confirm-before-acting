#!/usr/bin/env bash
# Main pipeline on one GPU: degraded audio -> two recognisers -> two agents -> evaluation.
# Needs data/ from prepare_data.py and ESC-50 unpacked in data/ (see README).
# Every stage resumes from its own cache, so re-running after an interruption only does what is missing.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
TURBO=mobiuslabsgmbh/faster-whisper-large-v3-turbo
V3=Systran/faster-whisper-large-v3
DEGRADED="phone phone_noise10 phone_noise0"

[ -d data/ESC-50-master/audio ] || { echo "ESC-50 not found in data/ESC-50-master (see README)"; exit 1; }
[ -d data/phone_noise0 ] && [ "$(ls data/phone_noise0 | wc -l)" -ge 840 ] || $PY degrade.py --data data --esc50 data/ESC-50-master/audio

$PY asr.py --model $TURBO --conditions clean $DEGRADED &
$PY asr.py --model $V3 --nbest 1 --conditions clean $DEGRADED &
wait

TAGS="mobiuslabsgmbh_faster-whisper-large-v3-turbo Systran_faster-whisper-large-v3"
$PY agent.py --model qwen2.5:7b-instruct --asr-tags $TAGS
$PY agent.py --model llama3.1:8b --asr-tags $TAGS
echo ALL_DONE
