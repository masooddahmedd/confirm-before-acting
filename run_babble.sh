#!/usr/bin/env bash
# Second pass, after run_all.sh: builds the background-chatter conditions and transcribes them.
# Their agent calls and the self-consistency baseline are in run_extra.sh.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
COND="phone_babble5 phone_babble0"
$PY degrade.py --babble
$PY asr.py --model mobiuslabsgmbh/faster-whisper-large-v3-turbo --conditions $COND &
$PY asr.py --model Systran/faster-whisper-large-v3 --nbest 1 --conditions $COND &
wait
echo BABBLE_ASR_DONE
