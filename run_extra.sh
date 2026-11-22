#!/usr/bin/env bash
# Third pass, after run_all.sh and run_babble.sh: the background-chatter conditions' agent calls and the
# agent self-consistency baseline. Every stage resumes from its cache.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
TAGS="mobiuslabsgmbh_faster-whisper-large-v3-turbo Systran_faster-whisper-large-v3"
for M in qwen2.5:7b-instruct llama3.1:8b; do
  $PY agent.py --model $M --asr-tags $TAGS --conditions phone_babble5 phone_babble0
  $PY agent_sc.py --model $M --k 4
done
echo EXTRA_DONE
