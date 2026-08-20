#!/usr/bin/env bash
#
# Wait for the GPU pass to finish, then transcribe + diacritize the corpus.
#
# The two passes cannot overlap: transcribe_gemini.py rewrites metadata.jsonl
# wholesale while `yt2ds run` is still appending to it. So this waits on the
# running pipeline's PID rather than a process-name pattern -- a pattern would
# also match this script's own command line and the wait would fall through
# immediately.
#
#   scripts/run_gemini_after_phase1.sh <phase1-pid>

set -uo pipefail

cd /home/ai2/saudi-youtube-data/youtube-to-dataset

PHASE1_PID="${1:?usage: run_gemini_after_phase1.sh <phase1-pid>}"
DATASET=dataset-socrates
EXPECTED=209
LOG=logs/gemini-transcribe.log

say() { echo "$(date '+%H:%M:%S') $*" | tee -a "$LOG"; }

say "waiting for phase 1 (pid $PHASE1_PID) to finish"
while kill -0 "$PHASE1_PID" 2>/dev/null; do
    sleep 60
done
say "phase 1 process exited"

# The state directory is the authority on what actually completed: a video that
# failed writes no state file, so counting them distinguishes "finished" from
# "died two thirds of the way through".
complete=$(grep -l '"complete": true' "$DATASET"/.work/state/*.json 2>/dev/null | wc -l)
say "episodes complete: $complete / $EXPECTED"

if [ "$complete" -lt "$EXPECTED" ]; then
    say "WARNING: $((EXPECTED - complete)) episode(s) did not complete."
    say "Transcribing what is there anyway -- rows for missing episodes simply"
    say "do not exist yet, and re-running phase 1 then this pass picks them up"
    say "(the per-clip cache means nothing already done is paid for twice)."
fi

say "starting gemini transcription + tashkeel (najdi, 32 workers)"
.venv/bin/python scripts/transcribe_gemini.py "$DATASET" \
    --dialect najdi \
    --workers 32 \
    -v >> "$LOG" 2>&1
status=$?
say "transcribe_gemini exited with status $status"

if [ "$status" -eq 0 ]; then
    say "done. Remaining step, run manually once you are happy with the text:"
    say "  yt2ds report $DATASET --link-speakers"
fi
exit "$status"
