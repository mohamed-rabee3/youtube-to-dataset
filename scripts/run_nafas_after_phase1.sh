#!/usr/bin/env bash
#
# Passes 2 and 3 of the nafas dataset, chained behind the GPU pass.
#
# Pass 1 (`yt2ds run ... --asr-backend google_batch`) emits clips whose text is
# "pending". This script waits for it, fills the text in through Vertex batch
# prediction -- transcript and tashkeel in the same call -- and then splits the
# corpus into one folder per voice.
#
# It waits on a concrete PID rather than a pgrep pattern: a pattern would also
# match this script's own command line and the wait would fall through at once.
# The wait is not optional -- the transcription pass rewrites metadata.jsonl
# wholesale, so it cannot overlap a pipeline still appending to it.
#
#   scripts/run_nafas_after_phase1.sh <phase1-pid>

set -uo pipefail

cd /home/ai2/saudi-youtube-data/youtube-to-dataset

PHASE1_PID="${1:?usage: run_nafas_after_phase1.sh <phase1-pid>}"
DATASET=dataset-nafas
EXPECTED=101
BUCKET=project-50fbdf1b-139e-4602-bb4-yt2ds-asr
PROJECT=project-50fbdf1b-139e-4602-bb4
DIALECT=najdi
REHEARSAL=40          # clips sent as an end-to-end trial before the whole corpus
MIN_REHEARSAL_OK=20   # below half landing, something is wrong -- do not scale up
LOG=logs/nafas-vertex.log

say() { echo "$(date '+%H:%M:%S') $*" | tee -a "$LOG"; }

# Rows whose text actually arrived. `_write` sets text_source to "gemini" when
# the model returned text and leaves it "pending" when it did not, so this
# counts real successes rather than merely rows touched.
transcribed() {
    grep -c '"text_source": "gemini"' "$DATASET/metadata.jsonl" 2>/dev/null || echo 0
}

vertex() {
    .venv/bin/python scripts/transcribe_vertex_batch.py "$DATASET" \
        --bucket "$BUCKET" --project "$PROJECT" --dialect "$DIALECT" "$@" \
        >> "$LOG" 2>&1
}

say "waiting for phase 1 (pid $PHASE1_PID)"
while kill -0 "$PHASE1_PID" 2>/dev/null; do
    sleep 60
done
say "phase 1 process exited"

# The state directory is the authority on what completed: an episode that
# failed writes no state file, so counting them separates "finished" from
# "died two thirds of the way through".
complete=$(grep -l '"complete": true' "$DATASET"/.work/state/*.json 2>/dev/null | wc -l)
say "episodes complete: $complete / $EXPECTED"

if [ "$complete" -eq 0 ]; then
    say "FATAL: nothing completed -- not sending anything to Vertex."
    exit 1
fi
if [ "$complete" -lt "$EXPECTED" ]; then
    say "WARNING: $((EXPECTED - complete)) episode(s) did not complete."
    say "Transcribing what is there. Re-running phase 1 then this script picks"
    say "up the rest -- the per-clip cache means nothing is paid for twice."
fi

# Rehearsal first. Upload/submit/poll/merge is four moving parts against a
# billed API, and a misconfiguration is far cheaper to find on 40 clips than on
# the whole corpus. The clips it does are cached, so this is not wasted work.
say "rehearsal: $REHEARSAL clips end-to-end"
before=$(transcribed)
vertex --limit "$REHEARSAL" -v
say "rehearsal exited with status $?"
gained=$(( $(transcribed) - before ))
say "rehearsal landed $gained/$REHEARSAL clips"

if [ "$gained" -lt "$MIN_REHEARSAL_OK" ]; then
    say "FATAL: rehearsal landed $gained of $REHEARSAL, below the floor of"
    say "$MIN_REHEARSAL_OK. Stopping rather than spending the whole corpus on a"
    say "path that is not working. See $LOG."
    exit 1
fi

# Two full passes. Failures arrive as whole requests returning nothing rather
# than as bad clips, and a second run re-queues only the non-ok cache rows; on
# the socrates corpus that recovered about 70% of what the first pass lost.
# --clips-per-request is left at its default of 4 deliberately: above that the
# model stops reliably telling one audio part from another and returns clip N's
# words for clip M, with the right count and indices, so nothing looks wrong.
status=0
for attempt in 1 2; do
    say "full pass $attempt: vertex batch transcribe + tashkeel ($DIALECT)"
    vertex -v
    status=$?
    say "full pass $attempt exited with status $status, transcribed now $(transcribed)"
    [ "$status" -ne 0 ] && break
done

if [ "$status" -ne 0 ]; then
    say "transcription failed; stopping before the speaker split so a"
    say "half-filled metadata.jsonl is not baked into speaker folders."
    exit "$status"
fi

# One folder per voice, clips hardlinked into it so the split costs inodes
# rather than a second copy of the corpus. The host is simply the identity with
# the most hours -- the ranking this prints puts it first.
say "splitting speakers into $DATASET/speakers/"
.venv/bin/python scripts/split_speakers.py "$DATASET" -v >> "$LOG" 2>&1
say "split_speakers exited with status $?"

say "done. Review the ranking in $DATASET/speakers/report.json and listen to"
say "the sample clips before treating the top identity as the host."
