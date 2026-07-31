#!/usr/bin/env bash
#
# Build both virtualenvs.
#
#   .venv           main pipeline: torch 2.9 + transformers 5.x (Cohere ASR,
#                   AST, Demucs, ECAPA, MMS aligner, SQUIM, yt-dlp, Silero)
#   .venv-diarize   DiariZen only: torch 2.1.1 + vendored pyannote-audio
#
# They are deliberately separate. DiariZen pins an older torch and vendors
# pyannote-audio; Cohere Transcribe Arabic needs transformers>=5.4. Trying to
# satisfy both in one environment does not work, so they talk over a subprocess
# boundary (scripts/diarize_worker.py) instead.
#
# Usage:
#   scripts/setup.sh              # both venvs
#   scripts/setup.sh main         # main venv only
#   scripts/setup.sh diarize      # diarization venv only
#   scripts/setup.sh prefetch     # download model weights ahead of a run

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TARGET="${1:-all}"

MAIN_VENV="$ROOT/.venv"
DIARIZE_VENV="$ROOT/.venv-diarize"
DIARIZE_SRC="$ROOT/.diarizen"

# CUDA wheel index for each stack. Ada (sm_89) is supported by both.
MAIN_TORCH_INDEX="https://download.pytorch.org/whl/cu128"
DIARIZE_TORCH_INDEX="https://download.pytorch.org/whl/cu121"

log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31merror: %s\033[0m\n' "$*" >&2; exit 1; }

need() { command -v "$1" >/dev/null 2>&1 || die "$1 not found on PATH"; }

need uv
need git
need ffmpeg
need ffprobe

# ---------------------------------------------------------------- main venv
setup_main() {
  log "main venv: $MAIN_VENV"
  [ -d "$MAIN_VENV" ] || uv venv --python 3.10 "$MAIN_VENV"

  log "installing torch 2.9.1 (cu128)"
  uv pip install --python "$MAIN_VENV/bin/python" \
    torch==2.9.1 torchaudio==2.9.1 --index-url "$MAIN_TORCH_INDEX"

  log "installing yt2ds and its dependencies"
  uv pip install --python "$MAIN_VENV/bin/python" -e ".[dev]"

  "$MAIN_VENV/bin/python" - <<'PY'
import torch, transformers
from transformers import CohereAsrForConditionalGeneration  # noqa: F401
import torchaudio.functional as F
from torchaudio.pipelines import SQUIM_OBJECTIVE  # noqa: F401
assert hasattr(F, "forced_align"), "torchaudio is missing forced_align"
print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()}")
print(f"  transformers {transformers.__version__}")
PY
  log "main venv ready"
}

# ------------------------------------------------------------ diarize venv
setup_diarize() {
  log "diarization venv: $DIARIZE_VENV"

  if [ ! -d "$DIARIZE_SRC" ]; then
    log "cloning DiariZen"
    git clone --depth 1 https://github.com/BUTSpeechFIT/DiariZen.git "$DIARIZE_SRC"
  fi

  [ -d "$DIARIZE_VENV" ] || uv venv --python 3.10 "$DIARIZE_VENV"

  log "installing torch 2.1.1 (cu121)"
  uv pip install --python "$DIARIZE_VENV/bin/python" \
    torch==2.1.1 torchaudio==2.1.1 --index-url "$DIARIZE_TORCH_INDEX"

  log "installing DiariZen requirements"
  # onnxruntime-gpu is listed but unused at inference time and often fails to
  # resolve against older CUDA stacks; drop it rather than block the install.
  grep -v '^onnxruntime-gpu' "$DIARIZE_SRC/requirements.txt" \
    | grep -v '^jupyterlab' \
    | grep -v '^pre-commit' > "$DIARIZE_SRC/.requirements.filtered"
  uv pip install --python "$DIARIZE_VENV/bin/python" -r "$DIARIZE_SRC/.requirements.filtered"

  log "installing vendored pyannote-audio"
  uv pip install --python "$DIARIZE_VENV/bin/python" --no-deps -e "$DIARIZE_SRC/pyannote-audio"
  # pyannote-audio's own runtime deps, minus the torch pin we already satisfied.
  # transformers is capped: 4.45+ refuses to bind to torch < 2.4 and silently
  # disables its torch backend, which breaks pyannote's imports. DiariZen only
  # uses transformers on the training/pruning path, so 4.44.2 is ample.
  uv pip install --python "$DIARIZE_VENV/bin/python" \
    "asteroid-filterbanks>=0.4" "pyannote.core>=5.0" "pyannote.database>=5.0" \
    "pyannote.metrics>=3.2" "pyannote.pipeline>=3.0" "torch-audiomentations>=0.11" \
    "torchmetrics>=0.11" "speechbrain>=1.0" "lightning>=2.0" "rich" "semver" \
    "huggingface_hub>=0.13" "omegaconf" "pytorch-metric-learning" "transformers==4.44.2"

  log "installing diarizen"
  uv pip install --python "$DIARIZE_VENV/bin/python" --no-deps -e "$DIARIZE_SRC"

  "$DIARIZE_VENV/bin/python" - <<'PY'
import torch
from diarizen.pipelines.inference import DiariZenPipeline  # noqa: F401
print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()}")
print("  DiariZenPipeline import OK")
PY
  log "diarization venv ready"
}

# ------------------------------------------------------------- model cache
prefetch() {
  log "prefetching model weights"
  "$MAIN_VENV/bin/python" - <<'PY'
from yt2ds.config import Config
from yt2ds.models import ModelRegistry

cfg = Config.load()
reg = ModelRegistry(cfg)
for name in ("vad", "ast", "demucs", "speaker_encoder", "asr", "aligner",
             "squim_objective", "squim_subjective"):
    try:
        getattr(reg, name)
        print(f"  ok   {name}")
    except Exception as exc:
        print(f"  FAIL {name}: {type(exc).__name__}: {exc}")
PY

  if [ -x "$DIARIZE_VENV/bin/python" ]; then
    "$DIARIZE_VENV/bin/python" - <<'PY'
try:
    from diarizen.pipelines.inference import DiariZenPipeline
    DiariZenPipeline.from_pretrained("BUT-FIT/diarizen-wavlm-large-s80-md-v2")
    print("  ok   diarizen")
except Exception as exc:
    print(f"  FAIL diarizen: {type(exc).__name__}: {exc}")
PY
  fi
}

case "$TARGET" in
  main)     setup_main ;;
  diarize)  setup_diarize ;;
  prefetch) prefetch ;;
  all)      setup_main; setup_diarize; prefetch ;;
  *)        die "unknown target: $TARGET (expected main|diarize|prefetch|all)" ;;
esac

log "done"
