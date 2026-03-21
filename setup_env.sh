#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=""

log() {
  printf '[setup] %s\n' "$1"
}

pick_python() {
  if [[ -n "${CONDA_PREFIX:-}" ]] && command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
    log "Detected active Conda environment: ${CONDA_DEFAULT_ENV:-unknown}"
    return
  fi

  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
    return
  fi

  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
    return
  fi

  log "Python was not found. Install Python 3.10+ or activate a Conda environment first."
  exit 1
}

ensure_pip() {
  if ! "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
    log "pip is unavailable for $PYTHON_BIN"
    exit 1
  fi
}

install_requirements() {
  log "Installing Python dependencies from requirements.txt"
  "$PYTHON_BIN" -m pip install --upgrade pip
  "$PYTHON_BIN" -m pip install -r "$ROOT_DIR/requirements.txt"
}

run_hardware_check() {
  log "Running hardware sanity check"
  "$PYTHON_BIN" - <<'PY'
from novamind.core.device_manager import detect_hardware_tier, get_device, get_dtype, get_hardware_banner

tier = detect_hardware_tier()
device = get_device()
dtype = get_dtype(tier)
print(get_hardware_banner(device))
print(f"[setup] detected_tier={tier.value}")
print(f"[setup] detected_device={device}")
print(f"[setup] detected_dtype={dtype}")
PY
}

run_verification() {
  log "Executing repository verification suite"
  (cd "$ROOT_DIR" && "$PYTHON_BIN" test_novamind.py)
}

main() {
  log "Bootstrapping Novamind-CS environment"
  pick_python
  log "Using Python interpreter: $PYTHON_BIN"
  "$PYTHON_BIN" --version
  ensure_pip
  install_requirements
  run_hardware_check
  run_verification
  log "System Ready"
}

main "$@"
