#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-$(date +%Y%m%d_%H%M%S)}"
BASE_LOGS="/workspace/logs/experiments/${RUN_ID}"
BASE_OUT="/workspace/outputs/experiments/${RUN_ID}"

echo "[experiments] RUN_ID=${RUN_ID}"

# Preparar dados uma vez
docker compose up data_prep

run_mode() {
  local mode="$1"   # async | sync
  echo "[experiments] running mode=${mode}"

  export LOGS_DIR="${BASE_LOGS}/${mode}"
  export OUT_DIR="${BASE_OUT}/${mode}"

  if [[ "${mode}" == "sync" ]]; then
    export SYNC_MODE=1
  else
    export SYNC_MODE=0
  fi

  export EDGE1_CLIENTS="iot1,iot2"
  export EDGE2_CLIENTS="iot3,iot4"

  docker compose up --build -d

  # Aguarda todos os IoTs finalizarem
  while true; do
    running=0
    for c in iot1 iot2 iot3 iot4; do
      if docker inspect -f '{{.State.Running}}' "$c" 2>/dev/null | grep -q true; then
        running=1
      fi
    done
    if [[ $running -eq 0 ]]; then
      break
    fi
    sleep 5
  done

  docker compose down

  # Gera análises para este modo
  docker compose run --rm analyzer python -m project.analysis.extract_metrics
  docker compose run --rm analyzer python -m project.analysis.plot_curves
  docker compose run --rm analyzer python -m project.analysis.summarize
  docker compose run --rm analyzer python -m project.analysis.paper_metrics
  docker compose run --rm analyzer python -m project.analysis.paper_report
}

run_mode async
run_mode sync

export LOGS_DIR="${BASE_LOGS}/async"
export OUT_DIR="${BASE_OUT}/async"
docker compose run --rm analyzer python -m project.analysis.paper_report --compare-dir "${BASE_OUT}/sync"
docker compose run --rm analyzer python -m project.analysis.compare_async_sync --async-dir "${BASE_OUT}/async" --sync-dir "${BASE_OUT}/sync"

echo "[experiments] done. Logs in ./logs/experiments/${RUN_ID}, outputs in ./outputs/experiments/${RUN_ID}"
