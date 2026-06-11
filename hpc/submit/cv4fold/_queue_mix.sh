#!/bin/bash
# Queue mix per 10 jobs: 8× gpuv100, 1× gpua100, 1× gpul40s (no gpua10/gpua40).
# Canonical mix (5/2/1/1/1) documented in .cursor/rules/hpc-gpu-queue-split.mdc.
# Uses global JOB_IDX. Call _pick_queue then read PICKED_QUEUE (no $() — subshell breaks JOB_IDX).

_pick_queue() {
  local r=$(( JOB_IDX % 10 ))
  JOB_IDX=$(( JOB_IDX + 1 ))
  if [[ $r -lt 8 ]]; then
    PICKED_QUEUE="gpuv100"
  elif [[ $r -eq 8 ]]; then
    PICKED_QUEUE="gpua100"
  else
    PICKED_QUEUE="gpul40s"
  fi
}
