#!/bin/bash
# Queue mix per 10 jobs: 5× gpuv100, 2× gpua100, 1× gpua10, 1× gpul40s, 1× gpua40.
# Uses global JOB_IDX. Call _pick_queue then read PICKED_QUEUE (no $() — subshell breaks JOB_IDX).

_pick_queue() {
  local r=$(( JOB_IDX % 10 ))
  JOB_IDX=$(( JOB_IDX + 1 ))
  if [[ $r -lt 5 ]]; then
    PICKED_QUEUE="gpuv100"
  elif [[ $r -lt 7 ]]; then
    PICKED_QUEUE="gpua100"
  elif [[ $r -eq 7 ]]; then
    PICKED_QUEUE="gpua10"
  elif [[ $r -eq 8 ]]; then
    PICKED_QUEUE="gpul40s"
  else
    PICKED_QUEUE="gpua40"
  fi
}
