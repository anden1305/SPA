#!/bin/bash
# Queue routing for cv4fold (source from submit_*.sh).
#
# Capacity (typical): gpuv100 many parallel; gpua100 1–2; gpua40 1; gpul40s 1.
# Aligns with subjectwise_raw: lab-3 mice 056/059/060/069 -> a100/l40s spill.

cv4fold_pick_queue() {
  local cfg="$1"

  # Phase 0 smoke: all on gpuv100 (submit_smoke_80.sh); not routed here
  if [[ "$cfg" == */smoke/* ]]; then
    echo "gpuv100"
    return 0
  fi

  # Phase 1 tune: W&B sweeps via submit_tune_sweep.sh (not routed here)
  if [[ "$cfg" == */tune/* ]]; then
    echo "gpuv100"
    return 0
  fi

  # Full-grid HMM raw (16 jobs)
  if [[ "$cfg" == */hmm/cv4fold/* ]]; then
    if [[ "$cfg" == *per_lab/lab_5/fold_4/* ]]; then
      echo "gpua40"
      return 0
    fi
    if [[ "$cfg" == *per_lab/lab_3/* ]] || [[ "$cfg" == *per_lab/lab_5/* ]]; then
      echo "gpul40s"
      return 0
    fi
    if [[ "$cfg" == *joint/fold_2/* ]] || [[ "$cfg" == *joint/fold_4/* ]]; then
      echo "gpua100"
      return 0
    fi
    echo "gpuv100"
    return 0
  fi

  # Full-grid VAE (144 jobs): lab_3 + slow priors off v100
  if [[ "$cfg" == *per_lab/lab_3/* ]]; then
    echo "gpua100"
    return 0
  fi
  if [[ "$cfg" == *chmmgmvae* ]]; then
    if [[ "$cfg" == *joint/fold_2/* ]] || [[ "$cfg" == *joint/fold_4/* ]]; then
      echo "gpua100"
      return 0
    fi
    if [[ "$cfg" == *per_lab/lab_5/* ]]; then
      echo "gpul40s"
      return 0
    fi
  fi
  if [[ "$cfg" == *hmmgmvae* ]] && [[ "$cfg" == *per_lab/lab_5/fold_4/* ]]; then
    echo "gpul40s"
    return 0
  fi

  echo "gpuv100"
}
