#!/bin/bash
# Supplemental K-sweep: 2 extra seeds only (127, 128) — does NOT rerun seeds 1–3.
#
# Generate:
#   PYTHONPATH=. python3 scripts/cv4fold/generate_chmm_k_sweep_extra_seeds.py --fold 4
# Submit:
#   bash hpc/submit/cv4fold/submit_chmm_k_sweep_extra2_v100.sh

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "${SPA_ROOT}"

FOLD=4
QUEUE="gpuv100"
MEM="6GB"
CPUS="4"
WALLTIME="1:30"
K_VALUES=($(seq 3 15))

mkdir -p hpc/output/cv4fold/paper_k_sweep

for k in "${K_VALUES[@]}"; do
  config="src/config/run/cvaemarhmm/cv4fold/paper_k_sweep/fold_${FOLD}/chmmgmvae_K${k}_extra2.yaml"
  if [[ ! -f "${config}" ]]; then
    echo "Missing ${config} — run generate_chmm_k_sweep_extra_seeds.py" >&2
    exit 1
  fi
  tag="cv4_kx2_f${FOLD}_K${k}"

  bsub <<EOF
#!/bin/bash
#BSUB -J ${tag}
#BSUB -q ${QUEUE}
#BSUB -W ${WALLTIME}
#BSUB -n ${CPUS}
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=${MEM}]"
#BSUB -o hpc/output/cv4fold/paper_k_sweep/k_extra2_K${k}_%J.out
#BSUB -e hpc/output/cv4fold/paper_k_sweep/k_extra2_K${k}_%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

set -euo pipefail
cd /work3/s204070/SPA
module load cuda/12.8.1
source .venv/bin/activate
python3 main.py --method train_vae --config_path ${config}
EOF

  echo "Submitted: ${tag} (${QUEUE}, ${WALLTIME})"
done

echo ""
echo "Done. ${#K_VALUES[@]} extra-seed jobs (2 seeds each, no rerun of 1–3)."
echo "Monitor: bjobs -J cv4_kx2_f${FOLD}"
