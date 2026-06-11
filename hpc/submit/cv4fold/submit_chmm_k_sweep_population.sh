#!/bin/bash
# Population K-sweep: all 20 mice train+val, 5 seeds, K=3…15 (appendix / biology).
#
# Generate:
#   PYTHONPATH=. python3 scripts/cv4fold/generate_chmm_k_sweep_population.py
# Submit:
#   bash hpc/submit/cv4fold/submit_chmm_k_sweep_population.sh
#
# After jobs finish, plot:
#   bsub < hpc/submit/paper/run_population_biology_meeting.sh

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "${SPA_ROOT}"

QUEUE="gpuv100"
MEM="8GB"
CPUS="4"
WALLTIME="2:30"
K_VALUES=($(seq 3 15))

mkdir -p hpc/output/cv4fold/paper_k_sweep/population

for k in "${K_VALUES[@]}"; do
  config="src/config/run/cvaemarhmm/cv4fold/paper_k_sweep/population/chmmgmvae_K${k}.yaml"
  if [[ ! -f "${config}" ]]; then
    echo "Missing ${config} — run generate_chmm_k_sweep_population.py" >&2
    exit 1
  fi
  tag="cv4_k_sweep_pop_K${k}"

  bsub <<EOF
#!/bin/bash
#BSUB -J ${tag}
#BSUB -q ${QUEUE}
#BSUB -W ${WALLTIME}
#BSUB -n ${CPUS}
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=${MEM}]"
#BSUB -o hpc/output/cv4fold/paper_k_sweep/population/k_sweep_K${k}_%J.out
#BSUB -e hpc/output/cv4fold/paper_k_sweep/population/k_sweep_K${k}_%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

set -euo pipefail
cd /work3/s204070/SPA
module load cuda/12.8.1
source .venv/bin/activate
export PYTHONPATH=/work3/s204070/SPA
python3 main.py --method train_vae --config_path ${config}
EOF

  echo "Submitted: ${tag} (${QUEUE}, ${WALLTIME}, 5 seeds)"
done

echo ""
echo "Done. ${#K_VALUES[@]} population K-sweep jobs on ${QUEUE}."
echo "Monitor: bjobs -u \$USER | grep cv4_k_sweep_pop"
echo "Logs: hpc/output/cv4fold/paper_k_sweep/population/k_sweep_K*_*.out"
