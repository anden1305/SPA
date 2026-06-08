#!/bin/bash
# cGMVAE incohort: encoder+decoder conditioning on locked winner recipes (3 labs).
# Generate first:
#   PYTHONPATH=. python3 scripts/cv4fold/generate_ablation_enc_dec_conditioning.py
# Run with bash, NOT `bsub < ...`.

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "${SPA_ROOT}"

MEM="6GB"
CPUS="4"
WALLTIME="6:00"
QUEUE="gpuv100"

mkdir -p hpc/output/cv4fold/ablation_conditioning

for lab in lab_2 lab_3 lab_5; do
  config="src/config/run/cvaemarhmm/cv4fold/ablation_conditioning/${lab}/enc_dec_locked.yaml"
  if [[ ! -f "${config}" ]]; then
    echo "Missing ${config} — run generate_ablation_enc_dec_conditioning.py" >&2
    exit 1
  fi

  bsub <<EOF
#!/bin/bash
#BSUB -J abl_encdec_${lab}
#BSUB -q ${QUEUE}
#BSUB -W ${WALLTIME}
#BSUB -n ${CPUS}
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=${MEM}]"
#BSUB -o hpc/output/cv4fold/ablation_conditioning/${lab}_enc_dec_%J.out
#BSUB -e hpc/output/cv4fold/ablation_conditioning/${lab}_enc_dec_%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

set -euo pipefail
cd /work3/s204070/SPA
module load cuda/12.8.1
source .venv/bin/activate
python3 main.py --method train_vae --config_path ${config}
EOF

  echo "Submitted: abl_encdec_${lab} (${QUEUE}, ${WALLTIME})"
done

echo "Done. Monitor: bjobs -u \$USER | grep abl_encdec"
