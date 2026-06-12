#!/bin/bash
# Submit population K-sweep FIRST (FIFO), then other v100 jobs.
#   bash hpc/submit/cv4fold/submit_population_retry_then_others.sh

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "${SPA_ROOT}"

echo "=== 1/2 Population K-sweep (priority) ==="
bash hpc/submit/cv4fold/submit_chmm_k_sweep_population_retry.sh

echo ""
echo "=== 2/2 Other cv4fold jobs (queue behind K-sweep) ==="
bash hpc/submit/cv4fold/resubmit_killed_pending_jun12.sh

echo ""
echo "K-sweep PEND jobs are ahead of seed_reruns/kx2/hmm on gpuv100."
