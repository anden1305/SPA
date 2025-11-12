#!/bin/bash
#BSUB -J run_training_synth                    
#BSUB -q gpul40s
#BSUB -W 01:00  
#BSUB -n 4
#BSUB -R "span[hosts=1]"                      
#BSUB -R "rusage[mem=1GB]"
#BSUB -gpu "num=1:mode=exclusive_process"  
#BSUB -o hpc/output/test_run_synth%J.out          
#BSUB -e hpc/output/test_run_synth%J.err     


# module purge
module load cuda/12.8.1
source .venv/bin/activate
mkdir -p hpc/output/

export OMP_NUM_THREADS=${THREADS}
export MKL_NUM_THREADS=${THREADS}
export OPENBLAS_NUM_THREADS=${THREADS}


# Threading: match your core allocation (e.g., -n 4 => use 3 threads for libs, leave 1 for main)
if [ -n "${LSB_DJOB_NUMPROC}" ]; then
	THREADS=$((${LSB_DJOB_NUMPROC}-1))
	if [ ${THREADS} -lt 1 ]; then THREADS=1; fi
else
	THREADS=3
fi

# Optional: improve NUMA locality. Uncomment ONE of the following lines to test.
# export SPA_USE_NUMACTL=1
# NUMACTL_OPTS="--cpunodebind=0 --membind=0"   # bind to one socket (low latency, limited bandwidth)
# NUMACTL_OPTS="--interleave=all"               # interleave memory across sockets (uniform latency)

# Data loader and device transfer tuning
export SPA_BUILD_WORKERS=${THREADS}
export SPA_PRELOAD_TO_DEVICE=1
# Optional trainer timings (set to 1 to log per-batch averages at validation)
# export SPA_TRAIN_TIMING=1

# Prefer manual redirection to reduce LSF I/O overhead; keep -o/-e as fallback
PYTHON_CMD="python3 -u main.py --config src/config/run/hmm/experiments/experiment_synth_features.yaml"

if [ "${SPA_USE_NUMACTL}" = "1" ] && command -v numactl >/dev/null 2>&1; then
	numactl ${NUMACTL_OPTS} ${PYTHON_CMD} \
		1> hpc/output/stdout_${LSB_JOBID}.log \
		2> hpc/output/stderr_${LSB_JOBID}.log
else
	${PYTHON_CMD} \
		1> hpc/output/stdout_${LSB_JOBID}.log \
		2> hpc/output/stderr_${LSB_JOBID}.log
fi
