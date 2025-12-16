# HPC Submit Script Guide (SLURM-focused)

This guide explains how to write and use HPC submit scripts (SLURM `sbatch`) with practical, copy-ready examples: simple jobs, job arrays, capturing job IDs, dependencies, GPU/MPI flags, monitoring, and best practices. Adapt flags to your cluster's policies.

## Anatomy of a submit script

A submit script is a shell script with SBATCH directives near the top.

Minimum structure:

```bash
#!/usr/bin/env bash

## Advanced patterns

#BSUB -W 45
```
- GPUs: `#SBATCH --gres=gpu:1` or with type `--gres=gpu:tesla:2`
# HPC Submit Script Guide (LSF / DTU-focused)

This guide explains how to write and use HPC submit scripts for LSF-based clusters (as used at DTU). It contains copy-ready examples for submit scripts, job arrays, capturing job IDs, dependencies, resource flags (CPU/GPU/memory), DTU helper commands, monitoring, best practices, and common pitfalls. Adapt queue/feature names to your specific DTU cluster.

## Overview — what is LSF?

LSF (IBM Platform LSF) is a job scheduler/resource manager commonly used on HPC clusters. You submit job scripts or commands with `bsub`, the scheduler queues and dispatches jobs to available nodes, and you can monitor and manage jobs with a suite of commands (`bjobs`, `bstat`, `bhist`, `bkill`, etc.). DTU provides helpful wrapper utilities such as `bstat`, `nodestat` and `classstat` described in DTU docs.

## Anatomy of an LSF submit script

An LSF submit script is a shell script where in-script directives use `#BSUB` lines. Minimal structure:

```bash
#!/usr/bin/env bash
#BSUB -J myjob                 # job name
#BSUB -o logs/%J.out           # %J = job id
#BSUB -e logs/%J.err
#BSUB -n 4                     # number of cores (tasks)
#BSUB -R "rusage[mem=4000]"   # memory usage (per slot in MB)
#BSUB -W 60                    # runtime limit in minutes
#BSUB -q hpc                   # queue name (partition)

set -euo pipefail

# optional: load modules or activate conda
# module load python/3.10
# source /path/to/conda_activate.txt

python myscript.py
```

Notes:
- `%J` is the job id placeholder in LSF formatting. Some sites also support other placeholders (array index formatting varies — see below).
- `#BSUB -R` is used to express resource requirements; the syntax can be cluster-specific.
- `#BSUB -n` requests cores/tasks. For MPI runs you may combine `-n` and host/resource requests.

## Useful environment variables inside LSF jobs

- `LSB_JOBID` — the job id (often available as `$LSB_JOBID`).
- `LSB_JOBINDEX` — the array task index when running job arrays.
- Check `man bsub` on DTU for site-specific variables and conventions.

## Submitting jobs

Submit a script with:

```bash
bsub < my_job.sh
```

Or submit from the command line with flags:

```bash
bsub -J myjob -o logs/%J.out -n 2 -W 30 -q hpc -- some_command
```

To programmatically capture the job id returned by `bsub` (it prints a line like `Job <44951> is submitted to default queue .`):

```bash
line=$(bsub < my_job.sh)
jid=$(echo "$line" | sed -n 's/.*<\([0-9]*\)>.*/\1/p')
echo "LSF job id: $jid"
```

## Job arrays (parameter sweeps / many independent tasks)

LSF supports job arrays using the `-J` flag with bracket notation. Concurrency throttling is supported with `%N` similarly to SLURM.

Submit an array from the command line:

```bash
bsub -J "sweep[1-100%20]" < my_array_job.sh
```

Inside `my_array_job.sh` use `$LSB_JOBINDEX` to pick parameters:

```bash
#!/usr/bin/env bash
#BSUB -J sweep
#BSUB -o logs/sweep_%J_%I.out   # note: %I usage may vary by site
#BSUB -e logs/sweep_%J_%I.err

echo "array index = $LSB_JOBINDEX"
param=$(sed -n "${LSB_JOBINDEX}p" params.txt)
python run_experiment.py --param "$param"
```

Notes and tips:
- Concurrency control: `sweep[1-100%20]` runs the array indices 1..100 but only 20 concurrently.
- Use mapping files (e.g., `params.txt`) to map indices to parameter sets; this keeps the job invocation simple.
- Check DTU's `bsub`/`bjobs` docs for which formatting placeholders (`%I`, `%J`) are supported on their cluster.

## Dependencies (chain jobs)

LSF supports conditional submission using the `-w` option with logical expressions. Some common patterns:

- Run job B after job A completes successfully:

```bash
# submit job A and capture id
line=$(bsub < jobA.sh)
MPI example:

# submit job B that waits for A to finish successfully
bsub -w "done(${jid})" < jobB.sh
```

- Combine dependencies: `-w "done(12345) && done(12346)"`.

- Use `bsub -w "ended(12345)"` or other LSF conditions depending on the exact semantics you need (check `man bsub`).

## MPI / multi-node / GPU resource flags

LSF resource requests can be cluster-specific. Common examples:

- Cores/tasks: `#BSUB -n 16`
- Memory per slot: `#BSUB -R "rusage[mem=4000]"` (memory in MB)
- GPUs: some clusters use `-R "rusage[ngpus=1]"` or other `-gpu`/`-R` resource specs — check DTU docs or `nodestat -g` for the exact flags.
- Wall-time: `#BSUB -W 120` (minutes)

MPI example (site-specific launcher may vary):

```bash
#BSUB -n 32
#BSUB -R "span[ptile=16]"
#BSUB -W 120

# then run MPI launcher used by the site, e.g.:
mpirun -np 32 ./mpi_program
# or site-specific: jsrun / srun-like commands depending on cluster
```

## Full LSF workflow example — sweep + aggregate

`submit_sweep_lsf.sh` (array example):

```bash
#!/usr/bin/env bash
#BSUB -J sweep[1-200%40]
#BSUB -o logs/sweep_%J_%I.out
#BSUB -e logs/sweep_%J_%I.err
#BSUB -n 2
#BSUB -W 45

param=$(sed -n "${LSB_JOBINDEX}p" params.txt)
python run_experiment.py --param "$param" --out results/result_${LSB_JOBINDEX}.npy
```

Submit and capture the array job id:

```bash
line=$(bsub -J "sweep[1-200%40]" < submit_sweep_lsf.sh)
array_jid=$(echo "$line" | sed -n 's/.*<\([0-9]*\)>.*/\1/p')
echo "array job: $array_jid"
```

`aggregate_lsf.sh` (aggregator):

```bash
#!/usr/bin/env bash
#BSUB -J aggregate
#BSUB -o logs/aggregate_%J.out
#BSUB -n 2
#BSUB -W 10

python aggregate_results.py results/ output_summary.json
```

Submit the aggregator conditional on the sweep finishing successfully:

```bash
bsub -w "done(${array_jid})" < aggregate_lsf.sh
```

## Monitoring, DTU helpers and useful commands

DTU provides helpful wrapper commands that present compact summaries and resource usage (check the DTU docs for the most current list):

- `bstat` — DTU helper that summarizes LSF jobs (compact format). Use `bstat -C` for CPU efficiency or `bstat -M` for memory stats.
- `bjobs` — list running/pending jobs (native LSF command).
- `bhist`, `bacct` — job history and accounting.
- `bpeek` — peek at output of running jobs.
- `bkill` — cancel a job (`bkill <jobid>`), or send a signal (`bkill -s SIGTERM <jobid>`).
- `nodestat`, `classstat`, `showstart` — DTU helpers for nodes/queues/estimated start times.

Quick checks on a DTU login node:

```bash
which bsub && bsub -V
bjobs -u $(whoami)
bstat -h
nodestat -F hpc
```

## Best practices and troubleshooting

- Logging: include job and array ids in log names, e.g. `#BSUB -o logs/%J_%I.out` to avoid overwriting.
- Reproducible environment: source `conda_activate.txt` or load modules inside the script. Do not rely on login node environment being present on compute nodes.
- Throttle concurrency: use `[%N]` with arrays so you don't overwhelm schedulers (and to be cluster-friendly).
- Resource sizing: request realistic wall-time and memory to increase scheduler throughput and reduce wasted time.
- Small tasks: group many small tasks into arrays instead of submitting thousands of separate jobs.
- Diagnostics: use `bstat -C` and `bstat -M` to check CPU efficiency and memory usage; increase `-R` resource requests if jobs are killed due to memory limits.

## Common pitfalls

- Placeholders and env vars differ from SLURM: LSF uses `%J` / `%I` and `$LSB_JOBINDEX` — confirm the site-specific placeholders.
- Resource flag syntax (`-R`) can be cluster-specific — check DTU docs and `man bsub`.
- Writing to shared output files without unique names will cause clobbering; include job/array ids in file names.

## Quick copyable commands (LSF)

Submit a job script:

```bash
bsub < my_job.sh
```

Submit an array (command-line) and capture id:

```bash
line=$(bsub -J "sweep[1-50%10]" < my_array_job.sh)
jid=$(echo "$line" | sed -n 's/.*<\([0-9]*\)>.*/\1/p')
echo "array job submitted: $jid"
```

Submit dependent job:

```bash
bsub -w "done(${jid})" < postprocess.sh
```

Check status and history:

```bash
bjobs -u $(whoami)
bhist -l <jobid>
bstat -C
```

## Where this ties into your repo

- Your `scripts/` folder contains several helpers and batch wrappers; use them as templates and convert their `#SBATCH`/`sbatch` style into `#BSUB`/`bsub` style if you will run on DTU.

## Good practices

- Use `--output=logs/%x_%A_%a.out` and `--error=logs/%x_%A_%a.err` for arrays.
- Add `set -euo pipefail` to scripts.
- Request only the resources you need.
- Limit array concurrency with `%N` to be cluster-friendly.

## Common pitfalls

- Forgetting to set up the environment (modules/venv) inside the job.
- Requesting too many resources (time/memory) — worsens scheduling time.
- Submitting huge numbers of small single-job requests instead of arrays.

## Summary and next steps

- This file provides ready-to-use SBATCH patterns: arrays, concurrency limits, capturing job ids with `--parsable`, and chaining with `--dependency`.

If you'd like, I can:
- Convert a specific script in `scripts/` to use arrays + dependencies and add tests/examples.
- Create a small runnable example directory with `params.txt`, `submit_sweep.sh`, and a simple `run_experiment.py` to demonstrate locally.

---

*File created in repository:* `HPC_SUBMIT_GUIDE.md`




# MAKE A SWEEP ID - example
python3 -m src.training.wandb_sweep_runner src/config/sweep/marhmm/marhmm_mar_proof.yaml --create-only

returns the SWEEP ID, e.g. "k064frrl"

# SUBMIT THE SWEEP
python3 -m src.training.wandb_sweep_runner src/config/sweep/marhmm/marhmm_mar_proof.yaml --agent-only --sweep-id YOUR_SWEEP_ID --trials-per-agent 1


