#!/bin/bash
### Convenience wrapper: submit all three cHMMGMVAE LOLO holdout jobs at once.

set -euo pipefail

bash hpc/submit/hmmgmm/run_lolo_subject_lab.sh
