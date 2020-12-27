#!/bin/bash
#SBATCH --account=def-jiayuan
#SBATCH --cpus-per-task=40
#SBATCH --time=0-12:00
#SBATCH --mem=64GB

#!/bin/bash

dirname=$1

module load python/3.6
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index -r requirements.txt

python run_sim.py
