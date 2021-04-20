#!/bin/bash
#SBATCH --mem=10g
#SBATCH -c 6
#SBATCH --time=1-0
#SBATCH --gres=gpu:1,vmem:8g

dir=/cs/labs/daphna/avihu.dekel/simCLR/
cd $dir
source /cs/labs/daphna/avihu.dekel/env/bin/activate
module load torch
python run.py --lambda 0.1 --workers 6
