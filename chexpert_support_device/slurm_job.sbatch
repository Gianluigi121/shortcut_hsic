#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --output=gpu.out
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1

#SBATCH --account=precisionhealth_owned1
#SBATCH --partition=precisionhealth
#SBATCH --mem-per-gpu=20000m

# TODO check if there is room on the dirs 

# TODO check if overwrite 

python -m chexpert_support_device.main > /nfs/turbo/coe-rbg/mmakar/multiple_shortcut/chexpert/logs/test.log 2>&1 
