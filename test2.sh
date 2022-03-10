#!/bin/bash
#SBATCH --job-name=test2
#SBATCH --mail-user=zhengji@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=12GB
#SBATCH --time=6:00:00
#SBATCH -A precisionhealth_owned1
#SBATCH -p precisionhealth
#SBATCH --gres=gpu:1
#SBATCH --output=/nfs/turbo/coe-rbg/zhengji/age/test2.log

python -m train