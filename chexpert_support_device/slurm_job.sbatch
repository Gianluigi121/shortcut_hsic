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

python -m chexpert_support_device.main \
--skew_train='True' \
--p_tr=0.7 \
--p_val=0.25 \
--pixel=128 \
--data_dir='/nfs/turbo/coe-rbg/mmakar/multiple_shortcut/chexpert' \
--exp_dir='/nfs/turbo/coe-rbg/mmakar/multiple_shortcut/chexpert/tuning/trial' \
--checkpoint_dir='/scratch/mmakar_root/mmakar0/mmakar/multiple_shortcut/chexpert/tuning/trial' \
--architecture='pretrained_densenet' \
--batch_size=64 \
--num_epochs=1 \
--training_steps=0 \
--alpha=0.0 \
--sigma=10.0 \
--weighted='False' \
--conditional_hsic='False' \
--l2_penalty=0.0 \
--embedding_dim=-1 \
--random_seed=0 \
--cleanup='False' \
--debugger='True' > /nfs/turbo/coe-rbg/mmakar/multiple_shortcut/chexpert/logs/test.log 2>&1
