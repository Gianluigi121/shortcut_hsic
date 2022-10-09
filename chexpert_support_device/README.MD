# Chexpert experiment with sex as a shortcut and multiple redundant variables 

## Step 1: create the data 
```bash
srun --account=engin1 --partition=standard --time=1:00:00 --tasks-per-node=1 --mem=1gb --pty /bin/bash

source activate env
python -m chexpert_support_device.create_data \
	--save_directory /nfs/turbo/coe-soto/mmakar/multiple_shortcut/chexpert_sd/
```

## Step 2: Train 

### Run one model 
```bash
srun --mem-per-gpu=100000m  --gres=gpu:1 --account=mmakar0 --partition=gpu --time=3:00:00 --pty /bin/bash

source activate env

python -m chexpert_support_device.main \
	--skew_train='True' \
	--v_dim=0 \
	--p_tr=0.7 \
	--p_val=0.25 \
	--pixel=256 \
	--data_dir=/nfs/turbo/coe-soto/mmakar/multiple_shortcut/chexpert_sd/ \
	--exp_dir=/nfs/turbo/coe-soto/mmakar/multiple_shortcut/chexpert_sd/tuning/ \
	--checkpoint_dir=/scratch/mmakar_root/mmakar0/multiple_shortcut/chexpert_sd/ \
	--architecture='pretrained_densenet' \
	--batch_size=16 \
	--num_epochs=1 \
	--training_steps=1 \
	--alpha=0.0 \
	--sigma=10.0 \
	--weighted='False' \
	--conditional_hsic='False' \
	--l2_penalty=0.0 \
	--embedding_dim=-1 \
	--random_seed=0 \
	--cleanup='False' \
	--debugger='True'
```


### Run multiple models 
on GL
```bash
python -m chexpert_support_device.create_submit_slurm \
	--experiment_name 'skew_train' \
	--base_dir '/nfs/turbo/coe-soto/mmakar/multiple_shortcut/chexpert_sd/' \
	--checkpoint_dir '/scratch/mmakar_root/mmakar0/multiple_shortcut/chexpert_sd/' \
	--slurm_save_dir '/home/mmakar/projects/multiple_shortcut/shortcut_hsic/chexpert_slurm_scripts/' \
	--model_to_tune 'unweighted_hsic' \
	--v_mode 'normal' \
	--v_dim 1 \
	--batch_size 16 \
	--submit
```



## Cross validation 
If on GL

```bash
python -m chexpert_support_device.get_predictions \
	--base_dir '/nfs/turbo/coe-soto/mmakar/multiple_shortcut/chexpert_sd/' \
	--seed_list "0,1,2,3,4,5,6,7,8,9" \
	--batch_size 64 \
	--pixel 128 \
	--v_dim 0 \
	--model_name 'unweighted_baseline' \
	--xv_mode 'classic' \
	--eval_group 'valid' \
	--n_jobs 10
```

```bash
srun --cpus-per-task=10 --nodes=1 --ntasks-per-node=1 --account=mmakar0 --partition=standard --mem-per-cpu=5000m --pty /bin/bash

python -m chexpert_support_device.cross_validation \
	--base_dir '/nfs/turbo/coe-soto/mmakar/multiple_shortcut/chexpert_sd/' \
	--experiment_name 'skew_train' \
	--model_to_tune 'unweighted_baseline' \
	--xv_method 'classic' \
	--batch_size 64 \
	--num_workers 10 \
	--t1_error 0.001 \
	--v_dim 0 \
	--v_mode 'normal' \
	--n_permute 100
```
