# Chexpert experiment with support device and sex as a shortcut

## Step 1: create the data
Suppose you want to save the dataset in the directory `my_data_dir`. 
If on armis 
```bash
srun --cpus-per-task=1 --ntasks-per-node=1 --account=precisionhealth_owned1 --partition=precisionhealth --time=5:00:00 --tasks-per-node=1 --mem=50gb --pty /bin/bash

source activate env
python -m chexpert_support_device.create_data \
	--save_directory my_data_dir
```

if on GL
```bash
srun --account=engin1 --partition=standard --time=3:00:00 --tasks-per-node=1 --mem=1gb --pty /bin/bash

source activate env
python -m chexpert_support_device.create_data \
	--save_directory /nfs/turbo/coe-soto/mmakar/multiple_shortcut/chexpert_sd/
```
For me it is: 
```bash
srun --account=engin1 --partition=standard --time=1:00:00 --tasks-per-node=1 --mem=1gb --pty /bin/bash

source activate env
python -m chexpert_support_device.create_data \
	--save_directory /nfs/turbo/coe-soto/mmakar/multiple_shortcut/chexpert_sd/
```


## Train the model

### Run one model
Suppose you want to save the model in `my_model_dir`, and save the checkpoints in `my_checkpoint_dir`
```bash
srun --cpus-per-task=10 --nodes=1 --ntasks-per-node=1 --mem-per-gpu=2000m  --gres=gpu:1 --account=precisionhealth_owned1 --partition=precisionhealth --pty /bin/bash

source activate env

python -m chexpert_support_device.main \
	--skew_train='True' \
	--v_dim=6 \
	--p_tr=0.7 \
	--p_val=0.25 \
	--pixel=128 \
	--data_dir=my_data_dir \
	--exp_dir=my_model_dir \
	--checkpoint_dir=my_checkpoint_dir \
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
	--debugger='True'
```
On GL, for me: 

```bash
srun --mem-per-gpu=90000m  --gres=gpu:1 --account=engin1 --partition=gpu --pty /bin/bash

source activate env

python -m chexpert_support_device.main \
	--skew_train='True' \
	--v_dim=5 \
	--p_tr=0.7 \
	--p_val=0.25 \
	--pixel=128 \
	--data_dir=/nfs/turbo/coe-soto/mmakar/multiple_shortcut/chexpert_sd/ \
	--exp_dir=/nfs/turbo/coe-soto/mmakar/multiple_shortcut/chexpert_sd/tuning/ \
	--checkpoint_dir=/scratch/mmakar_root/mmakar0/multiple_shortcut/chexpert_sd/ \
	--architecture='pretrained_densenet' \
	--batch_size=64 \
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
```bash
python -m chexpert_support_device.create_submit_slurm \
	--base_dir '/nfs/turbo/coe-rbg/mmakar/multiple_shortcut/chexpert/' \
	--checkpoint_dir '/scratch/mmakar_root/mmakar0/mmakar/multiple_shortcut/chexpert/' \
	--slurm_save_dir '/home/mmakar/projects/multiple_shortcuts/shortcut_hsic/chexpert_slurm_scripts/' \
	--model_to_tune 'weighted_hsic' \
	--batch_size 64 \
	--submit
```

If running on MIT: 
```bash
python -m chexpert_support_device.create_submit_slurm \
	--experiment_name 'skew_train' \
	--base_dir '/data/ddmg/scate/multiple_shortcut/chexpert/' \
	--checkpoint_dir '/data/ddmg/scate/scratch' \
	--slurm_save_dir '/data/ddmg/users/mmakar/projects/multiple_shortcut/shortcut_hsic/chexpert_slurm_scripts/' \
	--model_to_tune 'unweighted_baseline' \
	--batch_size 64 \
	--v_mode 'normal' \
	--v_dim 0 \
	--submit 
```

If running on MIT (ahoy): 
```bash
python -m chexpert_support_device.runner \
	--experiment_name 'skew_train' \
	--base_dir '/data/ddmg/scate/multiple_shortcut/chexpert/' \
	--checkpoint_dir '/data/ddmg/scate/scratch' \
	--slurm_save_dir '/data/ddmg/users/mmakar/projects/multiple_shortcut/shortcut_hsic/chexpert_slurm_scripts/' \
	--model_to_tune 'weighted_baseline' \
	--batch_size 64 \
	--v_mode 'corry' \
	--v_dim 100 \
	--submit 
```




## Tensorboard launch

Suppose you saved your checkpoints on `my_checkpoint_dir`. On Armis2, run:
```bash
module load tensorflow/2.4.1
tensorboard --logdir=my_checkpoint_dir --port 8088 --host localhost
```
On your local machine, run
```bash
  ssh -NL 8088:localhost:8088 your_user_name@armis2.arc-ts.umich.edu
 ```

*Note*: make sure that the `save_checkpoints_steps` variable in the `train` function in `shared/train.py` is small if you're going to look at tensorboard.


## Cross validation
Run
```bash
python -m chexpert_support_device.cross_validation \
	--base_dir '/nfs/turbo/coe-rbg/mmakar/multiple_shortcut/chexpert/' \
	--experiment_name 'skew_train' \
	--model_to_tune 'weighted_hsic' \
	--xv_method 'two_step' \
	--batch_size 64 \
	--num_workers 1 \
	--t1_error 0.05 \
	--v_mode 'normal'
```

If running on MIT

```bash
python -m chexpert_support_device.cross_validation \
	--base_dir '/data/ddmg/scate/multiple_shortcut/chexpert/' \
	--experiment_name 'skew_train' \
	--model_to_tune 'weighted_baseline' \
	--xv_method 'classic' \
	--batch_size 64 \
	--num_workers 1 \
	--t1_error 0.05 \
	--v_mode 'normal' \
	--v_dim 0
```

## get predictions
```bash
python -m chexpert_support_device.get_predictions \
	--base_dir '/nfs/turbo/coe-rbg/mmakar/multiple_shortcut/chexpert/' \
	--experiment_name 'skew_train' \
	--random_seed 0 \
	--batch_size 64  \
	--pixel 128 \
	--model_name 'weighted_hsic' \
	--xv_mode 'two_step' \
	--fixed_joint \
	--aux_joint_skew 0.9 \
	--get_optimal_only
```
