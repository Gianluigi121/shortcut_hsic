# Chexpert experiment with support device and sex as a shortcut

## Step 1: create the data
Suppose you want to save the dataset in the directory `my_data_dir`, then run
```bash
srun --cpus-per-task=1 --ntasks-per-node=1 --account=precisionhealth_owned1 --partition=precisionhealth --time=5:00:00 --tasks-per-node=1 --mem=120gb --pty /bin/bash

source activate env
python -m chexpert_support_device.create_data \
	--save_directory my_data_dir
```

## Train the model

### Run one model
Suppose you want to save the model in `my_model_dir`, and save the checkpoints in `my_checkpoint_dir`
```bash
srun --cpus-per-task=10 --nodes=1 --ntasks-per-node=1 --mem-per-gpu=2000m  --gres=gpu:1 --account=precisionhealth_owned1 --partition=precisionhealth --pty /bin/bash

source activate env

python -m chexpert_support_device.main \
	--skew_train='True' \
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

### Run multiple models
```bash
python -m chexpert_support_device.create_submit_slurm \
	--base_dir '/nfs/turbo/coe-rbg/mmakar/multiple_shortcut/chexpert/' \
	--checkpoint_dir '/scratch/mmakar_root/mmakar0/mmakar/multiple_shortcut/chexpert/' \
	--slurm_save_dir '/home/mmakar/projects/multiple_shortcuts/shortcut_hsic/chexpert_slurm_scripts/' \
	--model_to_tune 'unweighted_baseline' \
	--batch_size 64 \
	--overwrite \
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

