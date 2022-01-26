# Chexpert experiment with support device and sex as a shortcut

## Step 1: create the data
pick the directory where you want to be saved
Run
```bash
srun --cpus-per-task=1 --ntasks-per-node=1 --account=precisionhealth_owned1 --partition=precisionhealth --time=5:00:00 --tasks-per-node=1 --mem=120gb --pty /bin/bash

source activate env
python -m chexpert_support_device.create_data \
	--save_directory '/nfs/turbo/coe-rbg/mmakar/multiple_shortcut/chexpert' \
	--random_seed 0
```

## Train the model
Run
```bash
srun --cpus-per-task=10 --nodes=1 --ntasks-per-node=1 --mem-per-gpu=2000m  --gres=precisionhealth --account=mmakar0 --partition=precisionhealth_owned1 --pty /bin/bash

source activate env
python -m chexpert_support_device.main
```

