## kernel test notes:
in
/data/ddmg/users/mmakar/shared_conda/anaconda3/envs/slabs/lib/python3.8/site-packages/causallearn/utils/cit.py
you modified line 15-22.

you also allowed epsilon to change

## Run multiple models
If on MIT
```bash
python -m waterbirds.create_submit_slurm \
	--base_dir '/data/ddmg/scate/multiple_shortcut/waterbirds/' \
	--checkpoint_dir '/data/ddmg/scate/scratch/waterbirds' \
	--slurm_save_dir '/data/ddmg/users/mmakar/projects/multiple_shortcut/shortcut_hsic/waterbirds_slurm_scripts/' \
	--model_to_tune 'weighted_hsic' \
	--batch_size 64 \
	--v_dim 12 \
	--submit
```

if on GL
```bash
python -m waterbirds.create_submit_slurm \
	--base_dir '/nfs/turbo/coe-soto/mmakar/multiple_shortcut/waterbirds/' \
	--checkpoint_dir '/scratch/mmakar_root/mmakar0/multiple_shortcut/waterbirds' \
	--slurm_save_dir '/home/mmakar/projects/multiple_shortcut/shortcut_hsic/waterbirds_slurm_scripts/' \
	--model_to_tune 'weighted_hsic' \
	--batch_size 64 \
	--v_dim 2 \
	--submit
```
```

```bash
nohup python -m waterbirds.runner \
	--base_dir '/data/ddmg/scate/multiple_shortcut/waterbirds/' \
	--checkpoint_dir '/data/ddmg/scate/scratch' \
	--slurm_save_dir '/data/ddmg/users/mmakar/projects/multiple_shortcut/shortcut_hsic/waterbirds_slurm_scripts/' \
	--model_to_tune 'unweighted_baseline' \
	--batch_size 64 \
	--v_dim 0 \
	--submit &

```

## Cross validation
If running on MIT

```bash
python -m waterbirds.get_predictions \
	--base_dir '/nfs/turbo/coe-soto/mmakar/multiple_shortcut/waterbirds/' \
	--batch_size 64 \
	--pixel 128 \
	--v_dim 2 \
	--model_name 'weighted_hsic' \
	--xv_mode 'two_step' \
	--n_jobs 10 \
	--eval_group 'valid'

```


```bash
python -m waterbirds.cross_validation \
	--base_dir '/data/ddmg/scate/multiple_shortcut/waterbirds/' \
	--model_to_tune 'weighted_hsic' \
	--xv_method 'two_step' \
	--batch_size 64 \
	--num_workers 1 \
	--t1_error 0.05 \
	--v_dim 12 \
	--n_permute 5
```
If on GL
```bash
python -m waterbirds.cross_validation \
	--base_dir '/nfs/turbo/coe-soto/mmakar/multiple_shortcut/waterbirds/' \
	--model_to_tune 'weighted_hsic' \
	--xv_method 'two_step' \
	--batch_size 64 \
	--num_workers 10 \
	--t1_error 0.001 \
	--v_dim 12 \
	--n_permute 100
```


## get predictions
on MIT
```bash
python -m waterbirds.get_predictions \
	--base_dir '/data/ddmg/scate/multiple_shortcut/waterbirds/' \
	--seed_list "0,1,2,3,4,5,6,7,8,9" \
	--batch_size 64 \
	--pixel 128 \
	--v_dim 12 \
	--model_name 'first_step' \
	--xv_mode 'classic'

```

```bash
python -m waterbirds.get_predictions \
	--base_dir '/nfs/turbo/coe-soto/mmakar/multiple_shortcut/waterbirds/' \
	--seed_list "0,1,2,3,4,5,6,7,8,9" \
	--batch_size 64 \
	--pixel 128 \
	--v_dim 1 \
	--model_name 'weighted_hsic' \
	--xv_mode 'two_step'

```

on GL
```bash
python -m waterbirds.get_predictions \
	--base_dir '/nfs/turbo/coe-soto/mmakar/multiple_shortcut/waterbirds/' \
	--seed_list "0,1,2,3,4,5,6,7,8,9" \
	--batch_size 64 \
	--pixel 128 \
	--v_dim 12 \
	--model_name 'weighted_hsic' \
	--xv_mode 'two_step' \
	--eval_group 'valid' \
	--n_jobs 10
```

```bash
python -m waterbirds.get_predictions \
	--base_dir '/nfs/turbo/coe-soto/mmakar/multiple_shortcut/waterbirds/' \
	--seed_list "0,1,2,3,4,5,6,7,8,9" \
	--batch_size 64 \
	--pixel 128 \
	--v_dim 12 \
	--model_name 'weighted_hsic' \
	--xv_mode 'two_step' \
	--get_optimal_only

```


## Do independence testing

on GL
```bash
python -m shared.indep_testing \
	--base_dir '/nfs/turbo/coe-soto/mmakar/multiple_shortcut/waterbirds/' \
	--experiment_name "waterbirds" \
	--seed_list "2" \
	--batch_size 64 \
	--pixel 128 \
	--v_dim 12 \
	--model_name 'first_step' \
	--xv_mode 'classic' \
	--test_pval 0.001 \
	--x_mode 'high_dim'

```



srun --cpus-per-task=12 --ntasks=1 --account=mmakar0 --partition=standard --time=5:00:00 --tasks-per-node=1 --mem=10gb --pty /bin/bash 