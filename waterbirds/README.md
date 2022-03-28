

## Run multiple models
```bash
python -m waterbirds.create_submit_slurm \
	--base_dir '/data/ddmg/scate/multiple_shortcut/waterbirds/' \
	--checkpoint_dir '/data/ddmg/scate/scratch/waterbirds' \
	--slurm_save_dir '/data/ddmg/users/mmakar/projects/multiple_shortcut/shortcut_hsic/waterbirds_slurm_scripts/' \
	--model_to_tune 'weighted_hsic' \
	--batch_size 64 \
	--v_dim 0 \
	--submit
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
python -m waterbirds.cross_validation \
	--base_dir '/data/ddmg/scate/multiple_shortcut/waterbirds/' \
	--model_to_tune 'unweighted_baseline' \
	--xv_method 'two_step' \
	--batch_size 64 \
	--num_workers 1 \
	--t1_error 0.05 \
	--v_dim 0
```

## get predictions
```bash
python -m waterbirds.get_predictions \
	--base_dir '/data/ddmg/scate/multiple_shortcut/waterbirds/' \
	--random_seed 0 \
	--batch_size 64 \
	--pixel 128 \
	--v_dim 0 \
	--model_name 'weighted_baseline' \
	--xv_mode 'classic'

```
