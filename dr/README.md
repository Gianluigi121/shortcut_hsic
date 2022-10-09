## Downloading the data
`wget -i url`
need to install p7zip through conda
`7z e "train.zip.*"`
# need opencv through anaconda
# run the data preprocessing.py


## Run multiple models
on MIT
```bash
python -m dr.create_submit_slurm \
	--base_dir '/data/ddmg/scate/multiple_shortcut/dr/' \
	--checkpoint_dir '/data/ddmg/scate/scratch/dr' \
	--slurm_save_dir '/data/ddmg/users/mmakar/projects/multiple_shortcut/shortcut_hsic/dr_slurm_scripts/' \
	--model_to_tune 'unweighted_baseline' \
	--batch_size 64 \
	--py1y0 0.9 \
	--submit
```
on GL
```bash
python -m dr.create_submit_slurm \
	--base_dir '/nfs/turbo/coe-soto/mmakar/multiple_shortcut/dr/' \
	--checkpoint_dir '/scratch/mmakar_root/mmakar0/multiple_shortcut/dr' \
	--slurm_save_dir '/home/mmakar/projects/multiple_shortcut/shortcut_hsic/dr_slurm_scripts/' \
	--model_to_tune 'weighted_hsic' \
	--batch_size 64 \
	--py1y0 0.9 \
	--submit
```


```bash
nohup python -m dr.runner \
	--base_dir '/data/ddmg/scate/multiple_shortcut/dr/' \
	--checkpoint_dir '/data/ddmg/scate/scratch' \
	--slurm_save_dir '/data/ddmg/users/mmakar/projects/multiple_shortcut/shortcut_hsic/dr_slurm_scripts/' \
	--model_to_tune 'unweighted_baseline' \
	--batch_size 64 \
	--py1y0 0.9 \
	--submit &

```

## Cross validation
If running on MIT

```bash
python -m dr.cross_validation \
	--base_dir '/data/ddmg/scate/multiple_shortcut/dr/' \
	--model_to_tune 'unweighted_baseline' \
	--xv_method 'classic' \
	--batch_size 64 \
	--num_workers 1 \
	--t1_error 0.05 \
	--py1y0 0.9 \
	--n_permute 5
```

If running on GL
```bash
python -m dr.cross_validation \
	--base_dir '/nfs/turbo/coe-soto/mmakar/multiple_shortcut/dr/' \
	--model_to_tune 'weighted_hsic' \
	--xv_method 'two_step' \
	--batch_size 64 \
	--num_workers 11 \
	--t1_error 0.001 \
	--py1y0 0.9 \
	--n_permute 100
```



## get predictions
```bash
python -m dr.get_predictions \
	--base_dir '/data/ddmg/scate/multiple_shortcut/dr/' \
	--seed_list "0,1,2,3,4,5,6,7,8,9" \
	--batch_size 64 \
	--pixel 299 \
	--py1y0 0.9 \
	--model_name 'weighted_hsic' \
	--xv_mode 'two_step' \
	--n_jobs 10 \
	--get_optimal_only
```
on GL

```bash
python -m dr.get_predictions \
	--base_dir '/nfs/turbo/coe-soto/mmakar/multiple_shortcut/dr/' \
	--batch_size 64 \
	--pixel 299 \
	--py1y0 0.9 \
	--model_name 'weighted_hsic' \
	--xv_mode 'two_step' \
	--n_jobs 10 \
	--eval_group 'valid'

```

```bash
python -m dr.get_predictions \
	--base_dir '/nfs/turbo/coe-soto/mmakar/multiple_shortcut/dr/' \
	--seed_list "0,1,2,3,4,5,6,7,8,9" \
	--batch_size 64 \
	--pixel 299 \
	--py1y0 0.9 \
	--model_name 'weighted_hsic' \
	--xv_mode 'two_step' \
	--n_jobs 20 \
	--eval_group 'test' \
	--get_optimal_only

```



