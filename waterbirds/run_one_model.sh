python -m waterbirds.main \
	--weighted='True' \
	--v_dim=0 \
	--clean_back='True' \
	--random_seed=0 \
	--alg_step='None' \
	--data_dir='/data/ddmg/scate/multiple_shortcut/waterbirds/' \
	--exp_dir='/data/ddmg/scate/multiple_shortcut/waterbirds/tuning/temp/' \
	--checkpoint_dir='/data/ddmg/scate/scratch/tuning/temp' \
	--architecture='pretrained_resnet' \
	--training_steps=10 \
	--pixel=128 \
	--batch_size=64 \
	--alpha=0.0 \
	--sigma=10.0 \
	--conditional_hsic='False' \
	--l2_penalty=0.0 \
	--embedding_dim=-1 \
	--cleanup='True'\
	--gpuid=0
