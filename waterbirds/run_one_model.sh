python -m waterbirds.main \
	--weighted='True' \
	--v_dim=0 \
	--clean_back='True' \
	--random_seed=20 \
	--alg_step='None' \
	--data_dir='/data/ddmg/scate/multiple_shortcut/waterbirds/' \
	--exp_dir='/data/ddmg/scate/multiple_shortcut/waterbirds/tuning/temp/' \
	--checkpoint_dir='/data/ddmg/scate/scratch/tuning/temp' \
	--architecture='pretrained_resnet' \
	--training_steps=5 \
	--pixel=128 \
	--batch_size=64 \
	--alpha=0.0 \
	--sigma=10.0 \
	--conditional_hsic='False' \
	--l2_penalty=0.0 \
	--embedding_dim=-1 \
	--cleanup='True' \
	--gpuid=4


python -m waterbirds.main \
	--weighted='True' \
	--v_dim=12 \
	--clean_back='True' \
	--random_seed=0 \
	--alg_step='first' \
	--data_dir='/nfs/turbo/coe-soto/mmakar/multiple_shortcut/waterbirds/' \
	--exp_dir='/nfs/turbo/coe-soto/mmakar/multiple_shortcut/waterbirds/tuning/temp/' \
	--checkpoint_dir='/scratch/mmakar_root/mmakar0/multiple_shortcut/waterbirds/tuning/temp' \
	--architecture='pretrained_resnet' \
	--training_steps=5 \
	--pixel=128 \
	--batch_size=64 \
	--alpha=0.0 \
	--sigma=10.0 \
	--conditional_hsic='False' \
	--l2_penalty=0.0 \
	--embedding_dim=-1 \
	--cleanup='True' \
	--gpuid=$CUDA_VISIBLE_DEVICES




