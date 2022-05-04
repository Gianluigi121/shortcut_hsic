# if running on mit

python -m chexpert.main \
	--weighted='True' \
	--v_dim=0 \
	--v_mode='dag1' \
	--random_seed=0 \
	--alg_step='None' \
	--data_dir='/data/ddmg/scate/multiple_shortcut/chexpert/' \
	--exp_dir='/data/ddmg/scate/multiple_shortcut/chexpert/tuning/temp/' \
	--checkpoint_dir='/data/ddmg/scate/scratch/tuning/temp' \
	--architecture='pretrained_densenet' \
	--training_steps=5 \
	--pixel=128 \
	--batch_size=64 \
	--alpha=0.0 \
	--sigma=10.0 \
	--conditional_hsic='False' \
	--l2_penalty=0.0 \
	--embedding_dim=-1 \
	--cleanup='True' \
	--gpuid=1

# if running on gl 
python -m chexpert.main \
	--weighted='True' \
	--v_dim=12 \
	--v_mode='dag1' \
	--random_seed=0 \
	--alg_step='None' \
	--data_dir='/nfs/turbo/coe-soto/mmakar/multiple_shortcut/chexpert/' \
	--exp_dir='/nfs/turbo/coe-soto/mmakar/multiple_shortcut/chexpert/tuning/temp/' \
	--checkpoint_dir='/scratch/mmakar_root/mmakar0/multiple_shortcut/chexpert' \
	--architecture='pretrained_densenet' \
	--training_steps=5 \
	--pixel=128 \
	--batch_size=64 \
	--alpha=0.0 \
	--sigma=10.0 \
	--conditional_hsic='False' \
	--l2_penalty=0.0 \
	--embedding_dim=-1 \
	--cleanup='True' \
	--gpuid="cpu"
