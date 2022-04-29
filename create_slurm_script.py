import os
import argparse

def create_script(batch_size, epoch_num, pixel, embed_dim, sigma, l2_penalty, lr, alpha, weighted):
    slurm_folder_dir = "/nfs/turbo/coe-rbg/zhengji/age_shortcut/"
    # filename = f"epoch{epoch_num}_l2{l2_penalty}.sh"
    
    if weighted:
        filename = f"w_epoch{epoch_num}_l2{l2_penalty}_a{alpha}_s{sigma}.sh"
        log_dir = f"logs_w_alpha/epoch{epoch_num}_l2{l2_penalty}_a{alpha}_s{sigma}.log"
    else:
        filename = f"unw_epoch{epoch_num}_l2{l2_penalty}_a{alpha}_s{sigma}.sh"
        log_dir = f"logs_unw_alpha/epoch{epoch_num}_l2{l2_penalty}_a{alpha}_s{sigma}.log"
    
    file_dir = os.path.join(slurm_folder_dir, filename)
    if not os.path.exists(slurm_folder_dir):
        os.mkdir(slurm_folder_dir)
    
    f = open(file_dir, 'x')
    f.write('#!/bin/bash\n')
    f.write(f'#SBATCH --job-name=age{epoch_num}l2{l2_penalty}\n')
    f.write('#SBATCH --mail-user=zhengji@umich.edu\n')
    f.write('#SBATCH --mail-type=BEGIN,END\n')
    f.write('#SBATCH --cpus-per-task=10\n')
    f.write('#SBATCH --nodes=1\n')
    f.write('#SBATCH --tasks-per-node=1\n')
    f.write('#SBATCH --time=20:00:00\n')
    f.write('#SBATCH --mem-per-cpu=2000mb\n')
    f.write('#SBATCH --gres=gpu:1\n')
    f.write('#SBATCH --account=mmakar0\n')
    f.write('#SBATCH --partition=gpu\n')
    f.write('#SBATCH --output=/nfs/turbo/coe-rbg/zhengji/age_shortcut/'+log_dir+'\n\n')

    if weighted:
        f.write(f'python -m train_weighted --lr={lr} --batch_size={batch_size} --epoch_num={epoch_num} --pixel={pixel} --embedding_dim={embed_dim} --l2_penalty={l2_penalty} --sigma={sigma} --alpha={alpha}')
    else:
        f.write(f'python -m train --lr={lr} --batch_size={batch_size} --epoch_num={epoch_num} --pixel={pixel} --embedding_dim={embed_dim} --l2_penalty={l2_penalty} --sigma={sigma} --alpha={alpha}')
    f.close()

if __name__ == "__main__":
    epoch_list = [20]
    batch_list = [32]
    embed_list = [10]
    sigma_list = [10]
    l2_list = [0, 1e-4]
    lr_list = [1e-4]
    alpha_list = [0, 100, 1000]

    for batch in batch_list:
        for epoch in epoch_list:
                for embed_dim in embed_list:
                    for l2 in l2_list:
                        for lr in lr_list:
                            for alpha in alpha_list:
                                for sigma in sigma_list:
                                    params = {'batch_size': batch,
                                            'epoch_num': epoch,
                                            'pixel': 128,
                                            'embed_dim': embed_dim,
                                            'sigma': sigma,
                                            'l2_penalty':l2,
                                            'lr': lr,
                                            'alpha': alpha,
                                            'weighted': False
                                            }
                                    create_script(**params)
                                    params = {'batch_size': batch,
                                            'epoch_num': epoch,
                                            'pixel': 128,
                                            'embed_dim': embed_dim,
                                            'sigma': sigma,
                                            'l2_penalty':l2,
                                            'lr': lr,
                                            'alpha': alpha,
                                            'weighted': True
                                            }
                                    create_script(**params)

