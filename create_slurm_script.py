import os
import argparse

def create_script(batch_size, epoch_num, pixel, embed_dim, l2_penalty):
    slurm_folder_dir = "/nfs/turbo/coe-rbg/zhengji/age_mask/"
    filename = f"epoch{epoch_num}_l2{l2_penalty}.sh"
    file_dir = os.path.join(slurm_folder_dir, filename)
    log_dir = f"logs/epoch{epoch_num}_l2{l2_penalty}.log"
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
    f.write('#SBATCH --account=precisionhealth_owned1\n')
    f.write('#SBATCH --partition=precisionhealth\n')
    f.write('#SBATCH --output=/nfs/turbo/coe-rbg/zhengji/age_mask/'+log_dir+'\n\n')

    f.write(f'python -m age_model --batch_size={batch_size} --epoch_num={epoch_num} --pixel={pixel} --embedding_dim={embed_dim} --l2_penalty={l2_penalty}')
    f.close()

if __name__ == "__main__":
    epoch_list = [10, 20, 30, 40, 50]
    batch_list = [32]
    pixel_list = [128]
    embed_list = [10]
    l2_list = [1e-2, 1e-3, 1e-4]

    for batch in batch_list:
        for epoch in epoch_list:
            for pixel in pixel_list:
                for embed_dim in embed_list:
                    for l2 in l2_list:
                        params = {'batch_size': batch,
                                'epoch_num': epoch,
                                'pixel':pixel,
                                'embed_dim': embed_dim,
                                'l2_penalty':l2
                                }
                        create_script(**params)