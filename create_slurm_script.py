import os
import argparse

def create_script(batch_size, epoch_num, pixel):
    slurm_folder_dir = "/nfs/turbo/coe-rbg/zhengji/age/scripts/"
    filename = f"epoch{epoch_num}.sh"
    file_dir = os.path.join(slurm_folder_dir, filename)
    log_dir = f"logs/epoch{epoch_num}.log"
    if not os.path.exists(slurm_folder_dir):
        os.mkdir(slurm_folder_dir)
    
    f = open(file_dir, 'x')
    f.write('#!/bin/bash\n')
    f.write(f'#SBATCH --job-name=age{epoch_num}\n')
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
    f.write('#SBATCH --output=/nfs/turbo/coe-rbg/zhengji/age/'+log_dir+'\n\n')


    f.write(f'python -m age.age_model --batch_size={batch_size} --epoch_num={epoch_num} --pixel={pixel}')
    f.close()

if __name__ == "__main__":
    epoch_list = [1, 10, 20]
    batch_list = [16]
    pixel_list = [128]

    for batch in batch_list:
        for epoch in epoch_list:
            for pixel in pixel_list:
                params = {'batch_size': batch,
                          'epoch_num': epoch,
                          'pixel':pixel
                         }
                create_script(**params)