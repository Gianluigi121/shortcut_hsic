import pandas as pd
import os
import tensorflow as tf
import numpy as np

"""
Step 1: Create zero-one noise images

Directory: main_dir/noise_01_mask/train_patient000001_study1_view1_frontal.jpg
Step 2: Createe pixel noise images

Directory: main_dir/noise_pixel_mask/train_patient000001_study1_view1_frontal.jpg
"""

def read_decode_png(file_path):
	img = tf.io.read_file(file_path)
	img = tf.image.decode_png(img, channels=1)
	return img

def read_decode_png3(file_path):
	img = tf.io.read_file(file_path)
	img = tf.image.decode_png(img, channels=3)
	return img

def read_decode_jpg(file_path):
	img = tf.io.read_file(file_path)
	img = tf.image.decode_jpeg(img, channels=3)
	return img

"""
Given a dataframe, randomly create a 01 mask
"""
def create_noise_01_mask(experiment_directory, df, rng):
    if not os.path.exists(f'{experiment_directory}/noise_01_masks'):
        os.mkdir(f'{experiment_directory}/noise_01_masks')

        # this is a blanck (no noise) image. Need it for code consistency
        no_noise_img = f'{experiment_directory}/noise_01_masks/no_noise.png'
        tf.keras.preprocessing.image.save_img(no_noise_img,
                tf.zeros(shape=[128, 128, 1]), scale=False
            )
    df['noise_01_masks'] = f'{experiment_directory}/noise_01_masks/no_noise.png'

    for i in range(df.shape[0]):
        # STEP 1: Create 0-1 mask
        # create random pixel "flips"
        noise = tf.constant(rng.binomial(n=1, p=0.0005,size=(128, 128)))
        noise = tf.reshape(noise, [128, 128, 1])
        noise = tf.cast(noise, dtype=tf.float32)

        # convolution to make patches
        kernel = tf.ones(shape=[10, 10, 1, 1])
        kernel = tf.cast(kernel, dtype=tf.float32)

        noise_reshaped = tf.reshape(
            noise, [1] + tf.shape(noise).numpy().tolist())

        noise_conv = tf.nn.conv2d(
            tf.cast(noise_reshaped, dtype=tf.float32),
            kernel, [1, 1, 1, 1], padding='SAME'
        )

        noise_conv = tf.squeeze(noise_conv, axis=0)

        # just need 1/0 so threshold
        noise_conv = noise_conv >= 1.0
        noise_conv = tf.cast(noise_conv, dtype=tf.float32)

        # save
        img_name_list = df.Path.iloc[i].split('/')[1:]
        mask_name = '_'.join(img_name_list)
        mask_name = f'{experiment_directory}/noise_01_masks/{mask_name}'
        mask_name = mask_name[:-4]+'.png'
        tf.keras.preprocessing.image.save_img(mask_name,
            tf.reshape(noise_conv, [128, 128, 1]), scale=False
        )
        df.noise_01_masks.iloc[i] = mask_name
    return df

"""
Given a dataframe, randomly create a 
"""
def create_noise_pixel_mask(experiment_directory, df):
    if not os.path.exists(f'{experiment_directory}/noise_pixel_masks'):
        os.mkdir(f'{experiment_directory}/noise_pixel_masks')

        # this is a blank (no noise) image. Need it for code consistency
        no_noise_img = f'{experiment_directory}/noise_pixel_masks/no_noise.png'
        tf.keras.preprocessing.image.save_img(no_noise_img,
                tf.zeros(shape=[128, 128, 3]), scale=False
                )
    df['noise_pixel_masks'] = f'{experiment_directory}/noise_pixel_masks/no_noise.png'

    for i in range(df.shape[0]):
        noise_mask_path = df.noise_01_masks.iloc[i]
        noise_mask = read_decode_png(noise_mask_path)
        
        age = df.Age.iloc[i]
        chan_num = age // 30
        if chan_num >= 2:
            chan_num = 2
        age -= chan_num*30
        color_num = int(float(age / 30)*255)

        new_mask = noise_mask.numpy()
        new_mask[new_mask >= 1.0] = color_num

        pixel_mask = np.zeros(shape=(128, 128, 3))
        pixel_mask[:, :, chan_num] =  new_mask[:, :, 0]

        # save
        img_name = df.noise_01_masks.iloc[i].split('/')[-1]
        mask_name = f'{experiment_directory}/noise_pixel_masks/{img_name}'
        tf.keras.preprocessing.image.save_img(mask_name,
            tf.reshape(pixel_mask, [128, 128, 3]), scale=False
        )
        df.noise_pixel_masks.iloc[i] = mask_name
    return df

def create_masked_imgs(experiment_directory, df):
    if not os.path.exists(f'{experiment_directory}/masked_imgs'):
        os.mkdir(f'{experiment_directory}/masked_imgs')

        # this is a blank (no noise) image. Need it for code consistency
        no_noise_img = f'{experiment_directory}/masked_imgs/no_noise.png'
        tf.keras.preprocessing.image.save_img(no_noise_img,
                tf.zeros(shape=[128, 128, 3]), scale=False
                )
    df['masked_imgs'] = f'{experiment_directory}/masked_imgs/no_noise.png'

    for i in range(df.shape[0]):
        noise_mask_path = df.noise_01_masks.iloc[i]
        noise_mask = read_decode_png(noise_mask_path)
        noise_mask = tf.cast(noise_mask, tf.float32)

        pixel_mask_path = df.noise_pixel_masks.iloc[i]
        pixel_mask = read_decode_png3(pixel_mask_path)
        pixel_mask = tf.cast(pixel_mask, tf.float32)

        img_path = df.Path.iloc[i]
        raw_img = read_decode_jpg('/nfs/turbo/coe-rbg/'+ img_path)
        raw_img = tf.image.resize(raw_img, [128,128])
        new_img = (1-noise_mask)*raw_img + pixel_mask

        # save
        img_name = df.noise_01_masks.iloc[i].split('/')[-1]
        masked_img_name = f'{experiment_directory}/masked_imgs/{img_name}'
        tf.keras.preprocessing.image.save_img(masked_img_name,
            new_img, scale=False
        )
        df.masked_imgs.iloc[i] = masked_img_name
    return df

random_seed = 0
rng = np.random.RandomState(random_seed)
FOLDER_DIR = '/nfs/turbo/coe-rbg/zhengji/age_mask/'
df = pd.read_csv(FOLDER_DIR + 'nofinding_cohort.csv')
print("Create the noise 01 mask")
new_df = create_noise_01_mask('img_data', df, rng)
print("Create the noise pixel mask")
new_df = create_noise_pixel_mask('img_data', new_df)
print("Create the masked imgs")
new_df = create_masked_imgs('img_data', new_df)
new_df.to_csv(FOLDER_DIR + 'final_nofinding_cohort.csv', index=False)
