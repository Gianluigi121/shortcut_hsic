import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

main_path = '/nfs/turbo/coe-rbg/zhengji/age_mask/img_data/'
file_name = 'train_patient00170_study5_view1_frontal.png'

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


noise_mask_path = main_path+'noise_01_masks/'+file_name
noise_mask = read_decode_png(noise_mask_path)
noise_mask = tf.cast(noise_mask, tf.float32)

pixel_mask_path = main_path+'noise_pixel_masks/'+file_name
pixel_mask = read_decode_png3(pixel_mask_path)
pixel_mask = tf.cast(pixel_mask, tf.float32)

img_path = '/nfs/turbo/coe-rbg/CheXpert-v1.0/train/patient00170/study5/view1_frontal.jpg'
raw_img = read_decode_jpg(img_path)
raw_img = tf.image.resize(raw_img, [128,128])
new_img = (1-noise_mask)*raw_img + pixel_mask

# save
mask_name = './out.png'
tf.keras.preprocessing.image.save_img(mask_name,
    new_img, scale=False
)
# df.masked_imgs.iloc[i] = mask_name
