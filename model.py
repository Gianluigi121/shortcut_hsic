import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import DenseNet121

def Resnet50(params):
    pixel = params['pixel']
    pretrained_model = ResNet50(input_shape=(pixel, pixel, 3), 
                             include_top=False, 
                             weights='imagenet')

    last_layer = pretrained_model.get_layer('conv5_block3_3_conv')
    print('last layer output shape: ', last_layer.output_shape)
    last_output = last_layer.output 

    x = layers.Flatten()(last_output)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.2)(x)                  
    x = layers.Dense(1, activation='linear')(x)  
    model = Model(pretrained_model.input, x)
    return model

def Densenet121(parmas):
    pixel = params['pixel']
    pretrained_model = DenseNet121(input_shape=(pixel, pixel, 3), 
                             include_top=False, 
                             weights='imagenet')

    last_layer = pretrained_model.get_layer('conv5_block16_2_conv')
    print('last layer output shape: ', last_layer.output_shape)
    last_output = last_layer.output 

    x = layers.Flatten()(last_output)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.2)(x)                  
    x = layers.Dense(1, activation='linear')(x)  
    model = Model(pre_trained_model.input, x)
    return model