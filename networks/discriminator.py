import numpy as np
from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, BatchNormalization, Deconvolution3D
from keras.layers.merge import concatenate
from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers.advanced_activations import LeakyReLU

def PatchGanDiscriminator(output_dim, patch_size, nb_patches, padding='same', strides=(1,1,1), kernel_size=(3,3,3)):
    """
    # -------------------------------
    # DISCRIMINATOR
    # C64-C128-C256-C512-C512-C512 (for 256x256)
    # otherwise, it scales from 64
    # 1 layer block = Conv - BN - LeakyRelu
    # -------------------------------
    """

    inputs = Input(shape=patch_size)
    filter_list = [64, 128, 256, 512, 512, 512]
    nb_layers = len(filter_list)

    # Layer1 without Batch Normalization
    disc_out = Conv3D(filters=filter_list[0], kernel_size=kernel_size, padding=padding, strides=strides)(inputs)
    disc_out = LeakyReLU(disc_out)

    # Build the rest Layers
    for


    return

def create_convolution_block(input_layer, n_filters, batch_normalization=False, kernel_size=(3, 3, 3), activation='relu',
                             padding='same', strides=(1, 1, 1), instance_normalization=False):
    # 3DConv + Normalization + Activation
    # Instance Normalization is said to perform better than Batch Normalization

    layer = Conv3D(n_filters, kernel_size, padding=padding, strides=strides)(input_layer)
    if batch_normalization:
        layer = BatchNormalization(axis=4)(layer)  # channel_last convention
    elif instance_normalization:
        layer = InstanceNormalization(axis=4)(layer)
    return Activation(activation)(layer)