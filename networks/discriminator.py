# import numpy as np
from keras.engine import Input, Model
from keras.layers import Conv3D, Flatten, Dense, BatchNormalization
# from keras.layers.merge import concatenate
# from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers.advanced_activations import LeakyReLU

def PatchGanDiscriminator(output_dim, patch_size, nb_patches, padding='same', strides=(2,2,2), kernel_size=(3,3,3),
                          mini_batch_discrimination=False):
    # output_dim = [samples, z, y, x, channels]
    # patch_size = [z,y,x]
    inputs = Input(shape=[patch_size[0], patch_size[1], patch_size[2], output_dim[4]])
    filter_list = [64, 128, 256, 512, 512, 512]

    # Layer1 without Batch Normalization
    disc_out = Conv3D(filters=filter_list[0], kernel_size=kernel_size, padding=padding, strides=strides)(inputs)
    disc_out = LeakyReLU(alpha=0.2)(disc_out)

    # build the rest Layers
    # Conv -> BN -> LeakyReLU
    for i, filter_size in enumerate(filter_list[1:]):
        name = 'disc_conv_{}'.format(i+1)
        disc_out = Conv3D(name=name, filters=filter_list[i+1], kernel_size=kernel_size, padding=padding, strides=strides)(disc_out)
        disc_out = BatchNormalization(name=name+'_bn', axis=4)(disc_out)
        disc_out = LeakyReLU(alpha=0.2)(disc_out)
    if mini_batch_discrimination:
    # -----------------------------------------------
    # build patch GAN with mini-batch discrimination
    # -----------------------------------------------
        patch_GAN_discriminator = generate_patch_gan_loss(last_disc_conv_layer=disc_out,
                                                      patch_dim=patch_size,
                                                      input_layer=inputs,
                                                      nb_patches=nb_patches)
    else:
        x_flat = Flatten()(disc_out)
        x = Dense(2, activation='softmax',name="disc_dense")(x_flat)
        patch_GAN_discriminator = Model(input=inputs, output=x, name="patch_gan")
    return patch_GAN_discriminator


# to be implemented:
def generate_patch_gan_loss(last_disc_conv_layer, patch_size, input_layer, nb_patches):
    list_input = [Input(shape=[patch_size[0], patch_size[1], patch_size[2], 1])]
    x_flat = Flatten()(last_disc_conv_layer)

    return 0