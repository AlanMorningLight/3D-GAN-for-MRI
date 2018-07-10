import numpy as np
from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, BatchNormalization, Deconvolution3D
from keras.layers.merge import concatenate
from keras_contrib.layers.normalization import InstanceNormalization
#  pip install git+https://www.github.com/keras-team/keras-contrib.git


def UNetGenerator(input_dim, output_channels, depth=4, nb_base_filters=32, batch_normalization=True,
                  deconvolution=True, pool_size=(2, 2, 2), activation_name='relu'):

    # depth: the depth of the U-shape structure
    # nb_base_filters: The number of filters that the first layer in the convolution network will have. Following
    # layers will contain a multiple of this number.

    # -------------------
    # build network
    # -------------------

    inputs = Input(input_dim)
    current_layer = inputs
    levels = list()

    # contracting path
    # for each level: Convlayer1 -> Convlayer2(double channels) -> maxpool(halve resolution)
    # for the deepest level: Convlayer1 -> Convlayer2
    for layer_depth in range(depth):
        layer1 = create_convolution_block(input_layer=current_layer, n_filters=nb_base_filters*(2**layer_depth),
                                          batch_normalization=batch_normalization)
        layer2 = create_convolution_block(input_layer=layer1, n_filters=nb_base_filters*(2**layer_depth)*2,
                                          batch_normalization=batch_normalization)
        if layer_depth < depth - 1:
            current_layer = MaxPooling3D(pool_size=pool_size)(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            levels.append([layer1, layer2])

    # expanding path
    # add levels with up-convolution or up-sampling
    # Upconv -> Convlayer1 -> Convlayer2
    for layer_depth in range(depth-2, -1, -1):
        up_convolution = get_up_convolution(pool_size=pool_size, deconvolution=deconvolution,
                                            n_filters=current_layer._keras_shape[4])(current_layer)
        concat = concatenate([up_convolution, levels[layer_depth][1]], axis=4)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[4],
                                                 input_layer=concat, batch_normalization=batch_normalization)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[4],
                                                 input_layer=current_layer,
                                                 batch_normalization=batch_normalization)

    final_convolution = Conv3D(output_channels, (1, 1, 1))(current_layer)
    act = Activation(activation_name)(final_convolution)
    unet_generator = Model(inputs=inputs, outputs=act)
    return unet_generator


def compute_level_output_shape(n_filters, depth, pool_size, image_shape):
    """
    Each level has a particular output shape based on the number of filters used in that level and the depth or number
    of max pooling operations that have been done on the data at that point.
    :param image_shape: shape of the 3d image.
    :param pool_size: the pool_size parameter used in the max pooling operation.
    :param n_filters: Number of filters used by the last node in a given level.
    :param depth: The number of levels down in the U-shaped model a given node is.
    :return: 5D vector of the shape of the output node
    """
    output_image_shape = np.asarray(np.divide(image_shape, np.power(pool_size, depth)), dtype=np.int32).tolist()
    return tuple([None, n_filters] + output_image_shape)


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


def get_up_convolution(n_filters, pool_size, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                       deconvolution=False):
    if deconvolution:
        return Deconvolution3D(filters=n_filters, kernel_size=kernel_size,
                               strides=strides)
    else:
        return UpSampling3D(size=pool_size)

def VNetGenerator(input_dim, output_channels):
    return 0
