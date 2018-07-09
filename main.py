from utils import patch_utils
from utils import data_generator
from utils import logger
from networks.generator import UNetGenerator
from networks.discriminator import PatchGANDiscriminator
from networks.DCGAN import DCGAN
from keras.optimizers import Adam
from keras.utils import generic_utils as keras_generic_utils
import os
import time
import numpy as np

# import argparse

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("modality", help="modality used to train the generator")
    # args = parser.parse_args()
    # print(args.modality)

    # ----------------------
    # set up the params
    # ----------------------

    # modality to train the generator
    modality = {'BRAVO', 'BRAVOGd', 'T2', 'GRE', 'DWI', 'DWI2'}
    # path to data
    path = './data/DL_Glioma_80P_2017'

    # image dims
    input_channels = len(modality)
    output_channels = 1
    input_img_dim = [512, 512, 256]
    output_img_dim = [512, 512, 256]
    patch_size = [64,64,64]
    batch_size = 1

    # input & output dims (for channels_last convention)
    input_dim = [batch_size, input_img_dim[0], input_img_dim[1], input_img_dim[2], input_channels]
    output_dim = [batch_size, output_img_dim[0], output_img_dim[1], output_img_dim[2], output_channels]

    # training params
    lr = 1e-4
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1e-8

    # ----------------------
    # build the network
    # ----------------------

    # optimizers
    opt_generator = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
    opt_discriminator = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
    opt_DCGAN = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)

    # UNetGenerator
    generator = UNetGenerator(input_dim, output_channels)
    generator.summary()
    generator.compile(loss='mae', optimizer=opt_generator) # L1 loss for image generation

    # Patch GAN Discriminator
    nb_patches, patch_gan_dim = patch_utils.num_patches(output_dim, patch_size)
    discriminator = PatchGANDiscriminator(output_img_dim, patch_gan_dim, nb_patches)
    discriminator.summary()
    discriminator.trainable = False

    # DCGAN
    dc_gan = DCGAN(generator, discriminator, input_dim, patch_size)
    dc_gan.summary()

    # loss
    loss = ['mae', 'binary_crossentropy']
    loss_weights = [1e2, 1]
    dc_gan.compile(loss=loss, loss_weights=loss_weights, optimizer=opt_DCGAN)

    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer=opt_discriminator)

    # ----------------------
    # train
    # ----------------------

    nb_epoch = 100
    n_images_per_epoch = 400
    print("Start training..")
    for epoch in range(0, nb_epoch):
        print('Epoch {}'.format(epoch))
        batch_counter = 1
        start = time.time()
        progress_bar = keras_generic_utils.Progbar(n_images_per_epoch)
        # data generator
        training_generator = data_generator(path, 'training', modality, input_img_dim, batch_size)
        validation_generator = data_generator(path, 'validation', modality, input_img_dim, batch_size)

        for batch_i in range(0, n_images_per_epoch, batch_size):
            # load a batch of images for training and validation
            source_images_for_training, target_images_for_training = next(training_generator)
            source_images_for_validation, target_images_for_validation = next(validation_generator)

            # generate image patches to train the discriminator
            # patch_images is image patches
            # patch_labels is batch_size x 2 vector for each patch (fake or not)
            patch_images, patch_labels = patch_utils.get_patches(source_images_for_training,
                                                                 target_images_for_training,
                                                                 generator, batch_counter, patch_size)
            # train the discriminator
            discriminator_loss = discriminator.train_on_batch(patch_images, patch_labels)
            # freeze the discriminator
            discriminator.trainable = False

            # train the GAN
            gan_labels = np.zeros(target_images_for_training.shape[0], 2)
            gan_labels[:, 1] = 1  # they are all fake images since they come fot the generator
            gan_loss = dc_gan.train_on_batch(source_images_for_training, [target_images_for_training, gan_labels])

            # unfreeze the discriminator
            discriminator.trainable = True

            batch_counter += 1

            # print losses
            D_log_loss = discriminator_loss
            gan_total_loss = gan_loss[0].tolist()
            gan_total_loss = min(gan_total_loss, 1000000)
            gan_mae = gan_loss[1].tolist()
            gan_mae = min(gan_mae, 1000000)
            gan_log_loss = gan_loss[2].tolist()
            gan_log_loss = min(gan_log_loss, 1000000)

            progress_bar.add(batch_size, values=[("Dis logloss", D_log_loss),
                                            ("GAN total", gan_total_loss),
                                            ("GAN L1 (mae)", gan_mae),
                                            ("GAN logloss", gan_log_loss)])
        print("")
        print('Epoch %s/%s, Time: %s' % (epoch + 1, nb_epoch, time.time() - start))
        # save weights and images on every 2nd epoch
        if epoch % 2 == 0:
            gan_weights_path = os.path.join('./weights/gen_weights_epoch_%s.h5' % epoch)
            generator.save_weights(gan_weights_path, overwrite=True)

            disc_weights_path = os.path.join('./weights/disc_weights_epoch_%s.h5' % epoch)
            discriminator.save_weights(disc_weights_path, overwrite=True)

            DCGAN_weights_path = os.path.join('./weights/DCGAN_weights_epoch_%s.h5' % epoch)
            dc_gan.save_weights(DCGAN_weights_path, overwrite=True)

        # save images for visualization every 10th batch
        if batch_counter % 10 == 0:
                logger.plot_generated_batch(source_images_for_training, target_images_for_training, generator, epoch,
                                            'training', 'png')
                logger.plot_generated_batch(source_images_for_validation, target_images_for_validation, generator, epoch,
                                            'validation', 'png')
















