import SimpleITK as sitk
import matplotlib.pyplot as plt
from utils.data_utils import make_dirs


def plot_generated_batch(source_images_for_training, target_images_for_training, generator, epoch,
                                            image_type, writing_format, nb_channels=1):
    # image_type: training, validation
    # writing_format: format for saving the images, e.g. nii, png
    # nb_channels： number of input channels to save, default is 1
    # for now, it only saves the first image in a batch of images

    dir = './logger/epoch_{}/'.format(epoch)
    source_dir = dir + image_type + '/source/'
    target_dir = dir + image_type + '/target/'
    generated_dir = dir + image_type + '/generated/'
    make_dirs(source_dir)
    make_dirs(target_dir)
    make_dirs(generated_dir)

    img_dim = source_images_for_training.shape  # channels_last:[batch_size, x, y, z, channels]
    nb_channels = min(nb_channels, img_dim[4]) # nb_channels cannot be more than the actual channels
    generated_image = generator.predict(source_images_for_training[0, :, :, :, :])

    if writing_format == 'nii':
        # write source images for
        for i in range(0, nb_channels):
            source_img = sitk.GetImageFromArray(source_images_for_training[0,:,:,:,i])
            source_file_name = source_dir + 'source_channel_{}.nii'.format(i)
            sitk.WriteImage(source_img, source_file_name)
        # write target images
        target_img = sitk.GetImageFromArray(target_images_for_training[0,:,:,:,0])
        target_file_name = target_dir + 'target.nii'
        sitk.WriteImage(target_img, target_file_name)
        # write generated images
        generated_img = sitk.GetImageFromArray(generated_image)
        generated_file_name = generated_dir + 'generated.nii'
        sitk.WriteImage(generated_img, generated_file_name)
    else:
        nb_images = img_dim[3]
        # write source images
        for i in range(0, nb_channels):
            for j in range(0, nb_images):
                img = source_images_for_training[0, :, :, j, i]*255
                make_dirs(source_dir+'channel_{}/'.format(i))
                plt.imsave(source_dir+'channel_{}/'.format(i)+'img_{}.png'.format(j), img, cmap='gray')
        # write target images
        for j in range(0, nb_images):
            img = target_images_for_training[0, :, :, j, 0] * 255
            plt.imsave(target_dir + 'img_{}.png'.format(j), img, cmap='gray')
        # write generated images
        for j in range(0, nb_images):
            img = generated_image[0, :, :, j, 0] * 255
            plt.imsave(generated_dir + 'img_{}.png'.format(j), img, cmap='gray')

    

