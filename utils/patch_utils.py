import numpy as np


def num_patches(output_dim, patch_size):
    # number of non-overlapping patches
    nb_non_overlapping_patches = 1
    for i in range(0,3):
        nb_non_overlapping_patches = nb_non_overlapping_patches*int(output_dim[i+1]/patch_size[i])
    # dimensions for the patch discriminator
    patch_disc_img_dim = (output_dim[0], patch_size[0], patch_size[1], patch_size[2])
    return int(nb_non_overlapping_patches), patch_disc_img_dim


def get_patches(source_images_for_training, target_images_for_training, generator, batch_counter, patch_size):
    patch_images = 0
    patch_labels = 0
    return patch_images, patch_labels
