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
    # generate fake image for even batch_counter
    if batch_counter % 2 == 0:
        image = generator.predict(source_images_for_training)
        patch_labels = np.zeros((image.shape[0], 2), dtype=np.unit8)
        patch_labels[:, 0] = 1  # set the first column to 1 because they are fake image
    else:
        image = target_images_for_training
        patch_labels = np.zeros((image.shape[0], 2), dtype=np.unit8)
        patch_labels[:, 1] = 1  # set the first column to 1 because they are real image

    input_dim = source_images_for_training.shape
    z, y, x = input_dim[1:4]
    pz, py, px = patch_size[:]

    list_z_idx = [(i * pz, (i + 1) * pz) for i in range(int(z / pz))]
    list_y_idx = [(i * py, (i + 1) * py) for i in range(int(y / py))]
    list_x_idx = [(i * px, (i + 1) * px) for i in range(int(x / px))]

    patch_images = []
    for z_idx in list_z_idx:
        for y_idx in list_y_idx:
            for x_idx in list_x_idx:
                patch = image[:, z_idx[0]:z_idx[1], y_idx[0]:y_idx[1], x_idx[0]:x_idx[1], :]
                patch_images.append(np.asarray(patch, dtype=np.float32))
    return patch_images, patch_labels

