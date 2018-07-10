import os
import numpy as np
import SimpleITK as sitk


def read_image(path):
    img = sitk.GetArrayFromImage(sitk.ReadImage(path))
    return img[np.newaxis, :, :, :, np.newaxis]


def test_data_generator(dir, data_type, modality, input_dim):
    # data_type：training, testing, or validation

    # inverse_table = {v:k for k,v in modality_table.items()}
    inverse_table = {0: 'BRAVO.nii', 1: 'corT2.nii', 2: 'corGRE.nii', 3: 'corDWI.nii', 4: 'corDWI2.nii'}
    data_dir = dir + data_type + '/'
    files = os.listdir(data_dir)

    batch_size = input_dim[0]
    input_channels = input_dim[4]
    count = 0
    # read batch_size of data
    for i in range(0, batch_size):
        single_img = read_image(data_dir + files[count] + '/' + inverse_table[modality[0]])
        for j in range(1, input_channels):
            img = read_image(data_dir + files[count] + '/' + inverse_table[modality[j]])
            single_img = np.concatenate((single_img, img), axis=4)
        if i == 0:
            source_img = single_img
        else:
            source_img = np.concatenate((source_img, single_img), axis=0)
    target_img = read_image(data_dir + files[count] + '/' + 'corBRAVOGd.nii')
    return source_img, target_img


def data_generator(dir, data_type, modality, batch_size):
    # data_type：training, testing, or validation

    # inverse_table = {v:k for k,v in modality_table.items()}
    inverse_table = {0: 'BRAVO.nii', 1: 'corT2.nii', 2: 'corGRE.nii', 3: 'corDWI.nii', 4: 'corDWI2.nii'}
    data_dir = dir + data_type + '/'
    files = os.listdir(data_dir)

    input_channels = len(modality)
    count = 0
    while True:
        # read batch_size of data
        for i in range(0, batch_size):
            single_img = read_image(data_dir + files[count] + '/' + inverse_table[modality[0]])
            for j in range(1, input_channels):
                img = read_image(data_dir + files[count] + '/' + inverse_table[modality[j]])
                single_img = np.concatenate((single_img, img), axis=4)  # add up channel
            if i == 0:
                source_img = single_img
            else:
                source_img = np.concatenate((source_img, single_img), axis=0)  # add up data
            count = (count+1)%len(files)
        target_img = read_image(data_dir + files[count] + '/' + 'corBRAVOGd.nii')
        yield source_img, target_img


if __name__ == '__main__':
    s, t = test_data_generator('/media/jiang/Elements/liujiangdata/DL_Glioma_500P_2017/NIFTII/', 'training', [0, 1, 2],
                               1)
