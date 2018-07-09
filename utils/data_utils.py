import os
import random


# make directory: if the directory exists, remove it and recreate the directory
def make_dirs(dir_name):
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    else:
        os.system('rm -rf ' + dir_name)
        os.makedirs(dir_name)


# split data into training, testing, and validation sets in seperate folders
def split_dataset(all_data_dir, testing_data_pct, validation_pct):
    training_dir = all_data_dir + 'training/'
    testing_dir = all_data_dir + 'testing/'
    validation_dir = all_data_dir + 'validation/'
    make_dirs(training_dir)
    make_dirs(testing_dir)
    make_dirs(validation_dir)

    folder_list = os.listdir(all_data_dir)
    folders = []
    for folder in folder_list:
        if folder.endswith('RAW'):
            folders.append(folder)
    random.shuffle(folders)
    nb_testing_data = int(len(folders) * testing_data_pct)
    nb_validation_data = int(len(folders) * validation_pct)
    for i in range(0, nb_testing_data):
        os.system('ln -s ' + all_data_dir + folders[i] + ' ' + testing_dir)
    for i in range(nb_testing_data, nb_testing_data+nb_validation_data):
        os.system('ln -s ' + all_data_dir + folders[i] + ' ' + validation_dir)
    for i in range(nb_testing_data+nb_validation_data, len(folders)):
        os.system('ln -s ' + all_data_dir + folders[i] + ' ' + training_dir)


if __name__ == '__main__':
    split_dataset('/media/jiang/Elements/liujiangdata/DL_Glioma_500P_2017/NIFTII/', 0.15, 0.15)