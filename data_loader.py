
from parameters import DATASET
import numpy as np

def load_data(validation=False, test=False):
    
    data_dict = dict()
    validation_dict = dict()
    test_dict = dict()

    if DATASET.name == "Fer2013":
        # load train set
        data_dict['X'] = np.load(DATASET.train_folder + '/landmarks.npy')
        data_dict['Y'] = np.load(DATASET.train_folder + '/labels.npy')
        if DATASET.trunc_trainset_to > 0:
            data_dict['X'] = data_dict['X'][0:DATASET.trunc_trainset_to, :]
            data_dict['Y'] = data_dict['Y'][0:DATASET.trunc_trainset_to]
        if validation:
            # load validation set
            validation_dict['X'] = np.load(DATASET.validation_folder + '/landmarks.npy')
            validation_dict['Y'] = np.load(DATASET.validation_folder + '/labels.npy')
            if DATASET.trunc_validationset_to > 0:
                validation_dict['X'] = validation_dict['X'][0:DATASET.trunc_validationset_to, :]
                validation_dict['Y'] = validation_dict['Y'][0:DATASET.trunc_validationset_to]
        if test:
            # load train set
            test_dict['X'] = np.load(DATASET.test_folder + '/landmarks.npy')
            test_dict['Y'] = np.load(DATASET.test_folder + '/labels.npy')
            np.save(DATASET.test_folder + "/lab.npy", test_dict['Y'])
            if DATASET.trunc_testset_to > 0:
                test_dict['X'] = test_dict['X'][0:DATASET.trunc_testset_to, :]
                test_dict['Y'] = test_dict['Y'][0:DATASET.trunc_testset_to]

        if not validation and not test:
            return data_dict
        elif not test:
            return data_dict, validation_dict
        else: 
            return data_dict, validation_dict, test_dict
    else:
        print "Unknown dataset"
        exit()
