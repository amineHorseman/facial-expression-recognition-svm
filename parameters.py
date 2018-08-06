import os

class Dataset:
    name = 'Fer2013'
    train_folder = 'fer2013_features/Training'
    validation_folder = 'fer2013_features/PublicTest'
    test_folder = 'fer2013_features/PrivateTest'
    trunc_trainset_to = -1
    trunc_validationset_to = -1
    trunc_testset_to = -1

class Hyperparams:
    random_state = 0
    epochs = 10000
    epochs_during_hyperopt = 500
    kernel = 'rbf'  # 'rbf', 'linear', 'poly' or 'sigmoid'
    decision_function = 'ovr'  # 'ovo' for OneVsOne and 'ovr' for OneVsRest'
    features = "landmarks_and_hog" # "landmarks" or "hog" or "landmarks_and_hog"
    gamma = 'auto' # use a float number or 'auto' 
 
class Training:
    save_model = True
    save_model_path = "saved_model.bin"

DATASET = Dataset()
TRAINING = Training()
HYPERPARAMS = Hyperparams()