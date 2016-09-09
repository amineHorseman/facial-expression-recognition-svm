
import os

class Dataset:
    name = 'Fer2013'
    train_folder = 'fer2013_landmarks/train_set'
    validation_folder = 'fer2013_landmarks/validation_set'
    test_folder = 'fer2013_landmarks/test_set'
    trunc_trainset_to = -1
    trunc_validationset_to = -1
    trunc_testset_to = -1

class Hyperparams:
    random_state = 0
    epochs = 1000
    kernel = 'rbf'  # 'rbf', 'linear', 'poly' or 'sigmoid'
    decision_function = 'ovr'  # 'ovo' for OneVsOne and 'ovr' for OneVsRest'
 
class Training:
    save_model = True
    save_model_path = "saved_model.bin"

DATASET = Dataset()
TRAINING = Training()
HYPERPARAMS = Hyperparams()