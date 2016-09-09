import time
import argparse
import os
import cPickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from data_loader import load_data 
from parameters import DATASET, TRAINING, HYPERPARAMS

def train(epochs=HYPERPARAMS.epochs, random_state=HYPERPARAMS.random_state, 
          kernel=HYPERPARAMS.kernel, decision_function=HYPERPARAMS.decision_function, train_model=True):

        print "loading dataset " + DATASET.name + "..."
        if train_model:
                data, validation = load_data(validation=True)
        else:
                data, validation, test = load_data(validation=True, test=True)
        
        if train_model:
            # Training phase
            print "building model..."
            model = SVC(random_state=random_state, max_iter=epochs, kernel=kernel, decision_function_shape=decision_function)

            print "start training..."
            print "--"
            print "kernel: {}".format(kernel)
            print "decision function: {} ".format(decision_function)
            print "max epochs: {} ".format(epochs)
            print "--"
            print "Training samples: {}".format(len(data['Y']))
            print "Validation samples: {}".format(len(validation['Y']))
            print "--"
            start_time = time.time()
            model.fit(data['X'], data['Y'])
            training_time = time.time() - start_time
            print "training time = {0:.1f} sec".format(training_time)

            if TRAINING.save_model:
                print "saving model..."
                with open(TRAINING.save_model_path, 'wb') as f:
                        cPickle.dump(model, f)

            print "evaluating..."
            validation_accuracy = evaluate(model, validation['X'], validation['Y'])
            print "  - validation accuracy = {0:.1f}".format(validation_accuracy*100)
            return validation_accuracy
        else:
            # Testing phase : load saved model and evaluate on test dataset
            print "start evaluation..."
            print "loading pretrained model..."
            if os.path.isfile(TRAINING.save_model_path):
                with open(TRAINING.save_model_path, 'rb') as f:
                        model = cPickle.load(f)
            else:
                print "Error: file '{}' not found".format(TRAINING.save_model_path)
                exit()

            print "--"
            print "Validation samples: {}".format(len(validation['Y']))
            print "Test samples: {}".format(len(test['Y']))
            print "--"
            print "evaluating..."
            start_time = time.time()
            validation_accuracy = evaluate(model, validation['X'],  validation['Y'])
            print "  - validation accuracy = {0:.1f}".format(validation_accuracy*100)
            test_accuracy = evaluate(model, test['X'], test['Y'])
            print "  - test accuracy = {0:.1f}".format(test_accuracy*100)
            print "  - evalution time = {0:.1f} sec".format(time.time() - start_time)
            return test_accuracy

def evaluate(model, X, Y):
        predicted_Y = model.predict(X)
        accuracy = accuracy_score(Y, predicted_Y)
        return accuracy

# parse arg to see if we need to launch training now or not yet
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--train", default="no", help="if 'yes', launch training from command line")
parser.add_argument("-e", "--evaluate", default="no", help="if 'yes', launch evaluation on test dataset")
parser.add_argument("-m", "--max_evals", help="Maximum number of evaluations during hyperparameters search")
args = parser.parse_args()
if args.train=="yes" or args.train=="Yes" or args.train=="YES":
        train()
if args.evaluate=="yes" or args.evaluate=="Yes" or args.evaluate=="YES":
        train(train_model=False)