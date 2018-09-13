import time
import argparse
import pprint
import numpy as np 
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from train import train
from parameters import HYPERPARAMS

# define the search space
fspace = {
    'decision_function': hp.choice('decision_function', ['ovr', 'ovo']),
    'gamma':  hp.uniform('gamma', 0.001, 0.0001),
}

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--max_evals", required=True, help="Maximum number of evaluations during hyperparameters search")
args = parser.parse_args()
max_evals = int(args.max_evals)
current_eval = 1
train_history = []

def function_to_minimize(hyperparams, gamma='auto', decision_function='ovr'):
    decision_function = hyperparams['decision_function']
    gamma = hyperparams['gamma']
    global current_eval 
    global max_evals
    print( "#################################")
    print( "       Evaluation {} of {}".format(current_eval, max_evals))
    print( "#################################")
    start_time = time.time()
    try:
        accuracy = train(epochs=HYPERPARAMS.epochs_during_hyperopt, decision_function=decision_function, gamma=gamma)
        training_time = int(round(time.time() - start_time))
        current_eval += 1
        train_history.append({'accuracy':accuracy, 'decision_function':decision_function, 'gamma':gamma, 'time':training_time})
    except Exception as e:
        print( "#################################")
        print( "Exception during training: {}".format(str(e)))
        print( "Saving train history in train_history.npy")
        np.save("train_history.npy", train_history)
        exit()
    return {'loss': -accuracy, 'time': training_time, 'status': STATUS_OK}

# lunch the hyperparameters search
trials = Trials()
best_trial = fmin(fn=function_to_minimize, space=fspace, algo=tpe.suggest, max_evals=max_evals, trials=trials)

# get some additional information and print( the best parameters
for trial in trials.trials:
    if trial['misc']['vals']['decision_function'][0] == best_trial['decision_function'] and \
            trial['misc']['vals']['gamma'][0] == best_trial['gamma']:
        best_trial['accuracy'] = -trial['result']['loss'] * 100
        best_trial['time'] = trial['result']['time']
print( "#################################")
print( "      Best parameters found")
print( "#################################")
pprint.pprint(best_trial)
print( "decision_function { 0: ovr, 1: ovo }")
print( "#################################")
