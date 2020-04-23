import tensorflow as tf
from tensorflow.keras.models import  Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import sys
import os
print(sys.path)
print(os.getcwd())
sys.path.insert(0, os.getcwd())
from scripts.classifier.load_classifier_data import load_data_test
from scripts.classifier.classifier_model import classifier
import shutil
import time
import pickle
import numpy as np


def testClassifier(expdir, exp, conf, dense_layer_size, phone_map):

    os.makedirs(os.path.join(exp, 'results'), exist_ok=True)

    input_shape = (int(conf['latent_space_dim']),)

    X_test, Y_test = load_data_test(expdir, conf['latent_var'], conf['phone_facs_file'], phone_map, seg_len=int(conf['seg_len']), accuracy_window=int(conf['accuracy_window']))

    model = classifier(input_shape=input_shape, nb_phonemes=int(conf['nb_phonemes']),
                        layers=int(conf['num_dense_layers']), dense_layer_size=dense_layer_size)
    model.load_weights(os.path.join(exp, 'weights-improvement-final.hdf5'))

    # calculate labels
    Y_class_pred = model.predict_classes(X_test)  # given integer of class number
    y_eval = np.array([next(iter(x)) for x in Y_test[0]])
    y_eval = to_categorical(y_eval, int(conf['nb_phonemes']))
    test_results = model.evaluate(x=X_test, y=y_eval)
    print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}%')

    evaluation = {}
    for k in range(int(conf['accuracy_window'])+1):
        y = Y_test[k]  # numpy array
        percentage_correct = []
        for i in range(len(y)):
            percentage_correct.append(Y_class_pred[i] in y[i])
        # evaluation[k] = percentage_correct
        evaluation[k] = sum(percentage_correct)/len(percentage_correct)


    print('Evaluation (with accuracywindow):  ', evaluation)

    with open(os.path.join(exp, "results", "accuracy.txt"), 'w') as fd:
        fd.write("Result for %i layers with layer_size %i" % (int(conf['num_dense_layers']), dense_layer_size))
        for window in evaluation:
            fd.write("\t With accuracy window of size %i: accuracy = %f" % (window, evaluation[window]))
    with open(os.path.join(exp, "results", "acc_win_eval_dict"), "wb") as fp:
        pickle.dump(evaluation, fp)

    return evaluation
