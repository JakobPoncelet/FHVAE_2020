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
from collections import defaultdict

def testClassifier(expdir, exp, conf, trainconf, fac, valid_classes, dense_layer_size, plot_SNR):

    os.makedirs(os.path.join(exp, 'results'), exist_ok=True)

    input_shape = (int(conf['latent_space_dim']),)

    if fac == 'phones':
        phone_map = trainconf['talab_vals']['phones']
    else:
        phone_map = None
        conf['accuracy_window'] = 0  # no need for accuracy window when predicting labs instead of talabs

    print('retrieving test data')
    X_test, Y_test, all_segs, seg_cnt = load_data_test(expdir, conf['latent_var'], conf['facdir'], fac, valid_classes, phone_map, seg_len=int(conf['seg_len']), accuracy_window=int(conf['accuracy_window']))

    print('retrieved test data')

    model = classifier(input_shape=input_shape, num_classes=int(conf['num_classes']),
                        layers=int(conf['num_dense_layers']), dense_layer_size=dense_layer_size)
    model.load_weights(os.path.join(exp, 'weights-improvement-final.hdf5'))

    # calculate labels
    Y_class_pred = model.predict_classes(X_test)  # given integer of class number

    y_eval = np.array([next(iter(x)) for x in Y_test[0]])
    y_eval = to_categorical(y_eval, int(conf['num_classes']))
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
    with open(os.path.join(exp, "results", "acc_win_eval_dict.pkl"), "wb") as fp:
        pickle.dump(evaluation, fp)

    # seg_cnt contains number of successive elements that come from the same sequence (test set is not shuffled)
    # seg_cnt = list of tuples of (num_segs in seq, lab of seq)
    # --> see what label is predicted most by the segments in the sequence
    average_evaluation = 0.0
    if seg_cnt is not None:
        #y = Y_test[0]
        y = Y_class_pred
        percentage_correct = []
        cnt = 0

        def get_most_common_labs(lst):
            # return list of labs that are predicted the most (if there is a tie, return all winners)
            max_cnt, max_labs = 0, []
            uniq_lst = list(set(lst))
            for lab in uniq_lst:
                cnt = lst.count(lab)
                if cnt > max_cnt:
                    max_cnt, max_labs = cnt, [lab]
                elif cnt == max_cnt:
                    max_labs.append(lab)
            return max_labs

        for el in seg_cnt:
            is_noisy, nsegs, lab = el

            # only use clean data, otherwise leave these lines out
            if is_noisy:
                cnt += nsegs
                continue

            predictions = list(y[cnt:cnt+nsegs])
            average_prediction = get_most_common_labs(predictions)
            percentage_correct.append(lab in average_prediction)
            cnt += nsegs

        average_evaluation = sum(percentage_correct)/len(percentage_correct)
        print('Label Accuracy when using most predicted label(s) by all segments of the sequence = %f' % average_evaluation)
        with open(os.path.join(exp, "results", "label_avg_accuracy.txt"), 'w') as fd:
            fd.write("Result for %i layers with layer_size %i: %f" % (int(conf['num_dense_layers']), dense_layer_size, average_evaluation))

    noisy_eval = {}
    if plot_SNR == True:
        for k in range(int(conf['accuracy_window'])+1):
            noisy_eval[k] = {'noisesrc' : {}, 'SNR' : {}, 'clean' : 0.0}
            percentages_correct_src = defaultdict(list)
            percentages_correct_snr = defaultdict(list)
            percentages_correct_clean = []
            y = Y_test[k]
            for i in range(len(y)):
                segm = all_segs[i]
                if segm.split('_')[-2] == 'dB':
                    noisesrc = segm.split('_')[-4]
                    SNR = int(segm.split('_')[-3])
                    percentages_correct_src[noisesrc].append(Y_class_pred[i] in y[i])
                    percentages_correct_snr[SNR].append(Y_class_pred[i] in y[i])
                else:
                    percentages_correct_clean.append(Y_class_pred[i] in y[i])

            noisy_eval[k]['clean'] = sum(percentages_correct_clean)/len(percentages_correct_clean)

            for src in percentages_correct_src:
                noisy_eval[k]['noisesrc'][src] = sum(percentages_correct_src[src])/len(percentages_correct_src[src])
            for snr in percentages_correct_snr:
                noisy_eval[k]['SNR'][snr] = sum(percentages_correct_snr[snr])/len(percentages_correct_snr[snr])

    print('Noisy evaluation: ', noisy_eval)

    with open(os.path.join(exp, "results", "accuracy_noisy.txt"), 'w') as fd:
        fd.write("Result for %i layers with layer_size %i" % (int(conf['num_dense_layers']), dense_layer_size))
        for window in noisy_eval:
            fd.write("\t With accuracy window of size %i" % window)
            for snr in noisy_eval[window]['SNR']:
                fd.write("\t \t For an SNR of %i dB, accuracy = %f" % (snr, noisy_eval[window]['SNR'][snr]))
            fd.write("\t \t On clean set, accuracy = %f" % noisy_eval[window]['clean'])


    with open(os.path.join(exp, "results", "acc_win_noisy_eval_dict.pkl"), "wb") as fp:
        pickle.dump(noisy_eval, fp)

    return evaluation, noisy_eval, average_evaluation
