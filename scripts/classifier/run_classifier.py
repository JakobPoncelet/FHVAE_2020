import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
#for gpu in gpus:
if gpus:
    gpu = gpus[0]
    try:
        tf.config.experimental.set_visible_devices(gpu, 'GPU')
        tf.config.experimental.set_memory_growth(gpu, enable=True)
        #break
    except RuntimeError as e:
        print(e)
        pass
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
import sys
import os
print(sys.path)
print(os.getcwd())
sys.path.insert(0, os.getcwd())
from configparser import ConfigParser
import argparse
import shutil
import time
import pickle
from scripts.classifier.load_classifier_data import load_data
from scripts.classifier.classifier_model import classifier
from scripts.classifier.test_classifier import testClassifier

## Usage: python scripts/classifier/run_classifier.py --expdir <expdir>
#                                                     --config configs/classifier/<...>
#                                                     --var <optional which latent variable as input, can be in config>
#                                                     --fac <optional which factor/label to classify, can be in config>
'''
Commands (pycharm setup)
Script path:
/users/spraak/jponcele/JakobFHVAE/scripts/classifier/run_classifier.py

Parameters:
--expdir /esat/spchdisk/scratch/jponcele/fhvae_jakob/exp_</>
--config configs/classifier/config_timit_39phones.cfg

Conda environment:
Python 3.6 -- conda env 'tf21'

Working directory:
/users/spraak/jponcele/JakobFHVAE
'''

def main(expdir, configfile, lvar, fac):

    if os.path.exists(os.path.join(expdir, 'config_classifier.cfg')):
        print("Expdir already contains a config file... Overwriting!")
        os.remove(os.path.join(expdir, 'config_classifier.cfg'))
    shutil.copyfile(configfile, os.path.join(expdir, 'config_classifier.cfg'))

    conf = load_config(os.path.join(expdir, 'config_classifier.cfg'))
    conf['expdir'] = expdir

    # Hardcoded :-(, to automatically loop over z1-z2-z1_z2 in condor_script if you want to
    if lvar is not None:
        if lvar == 'z1' or lvar == 'z2':
            conf['latent_var'] = lvar
            conf['latent_space_dim'] = 32
        elif lvar == 'z1_z2':
            conf['latent_var'] = lvar
            conf['latent_space_dim'] = 64
        else:
            print('USING LATENT VARIABLE FROM CONFIGURATION FILE')

    # for easy condor submission, put fac in config file, overwrite
    if 'fac' in conf:
        fac = conf['fac']
    print('............... Classifying on %s from %s ...........' % (fac, conf['latent_var']))

    with open(os.path.join(expdir, "trainconf.pkl"), "rb") as fid:
        trainconf = pickle.load(fid)

    if conf.get('plot_snr') in ['True', 'true', 'yes', 'Yes']:
        plot_SNR = True
    else:
        plot_SNR = False

    # compatibility with older configs
    if 'facdir' not in conf:
        conf['facdir'] = os.path.normpath(os.path.join(conf['phone_facs_file'], os.pardir))
    if 'num_classes' not in conf:
        conf['num_classes'] = conf['nb_phonemes']

    valid_classes = None
    if fac != 'phones':
        valid_classes = trainconf['lab2idx'][fac]
        valid_classes.pop(0)  # remove empty label ''
        conf['num_classes'] = len(valid_classes)

    X_train, Y_train, X_dev, Y_dev = load_data(expdir, conf['latent_var'], conf['facdir'], fac, valid_classes)
    Y_train = to_categorical(Y_train, int(conf['num_classes']))
    Y_dev = to_categorical(Y_dev, int(conf['num_classes']))

    evaluation_results = dict()
    noisy_evaluation_results = dict()
    avg_label_eval = dict()

    # test multiple layer sizes
    for layer_size in conf['dense_layer_size'].split(' '):
        layer_size = layer_size.strip()
        exp = os.path.join(expdir, 'classifier_exp', '%s_%s_%s_layers_size_%s' % (conf['latent_var'], fac, conf['num_dense_layers'], layer_size))
        os.makedirs(exp, exist_ok=True)

        # choose input shape
        input_shape = (int(conf['latent_space_dim']),)

        model = classifier(input_shape=input_shape, num_classes=int(conf['num_classes']), layers=int(conf['num_dense_layers']), dense_layer_size=int(layer_size))

        # checkpoints
        filepath = os.path.join(exp, "weights-improvement.hdf5")
        checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        training_log = os.path.join(exp, "training.log")
        csvLogger = CSVLogger(training_log, append=True)
        earlyStopping = EarlyStopping(patience=10, restore_best_weights=True)
        callbacks_list = [checkpoint, csvLogger, earlyStopping]

        history = model.fit(X_train, Y_train, epochs=int(conf['n_epochs']), batch_size=int(conf['batch_size']), verbose=True, validation_data=(X_dev, Y_dev), callbacks=callbacks_list)

        with open(os.path.join(exp, "historyDict"), 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

        model.save_weights(os.path.join(exp, "weights-improvement-final.hdf5"))

        ## test the model after training
        # test_results = model.evaluate(X_test, Y_test, verbose=1)
        # print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}%')

        acc_win_res, acc_noisy_win_res, avg_acc = testClassifier(expdir, exp, conf, trainconf, fac, valid_classes, int(layer_size), plot_SNR)

        evaluation_results[layer_size] = acc_win_res
        noisy_evaluation_results[layer_size] = acc_noisy_win_res
        avg_label_eval[layer_size] = avg_acc

    for layer in evaluation_results:
        print("\n \n Result for %i layers with layer_size %i" % (int(conf['num_dense_layers']), int(layer)))
        results = evaluation_results[layer]
        for window in results:
            print("\t With accuracy window of size ", window, " :  accuracy = ", results[window])

    for layer in noisy_evaluation_results:
        print("\n \n Result for %i layers with layer_size %i" % (int(conf['num_dense_layers']), int(layer)))
        results = noisy_evaluation_results[layer]
        for window in results:
            print("\t With accuracy window of size %i \n" % window)
            print("\t \t on clean test set, accuracy = %f \n" % results[window]['clean'])

            for snr in results[window]['SNR']:
                print("\t \t on test set with %i dB SNR, accuracy = %f \n" % (snr, results[window]['SNR'][snr]))

    for layer in avg_label_eval:
        result = avg_label_eval[layer]
        print("\n \n Result for %i layers with layer_size %i" % (int(conf['num_dense_layers']), int(layer)))
        print("\t Label accuracy using most predicted segment label = %f" % result)

    print("\n \n CLASSIFIER DONE.")


def load_config(conf):
    ''' Load configfile and extract arguments as a dict '''
    cfgfile = ConfigParser(interpolation=None)
    cfgfile.read(conf)
    train_conf = dict(cfgfile.items('RegFHVAE'))
    for key, val in train_conf.items():
        try:
            # get numbers as int/float/list
            train_conf[key] = ast.literal_eval(val)
        except:
            # text / paths
            pass
    return train_conf


if __name__ =='__main__':
    print("I am process %s, running on %s: starting (%s)" % (
        os.getpid(), os.uname()[1], time.asctime()))

    #parse the arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--expdir", type=str, default="./exp",
                        help="where the latent variables are stored")
    parser.add_argument("--config", type=str, default="./config_classifier.cfg",
                        help="config file for the classifier")
    parser.add_argument("--var", type=str, default=None,
                        help="which latent variable to use for classification")
    parser.add_argument("--fac", type=str, default="phones",
                        help='which factor to classify on (requires an all_facs_<factor>.scp file)')
    args = parser.parse_args()

    if os.path.isdir(args.expdir):
        print("Expdir already exists")
    os.makedirs(args.expdir, exist_ok=True)

    main(args.expdir, args.config, args.var, args.fac)
