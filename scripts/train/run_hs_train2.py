from __future__ import absolute_import
import os
import sys
print(sys.path)
print(os.getcwd())
sys.path.insert(0, os.getcwd())
import time
import argparse
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
import pickle
import ast
import shutil
from collections import OrderedDict
from configparser import ConfigParser
from scripts.train.hs_train_loaders import load_data_reg
from fhvae.models.init_model import init_model
from fhvae.runners.hs_train_fhvae import hs_train_reg
from fhvae.runners.finetuning_fhvae import finetune

# tf.autograph.set_verbosity(10)

# For debugging on different GPU: os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

'''
Commands (pycharm setup)
Script path:
/users/spraak/jponcele/JakobFHVAE/scripts/train/run_hs_train.py

Parameters:
--expdir /esat/spchdisk/scratch/jponcele/fhvae_jakob/exp_</> 
--config configs/timit/config_bidirectional_39phones.cfg

Conda environment:
Python 3.6 -- conda env 'tf21'

Working directory:
/users/spraak/jponcele/JakobFHVAE
'''

def main(expdir, configfile):
    ''' main function '''

    # read and copy the config file, change location if necessary
    if os.path.exists(os.path.join(expdir, 'config.cfg')):
        print("Expdir already contains a config file... Overwriting!")
        os.remove(os.path.join(expdir, 'config.cfg'))

    shutil.copyfile(configfile, os.path.join(expdir, 'config.cfg'))
    conf = load_config(os.path.join(expdir, 'config.cfg'))
    conf['expdir'] = expdir

    if str(conf.get('mvn', True)).lower() in ['no', 'false']:
        conf['mvn'] = False
    else:
        conf['mvn'] = True

    # symbolic link dataset to ./datasets
    if not os.path.exists("./datasets"):
        os.mkdir("./datasets")
    # if os.path.islink(os.path.join("./datasets", conf['dataset'])):
    #     os.unlink(os.path.join("./datasets", conf['dataset']))
    # if os.path.isdir(os.path.join("./datasets", conf['dataset'])):
    #     print("Physical directory already exists in ./datasets, cannot create symlink of same name to the dataset.")
    #     exit(1)
    try:
        os.symlink(os.path.join(conf['datadir'], conf['dataset']), os.path.join("./datasets", conf['dataset']))
    except Exception:
        print("Could not create a symbolic link to the dataset dir")

    # set up the iterators over the dataset (for large datasets this may take a while)
    tr_nseqs, tr_shape, tr_iterator_by_seqs, dt_iterator, dt_dset, tr_dset = \
        load_data_reg(conf['dataset'], conf['fac_root'], conf['facs'], conf['talabs'], conf.get('num_noisy_versions'), mvn=conf['mvn'])

    # identify regularizing factors
    used_labs = conf['facs'].split(':')
    lab2idx = {name:tr_dset.labs_d[name].lablist for name in used_labs}
    print("labels and indices of facs: ", lab2idx)
    conf['lab2idx'] = lab2idx

    used_talabs = conf['talabs'].split(':')
    conf['talab_vals'] = tr_dset.talab_vals
    print("labels and indices of talabs: ", tr_dset.talab_vals)

    # apply talabs on z1 (b, e.g. time-aligned phones), labs on z2 (c, e.g. region, gender).
    c_n = OrderedDict([(lab, tr_dset.labs_d[lab].nclass) for lab in used_labs])
    b_n = OrderedDict([(talab, len(tr_dset.talab_vals[talab])) for talab in used_talabs])

    # save input shape [e.g. tuple (20,80)] and numclasses for testing phase
    conf['tr_shape'] = tr_shape
    conf['c_n'] = c_n
    conf['b_n'] = b_n

    print('#nmu2: ', conf['nmu2'])

    num_phones = len(tr_dset.talab_vals['phones'])
    conf['num_phones'] = num_phones
    print('number of phones: ', num_phones)

    if 'num_noisy_versions' not in conf:
        conf['num_noisy_versions'] = 0

    # dump settings
    with open(os.path.join(expdir, 'trainconf.pkl'), "wb") as fid:
        pickle.dump(conf, fid)

    # START
    #model, optimizer = init_model(conf, finetuning=False)
    #hs_train_reg(expdir, model, optimizer, conf, tr_iterator_by_seqs, dt_iterator, tr_dset, dt_dset, num_phones, noise_training=False)

    if conf.get('finetuning', False) in ['true', 'True', 'yes', 'Yes']:
        print('################### STARTING FINETUNING #######################')
        model, optimizer = init_model(conf, finetuning=True)
        finetune(expdir, model, optimizer, conf, tr_iterator_by_seqs, dt_iterator, tr_dset, dt_dset, num_phones, noise_training=True)


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


if __name__ == '__main__':

    print("I am process %s, running on %s: starting (%s)" % (
        os.getpid(), os.uname()[1], time.asctime()))

    #parse the arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--expdir", type=str, default="./exp",
                        help="where to store the experiment")
    parser.add_argument("--config", type=str, default="./config.cfg",
                        help="config file for the experiment")
    args = parser.parse_args()

    if os.path.isdir(args.expdir):
        print("Expdir already exists")
    os.makedirs(args.expdir, exist_ok=True)

    main(args.expdir, args.config)
