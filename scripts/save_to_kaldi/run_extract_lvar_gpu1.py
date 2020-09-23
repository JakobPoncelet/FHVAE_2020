from __future__ import absolute_import
import os
import sys
print(sys.path)
print(os.getcwd())
sys.path.insert(0, os.getcwd())
import time
import argparse
import pickle
import shutil
import ast
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
#for gpu in gpus:
if gpus:
    gpu = gpus[1]
    try:
        tf.config.experimental.set_visible_devices(gpu, 'GPU')
        tf.config.experimental.set_memory_growth(gpu, enable=True)
        #break
    except RuntimeError as e:
        print(e)
        pass

from configparser import ConfigParser
from collections import OrderedDict
from scripts.save_to_kaldi.save_cgn_kaldi import save_cgn_kaldi
from fhvae.models.init_model import create_model

os.environ["CUDA_VISIBLE_DEVICES"]="1"

'''
example:
python ./scripts/save_to_kaldi/run_extract_lvar.py --expdir /esat/spchdisk/scratch/jponcele/fhvae_jakob/cgn_without_a/ --datadir cgn_kaldi_feats_sp --segments ./misc/cgn/vl_without_a.segments --suffix train --speed_factor 0.9
'''


def main(expdir, test_config, datadir, segments, suff, seg_len, seg_shift, lvar, speed_factor):
    ''' main function '''

    # cleanup, we don't want to append to existing files (but don't delete dir when generating speed perturbed files)
    if os.path.exists(os.path.join(expdir, 'output_features_%s_%s' % (lvar,suff))):
        shutil.rmtree(os.path.join(expdir, 'output_features_%s_%s' % (lvar, suff)))
    os.makedirs(os.path.join(expdir, 'output_features_%s_%s' % (lvar, suff)), exist_ok=True)

    # load config
    if test_config is not None:
        cfgpath = test_config
    else:
        cfgpath = os.path.join(expdir, 'config.cfg')
    conf = load_config(cfgpath)
    conf['expdir'] = expdir

    if str(conf.get('mvn', True)).lower() in ['no', 'false']:
        conf['mvn'] = False
    else:
        conf['mvn'] = True

    # load tr_shape and classes from training stage
    with open(os.path.join(expdir, "trainconf.pkl"), "rb") as fid:
        trainconf = pickle.load(fid)
    conf['tr_shape'] = trainconf['tr_shape']
    conf['lab2idx'] = trainconf['lab2idx']
    conf['train_talab_vals'] = trainconf['talab_vals']

    # identify regularizing factors
    used_labs = conf['facs'].split(':')

    # When testing on new dataset, set this to HACK  (EXPERIMENTAL, NOT TESTED !!)
    if conf['dataset_test'] == conf['dataset']:
        c_n = trainconf['c_n']
        b_n = trainconf['b_n']
    else:
        print("Testing on different dataset then trained on, set number of classes manually.")
        c_n = OrderedDict([(used_labs[0], 3), (used_labs[1], 9)])  # HACK
        b_n = c_n

    conf['b_n'] = b_n
    conf['c_n'] = c_n

    if 'nmu2' in trainconf:
        conf['nmu2'] = trainconf['nmu2']

    num_phones = len(conf['train_talab_vals']['phones'])
    conf['num_phones'] = num_phones
    print('number of phones: ', num_phones)

    # initialize the model
    model = create_model(conf)

    if 'num_noisy_versions' not in conf:
        conf['num_noisy_versions'] = 0

    # generate and write out Z1 as features in KALDI format
    save_cgn_kaldi(expdir, model, conf, datadir, segments, suff, seg_len, seg_shift, lvar, speed_factor)


def load_config(conf):
    ''' Load configfile and extract arguments as a dict '''
    cfgfile = ConfigParser(interpolation=None)
    cfgfile.read(conf)
    test_conf = dict(cfgfile.items('RegFHVAE'))
    for key, val in test_conf.items():
        try:
            # get numbers as int/float/list
            test_conf[key] = ast.literal_eval(val)
        except:
            # text / paths
            pass
    return test_conf


if __name__ == '__main__':

    print("I am process %s, running on %s: starting (%s)" % (
        os.getpid(), os.uname()[1], time.asctime()))

    #parse the arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--expdir", type=str, default="./exp",
                        help="where to store the experiment (directory containing the trained model)")
    parser.add_argument("--config", type=str, default=None,
                        help="configuration file to use for testing (default is the one stored in expdir)")
    parser.add_argument("--datadir", type=str, default="cgn_kaldi_feats",
                        help="input features to extract latent var z1 from (in ./datasets/<datadir>")
    parser.add_argument("--segments", type=str, default="./misc/cgn/vl_without_a.segments",
                        help="segmentation file to follow for feature extraction")
    #parser.add_argument("--wavscp", type=str, default="./misc/cgn/vl_without_a.wavscp",
    #                    help="wav.scp file that comes with the segments file")
    parser.add_argument("--suffix", type=str, default="train",
                        help="suffix to add to output dir (train/dev/test)")
    parser.add_argument("--seg_len", type=int, default=20,
                        help="length of segments in number of frames (i.e. 20 = 200ms)")
    parser.add_argument("--seg_shift", type=int, default=1,
                        help="time between start of segments in number of frames (i.e. 20=200ms")
    parser.add_argument("--lvar", type=str, default="z1",
                        help="which latent variable featurevector to save (z1 or z2)")
    parser.add_argument("--speed_factor", type=float, default=None,
                        help="to extract speed perturbed features with provided factor (should be 0.9 or 1.1), or None")
    args = parser.parse_args()

    print("Expdir: %s" % args.expdir)
    if not os.path.isdir(args.expdir):
        print("Expdir does not exist.")
        exit(1)

    if args.speed_factor is not None:
        suffix = "%s_sp%s" % (args.suffix, str(args.speed_factor))
    else:
        suffix = args.suffix

    if args.seg_shift > 1:
        suffix = suffix + "_20ms"

    if args.lvar not in ['z1', 'z2']:
        raise ValueError("--lvar should be z1 or z2")

    main(args.expdir, args.config, args.datadir, args.segments, suffix, args.seg_len, args.seg_shift, args.lvar, args.speed_factor)
