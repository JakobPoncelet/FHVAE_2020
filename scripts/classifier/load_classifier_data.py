import tensorflow as tf
import tensorflow.keras
import os
import numpy as np
import pickle
import bisect
from collections import defaultdict
import time

def load_data(expdir, latent_var, facdir, fac, valid_classes):

    # get data
    print('Retrieving training data')
    X_train, Y_train = getData(os.path.join(expdir, 'latent_vars', 'train'), latent_var, facdir, fac, valid_classes)
    print('Retrieving dev data ')
    X_dev, Y_dev = getData(os.path.join(expdir, 'latent_vars', 'dev'), latent_var, facdir, fac, valid_classes)

    return X_train, Y_train, X_dev, Y_dev


def getData(path, latent_var, facdir, fac, valid_classes):
    # returns two numpy arrays:
    # x contains the latent space representation(s) for every segment
    # y contains the centered talab for every segment

    z1_dict = {}
    z2_dict = {}

    if os.path.exists(os.path.join(path, 'z1', 'z1_mu_dict.pickle')):
        if 'z1' in latent_var.split('_'):
            with open(os.path.join(path, 'z1', 'z1_mu_dict.pickle'), 'rb') as handle_z1:
                z1_dict = pickle.load(handle_z1)
        if 'z2' in latent_var.split('_'):
            with open(os.path.join(path, 'z2', 'z2_mu_dict.pickle'), 'rb') as handle_z2:
                z2_dict = pickle.load(handle_z2)
    elif os.path.exists(os.path.join(path, 'z1', 'z1_0_dict.pickle')):
        if 'z1' in latent_var.split('_'):
            with open(os.path.join(path, 'z1', 'z1_0_dict.pickle'), 'rb') as handle_z1:
                z1_dict = pickle.load(handle_z1)
        if 'z2' in latent_var.split('_'):
            with open(os.path.join(path, 'z2', 'z2_0_dict.pickle'), 'rb') as handle_z2:
                z2_dict = pickle.load(handle_z2)
    else:
        if 'z1' in latent_var.split('_'):
            with open(os.path.join(path, 'z1', 'z1_dict.pickle'), 'rb') as handle_z1:
                z1_dict = pickle.load(handle_z1)
        if 'z2' in latent_var.split('_'):
            with open(os.path.join(path, 'z2', 'z2_dict.pickle'), 'rb') as handle_z2:
                z2_dict = pickle.load(handle_z2)

    z_dicts = {'z1': z1_dict, 'z2': z2_dict}

    facfile = os.path.join(facdir, "all_facs_%s.scp" % fac)

    if valid_classes is not None:
        valid_classes_as_set = set(valid_classes)

    seqlist = set()

    if fac == 'phones':
        with open(facfile, 'r') as fp:
            line = fp.readline()
            while line:
                if len(line.split(' ')) == 1:
                    seqname = line.rstrip()
                    seqlist.add(seqname)
                line = fp.readline()
    else:
        seq_labs = dict()
        with open(facfile, 'r') as fp:
            line = fp.readline()
            while line:
                seqname = line.rstrip().split(' ')[0]
                lab = line.rstrip().split(' ')[1]

                # under represented classes (not trained on in FHVAE)
                if lab in valid_classes_as_set:
                    seqlist.add(seqname)
                    seq_labs[seqname] = lab

                line = fp.readline()

    seg_starts = dict()
    seg_talabs = dict()
    with open(os.path.join(path, 'segments.txt')) as fid:
        line = fid.readline().rstrip()  # skip first
        line = fid.readline().rstrip()
        while line:
            seg_name = line.split(" ")[0]
            seg_start = int(line.split(" ")[1])
            seg_talab = int(line.split(" ")[2])

            seg_starts[seg_name] = seg_start
            seg_talabs[seg_name] = seg_talab

            line = fid.readline().rstrip()

    x_all = []
    y_all = []
    lablist = defaultdict(int)

    for seg in seg_starts:
        seq = '_'.join(seg.split('_')[:-1])

        # unlabeled or not trained on label
        if seq not in seqlist:
            continue

        if fac == 'phones':
            lab = int(seg_talabs[seg])
            lablist[lab] += 1
        else:
            lab = seq_labs[seq]
            lablist[lab] += 1
            lab = valid_classes.index(lab)

        if len(latent_var.split('_')) > 1:
            x = [z_dicts[var][seg] for var in latent_var.split('_')]
            x = tf.concat(x, axis=0)
        else:
            x = z_dicts[latent_var][seg]

        x_all.append(x)
        y_all.append(lab)

    x_all = tf.stack(x_all, axis=0)

    print('Number of segments for every lab in %s set: ' % os.path.basename(os.path.normpath(path)), lablist)

    return x_all, y_all


def load_data_test(expdir, latent_var, facdir, fac, valid_classes, phone_map=None, seg_len=20, accuracy_window=0):

    exp = os.path.join(expdir, 'latent_vars', 'test')

    z1_dict = {}
    z2_dict = {}

    if os.path.exists(os.path.join(exp, 'z1', 'z1_mu_dict.pickle')):
        if 'z1' in latent_var.split('_'):
            with open(os.path.join(exp, 'z1', 'z1_mu_dict.pickle'), 'rb') as handle_z1:
                z1_dict = pickle.load(handle_z1)
        if 'z2' in latent_var.split('_'):
            with open(os.path.join(exp, 'z2', 'z2_mu_dict.pickle'), 'rb') as handle_z2:
                z2_dict = pickle.load(handle_z2)
    elif os.path.exists(os.path.join(exp, 'z1', 'z1_0_dict.pickle')):
        if 'z1' in latent_var.split('_'):
            with open(os.path.join(exp, 'z1', 'z1_0_dict.pickle'), 'rb') as handle_z1:
                z1_dict = pickle.load(handle_z1)
        if 'z2' in latent_var.split('_'):
            with open(os.path.join(exp, 'z2', 'z2_0_dict.pickle'), 'rb') as handle_z2:
                z2_dict = pickle.load(handle_z2)
    else:
        if 'z1' in latent_var.split('_'):
            with open(os.path.join(exp, 'z1', 'z1_dict.pickle'), 'rb') as handle_z1:
                z1_dict = pickle.load(handle_z1)
        if 'z2' in latent_var.split('_'):
            with open(os.path.join(exp, 'z2', 'z2_dict.pickle'), 'rb') as handle_z2:
                z2_dict = pickle.load(handle_z2)

    z_dicts = {'z1': z1_dict, 'z2': z2_dict}

    facfile = os.path.join(facdir, "all_facs_%s.scp" % fac)

    if valid_classes is not None:
        valid_classes_as_set = set(valid_classes)

    x_all = []
    y_all = dict()
    for k in range(accuracy_window + 1):
        y_all[k] = []

    seg_starts = dict()
    seg_phones = dict()
    phones_per_seq = defaultdict(list)
    segs_per_seq = defaultdict(list)
    seg_cnt = None

    with open(os.path.join(exp, 'segments.txt'), 'r') as pid:
        line = pid.readline().rstrip()
        line = pid.readline().rstrip()
        while line:
            seg_name = line.split(' ')[0]
            seg_start = int(line.split(' ')[1])
            seg_phone = int(line.split(' ')[2])
            seq = '_'.join(seg_name.split('_')[:-1])

            seg_starts[seg_name] = seg_start
            seg_phones[seg_name] = seg_phone
            phones_per_seq[seq].append(seg_phone)
            segs_per_seq[seq].append(seg_name)

            line = pid.readline().rstrip()

    if fac == 'phones':
        flag = False
        phones_per_seq_new = dict()

        with open(facfile, 'r') as fp:
            line = fp.readline()
            while line:
                if len(line.split(' ')) == 1:
                    seq_name = line.rstrip()
                    # New 16/06
                    if seq_name in segs_per_seq:
                        phones_per_seq_new[seq_name] = {'starts': [], 'phones': []}
                        flag = True
                    else:
                        flag = False
                else:
                    if flag:
                        start = line.split(' ')[0].rstrip()
                        phone = line.split(' ')[2].rstrip()
                        phones_per_seq_new[seq_name]['starts'].append(int(start))
                        phones_per_seq_new[seq_name]['phones'].append(phone_map[phone])
                        # now convert phone to real phone through e.g. b_n (vb '0' --> int(1))

                line = fp.readline()

        if accuracy_window > 0:
            # also add e.g. next and previous phone as correct solutions (multiple phones can occur in same segment)

            all_segs = []
            for seq, segs in segs_per_seq.items():

                # unlabeled
                if seq not in phones_per_seq_new:
                    continue

                seq_phones = phones_per_seq_new[seq]['phones']
                seq_starts = phones_per_seq_new[seq]['starts']

                for seg in segs:
                    all_segs.append(seg)
                    seg_start = seg_starts[seg]
                    seg_center = int(seg_start + seg_len / 2)
                    if seg_center in seq_starts:
                        idx = seq_starts.index(seg_center)
                    else:
                        idx = bisect.bisect_left(seq_starts, seg_center) - 1
                    for k in range(0, accuracy_window+1):
                        phonelist = seq_phones[max(0,idx-k):idx+k+1]
                        y_all[k].append(set(phonelist))

                    if len(latent_var.split('_')) > 1:
                        x = [z_dicts[var][seg] for var in latent_var.split('_')]
                        x = tf.concat(x, axis=0)
                    else:
                        x = z_dicts[latent_var][seg]

                    x_all.append(x)

            x_all = tf.stack(x_all, axis=0)

            return x_all, y_all, all_segs, seg_cnt

        else:
            all_segs = []
            for seq, segs in segs_per_seq.items():

                # unlabeled
                if seq not in phones_per_seq_new:
                    continue

                for seg in segs:
                    all_segs.append(seg)
                    y_all[0].append(set([seg_phones[seg]]))

                    if len(latent_var.split('_')) > 1:
                        x = [z_dicts[var][seg] for var in latent_var.split('_')]
                        x = tf.concat(x, axis=0)
                    else:
                        x = z_dicts[latent_var][seg]

                    x_all.append(x)

            x_all = tf.stack(x_all, axis=0)

    # regular labs, no phone talabs
    else:
        seqlist = set()
        seq_labs = dict()

        with open(facfile, 'r') as fp:
            line = fp.readline()
            while line:
                seqname = line.rstrip().split(' ')[0]
                lab = line.rstrip().split(' ')[1]

                if lab in valid_classes_as_set:
                    seqlist.add(seqname)
                    seq_labs[seqname] = lab

                line = fp.readline()

        all_segs = []
        seg_cnt = []
        lablist = defaultdict(int)

        for seq, segs in segs_per_seq.items():

            if seq not in seqlist:
                continue

            lab = seq_labs[seq]
            lablist[lab] += 1
            lab = valid_classes.index(lab)

            is_noisy = 'dB' in seq
            seg_cnt.append((is_noisy, len(segs), lab))

            for seg in segs:
                all_segs.append(seg)
                y_all[0].append(set([lab]))

                if len(latent_var.split('_')) > 1:
                    x = [z_dicts[var][seg] for var in latent_var.split('_')]
                    x = tf.concat(x, axis=0)
                else:
                    x = z_dicts[latent_var][seg]

                x_all.append(x)

        x_all = tf.stack(x_all, axis=0)

        print('Number of segments for every lab in test set: ', lablist)

    return x_all, y_all, all_segs, seg_cnt
