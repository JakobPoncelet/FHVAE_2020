import tensorflow.keras
import os
import numpy as np
from collections import defaultdict
import bisect


def load_data(expdir, latent_var):

    # get data
    print('Retrieving training data')
    X_train, Y_train = getData(os.path.join(expdir, 'latent_vars', 'train'), latent_var)
    print('Retrieving test data')
    X_test, Y_test = getData(os.path.join(expdir, 'latent_vars', 'test'), latent_var)
    print('Retrieving dev data ')
    X_dev, Y_dev = getData(os.path.join(expdir, 'latent_vars', 'dev'), latent_var)

    return X_train, Y_train, X_dev, Y_dev, X_test, Y_test


def getData(path, latent_var):
    # returns two numpy arrays:
    # x contains the latent space representation(s) for every segment
    # y contains the centered talab for every segment

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

    x_all = np.expand_dims(np.array([]), axis=0)
    y_all = np.array([])

    for seg in seg_starts:
        phoneme = int(seg_talabs[seg])

        x_temp = np.array([])
        for var in latent_var.split('_'):
            x = np.expand_dims(np.load(os.path.join(path, var, '%s_%s.npy' % (seg,var))), axis=0)
            if x_temp.size:
                x_temp = np.concatenate((x_temp, x), axis=1)
            else:
                x_temp = x
        if x_all.size:
            x_all = np.concatenate((x_all, x_temp), axis=0)
        else:
            x_all = x_temp

        y_all = np.append(y_all, phoneme)
    
    return x_all, y_all


def load_data_test(expdir, latent_var, phone_facs_file, phone_map, seg_len=20, accuracy_window=0):

    exp = os.path.join(expdir, 'latent_vars', 'test')

    x_all = np.expand_dims(np.array([]), axis=0)
    y_all = dict()
    for k in range(accuracy_window + 1):
        y_all[k] = np.array([])

    seg_starts = dict()
    seg_phones = dict()
    phones_per_seq = defaultdict(list)
    segs_per_seq = defaultdict(list)

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

    phones_per_seq_new = dict()
    if accuracy_window > 0:
        with open(phone_facs_file, 'r') as fp:
            line = fp.readline()
            while line:
                if len(line.split(' ')) == 1:
                    seq_name = line.rstrip()
                    phones_per_seq_new[seq_name] = {'starts': [], 'phones': []}
                else:
                    start = line.split(' ')[0].rstrip()
                    phone = line.split(' ')[2].rstrip()
                    phones_per_seq_new[seq_name]['starts'].append(int(start))
                    phones_per_seq_new[seq_name]['phones'].append(phone_map[phone])
                    # phone nog omzetten naar phone via b_n (vb '0' --> int(1))

                line = fp.readline()

        for seq, segs in segs_per_seq.items():
            seq_phones = phones_per_seq_new[seq]['phones']
            seq_starts = phones_per_seq_new[seq]['starts']

            for seg in segs:
                seg_start = seg_starts[seg]
                seg_center = int(seg_start + seg_len / 2)
                if seg_center in seq_starts:
                    idx = seq_starts.index(seg_center)
                else:
                    idx = bisect.bisect_left(seq_starts, seg_center) - 1
                for k in range(0, accuracy_window+1):
                    phonelist = seq_phones[max(0,idx-k):idx+k+1]
                    y_all[k] = np.append(y_all[k], set(phonelist))

                x_temp = np.array([])

                for var in latent_var.split('_'):
                    x = np.expand_dims(np.load(os.path.join(exp, var, '%s_%s.npy' % (seg,var))), axis=0)
                    if x_temp.size:
                        x_temp = np.concatenate((x_temp, x), axis=1)
                    else:
                        x_temp = x

                if x_all.size:
                    x_all = np.concatenate((x_all, x_temp), axis=0)
                else:
                    x_all = x_temp

        return x_all, y_all

    else:
        for seq, segs in segs_per_seq.items():
            for seg in segs:
                y_all[0] = np.append(y_all[0], set(seg_phones[seg]))

                x_temp = np.array([])

                for var in latent_var.split('_'):
                    x = np.expand_dims(np.load(os.path.join(exp, var, '%s_%s.npy' % (seg, var))), axis=0)
                    if x_temp.size:
                        x_temp = np.concatenate((x_temp, x), axis=1)
                    else:
                        x_temp = x

                if x_all.size:
                    x_all = np.concatenate((x_all, x_temp), axis=0)
                else:
                    x_all = x_temp

        return x_all, y_all
