from __future__ import division
import os
import sys
import time
import pickle
import json
import numpy as np
import shutil
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
from sklearn.manifold import TSNE
import tensorflow as tf
from .plotter import plot_x, scatter_plot
from fhvae.datasets.seq2seg import seq_to_seg_mapper

np.random.seed(123)

# Dumping of all dictionaries can be disabled in function _dump at bottom of script (e.g. for large datasets)

def test_reg(expdir, model, conf, tt_iterator_by_seqs, tt_seqs, tt_dset):
    '''
    Compute variational lower bound
    '''

    if os.path.exists(os.path.join(expdir, 'test')):
        shutil.rmtree(os.path.join(expdir, 'test'))

    os.makedirs(os.path.join(expdir, 'test'))

    # FOR CGN NOISY!!!
    if 'interleave' in conf:
        swp = int(conf['interleave'])
        tt_seqs = [seq for idx, seq in enumerate(tt_seqs) if (idx-swp) % 6 == 0]

    # print output also to a file as well as console
    fff = os.path.join(expdir, 'test', 'output.txt')

    _print("\nRESTORING MODEL", fff)
    starttime = time.time()
    optimizer = tf.keras.optimizers.Adam(learning_rate=conf['lr'], beta_1=conf['beta1'], beta_2=conf['beta2'], epsilon=conf['adam_eps'], amsgrad=False)
    checkpoint_directory = os.path.join(expdir, 'training_checkpoints')
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

    if os.path.exists(os.path.join(checkpoint_directory, 'best_checkpoint')):
        with open(os.path.join(checkpoint_directory, 'best_checkpoint'), 'r') as pid:
            best_checkpoint = (pid.readline()).rstrip()

        # delete other checkpoints to save space
        cp_list = os.listdir(checkpoint_directory)
        for cp in cp_list:
            if (not cp.startswith('ckpt-'+str(best_checkpoint))) and (not cp == 'best_checkpoint') and (not cp == 'checkpoint'):
                os.remove(os.path.join(checkpoint_directory, cp))

        status = checkpoint.restore(os.path.join(checkpoint_directory, 'ckpt-' + str(best_checkpoint)))

    else:
        manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_directory, max_to_keep=5)
        status = checkpoint.restore(manager.latest_checkpoint)

    _print("restoring model takes %.2f seconds" % (time.time()-starttime), fff)
    status.assert_existing_objects_matched()
    #status.assert_consumed()

    _init_dirs(expdir)

    seg_len = tf.constant(tt_dset.seg_len, tf.int64)
    seg_shift = tf.constant(tt_dset.seg_shift, tf.int64)

    def segment_map_fn_test(idx, data):
        # map every sequence to <variable #> segments
        # + use enumeration-number as sequence key such that all keys in range [0, nmu2]
        keys, feats, lens, labs, talabs = data
        keys, feats, lens, labs, talabs, starts = seq_to_seg_mapper(idx, keys, feats, lens, labs, talabs, \
                                                            seg_len, seg_shift, False,
                                                            len(conf['b_n']), 0)
        return keys, feats, lens, labs, talabs, starts

    def create_test_dataset(seqs):

        Test_Dataset = tf.data.Dataset.from_generator(
            lambda: tt_iterator_by_seqs(seqs, bs=conf['batch_size']), \
            output_shapes=((), (None, conf['tr_shape'][1]), (), (len(conf['c_n']),), (len(conf['b_n']), 3, None)), \
            output_types=(tf.int64, tf.float32, tf.int64, tf.int64, tf.int64)) \
            .enumerate(start=0) \
            .map(segment_map_fn_test, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
            .unbatch() \
            .batch(batch_size=conf['batch_size']) \
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return Test_Dataset


    _print("\nCOMPUTING AVERAGE VALUES", fff)
    avg_loss, avg_vals = compute_average_values(expdir, model, tt_seqs, conf, create_test_dataset, fff)

    _print("\nCOMPUTING VALUES BY PHONE CLASS", fff)
    z1_by_phone, mu1_by_phone, z1_by_phone_and_lab = compute_values_by_phone(model, conf, create_test_dataset, tt_seqs, expdir, fff)

    _print("\nCOMPUTING VALUES BY SEQUENCE", fff)
    z1_by_seq, z2_by_seq, mu2_by_seq, regpost_by_seq, xin_by_seq, xout_by_seq, xoutv_by_seq, z1reg_by_seq, regpost_by_seq_z1, z2reg_by_seq, bReg_by_seq, cReg_by_seq, z1advreg_by_seq, z2advreg_by_seq, advregpost_by_seq, mu1_by_seq, z1_0_by_seq, z2_0_by_seq, z1_T_by_seq, z2_T_by_seq, mu2_by_lab = compute_values_by_seq(model, conf, create_test_dataset, tt_seqs, expdir, tt_dset, fff)

    _print("\nSAVE COMPUTED VALUES", fff)
    _dump(expdir, z1_by_seq, z2_by_seq, mu2_by_seq, regpost_by_seq, xin_by_seq, xout_by_seq, xoutv_by_seq, z1reg_by_seq, regpost_by_seq_z1, z2reg_by_seq, bReg_by_seq, cReg_by_seq, z1advreg_by_seq, z2advreg_by_seq, advregpost_by_seq, mu1_by_seq, z1_0_by_seq, z2_0_by_seq, z1_T_by_seq, z2_T_by_seq, z1_by_phone, mu1_by_phone, z1_by_phone_and_lab, mu2_by_lab, fff)

    _print("\nCOMPUTING CLUSTER ANALYSIS", fff)
    variances_mu2 = compute_cluster_analysis_mu2(expdir, conf, tt_seqs, tt_dset, mu2_by_seq, fff)
    variances_z1, mean_variances_z1 = compute_cluster_analysis_z1(expdir, conf, tt_seqs, tt_dset, mu1_by_phone, z1_by_phone_and_lab, fff)

    _print("\nCOMPUTING PREDICTION ACCURACIES FOR LABELS FROM Z2", fff)
    labels, accs = compute_pred_acc_z2(expdir, model, conf, tt_seqs, tt_dset, regpost_by_seq, z2reg_by_seq, cReg_by_seq, fff)

    _print("\nCOMPUTING PREDICTION ACCURACIES FOR TIME ALIGNED LABELS FROM Z1", fff)
    labels, accs = compute_pred_acc_z1(expdir, model, conf, tt_seqs, tt_dset, regpost_by_seq, z1reg_by_seq, bReg_by_seq, fff)

    _print("\nADVERSARIAL PREDICTION ACCURACIES FOR SEQUENCE LABELS FROM Z1", fff)
    labels, accs = compute_advpred_acc_z1(expdir, model, conf, tt_seqs, tt_dset, advregpost_by_seq, z1advreg_by_seq, fff)

    _print("\nADVERSARIAL PREDICTION ACCURACIES FOR TIME ALIGNED LABELS FROM Z2", fff)
    labels, accs = compute_advpred_acc_z2(expdir, model, conf, tt_seqs, tt_dset, z2advreg_by_seq, bReg_by_seq, fff)

    _print("\nVISUALIZING RESULTS", fff)
    visualize_reg_vals(expdir, model, tt_seqs, tt_dset, conf, z1_by_seq, z2_by_seq, mu2_by_seq, regpost_by_seq, xin_by_seq, xout_by_seq, xoutv_by_seq, z1reg_by_seq, z1_0_by_seq, z2_0_by_seq, z1_T_by_seq, z2_T_by_seq, mu2_by_lab, fff)

    _print("\nVISUALIZING TSNE BY LABEL", fff)
    tsne_by_label(expdir, model, conf, create_test_dataset, tt_seqs, tt_dset, bReg_by_seq, z1_by_phone, fff)

    _print("\nFINISHED\nResults stored in %s/test" % expdir, fff)


def compute_average_values(expdir, model, tt_seqs, conf, create_test_dataset, fff):

    num_seqs = len(tt_seqs)
    _print('num_seqs = %i   (note time/memory if >10k seqs)' % num_seqs, fff)

    seqs = list(tt_seqs)
    num_steps = len(seqs) // conf['nmu2']

    mu_stats = defaultdict(list)

    for step in range(0, num_steps + 1):
        sample_seqs = seqs[step * conf['nmu2']:min((step + 1) * conf['nmu2'], len(seqs))]
        Test_Dataset = create_test_dataset(seqs=sample_seqs)

        mu1_table = tf.zeros([conf['num_phones'], conf['z1_dim']], dtype=tf.float32)
        mu2_table = tf.zeros([conf['nmu2'], conf['z2_dim']], dtype=tf.float32)
        nsegs = tf.zeros([conf['nmu2']])
        phone_occs = tf.zeros([conf['num_phones']])

        # calculate mu1-dict and mu2-dict
        mu1_table, mu2_table, phone_occs = estimate_mu1_mu2_dict(model, Test_Dataset, mu1_table, mu2_table, phone_occs,
                                                                 nsegs, conf['tr_shape'])
        model.mu1_table.assign(mu1_table)
        model.mu2_table.assign(mu2_table)
        model.phone_occs.assign(phone_occs)

        mu1_stats = _print_mu_stat(mu1_table, 'mu1', fff)
        mu2_stats = _print_mu_stat(mu2_table, 'mu2', fff)
        mu_stats['mu1'].append(mu1_stats)
        mu_stats['mu2'].append(mu2_stats)

        sum_names = ['log_pmu2', 'log_pmu1', 'neg_kld_z2', 'neg_kld_z1', 'log_px_z', 'lb', 'log_qy_mu2', 'log_qy_mu1', 'log_b_loss', 'log_c_loss', 'advlog_b_loss', 'advlog_c_loss']
        sum_loss = 0.
        sum_vals = [0. for _ in range(len(sum_names))]
        tot_segs = 0.
        avg_vals = [0. for _ in range(len(sum_names))]

        stddevs_z1 = []
        stddevs_z2 = []
        vars_z1 = []
        vars_z2 = []

        for yval, xval, nval, cval, bval, _ in Test_Dataset:
            nval = tf.cast(nval, dtype=tf.float32)

            mu2, mu1, qz2_x, z2_sample, z2_sample_0, qz1_x, z1_sample, z1_sample_0, px_z, x_sample, z1_rlogits, z2_rlogits, z1_advrlogits, z2_advrlogits = model(xval, yval, bval[:, 0])

            loss, log_pmu2, log_pmu1, neg_kld_z2, neg_kld_z1, log_px_z, lb, log_qy_mu2, log_qy_mu1, log_b_loss, log_c_loss, advlog_b_loss, advlog_c_loss, noise_loss = \
                model.compute_loss(xval, yval, nval, bval, cval, mu2, mu1, qz2_x, z2_sample, qz1_x, z1_sample, px_z, x_sample, z1_rlogits, z2_rlogits, z1_advrlogits, z2_advrlogits, num_seqs)

            results = [log_pmu2, log_pmu1, neg_kld_z2, neg_kld_z1, log_px_z, lb, log_qy_mu2, log_qy_mu1, log_b_loss, log_c_loss, advlog_b_loss, advlog_c_loss]
            for i in range(len(sum_vals)):
                sum_vals[i] += -tf.reduce_mean(results[i])
            sum_loss += loss
            tot_segs += len(xval)

            stddevs_z1.append(tf.reduce_mean(tf.exp(qz1_x[1]*0.5)))
            stddevs_z2.append(tf.reduce_mean(tf.exp(qz2_x[1]*0.5)))
            vars_z1.append(tf.reduce_mean(tf.exp(qz1_x[1])))
            vars_z2.append(tf.reduce_mean(tf.exp(qz2_x[1])))

    _print('mean variance z1: %f' % tf.reduce_mean(vars_z1).numpy(), fff)
    _print('mean stddev z1: %f' % tf.reduce_mean(stddevs_z1).numpy(), fff)
    _print('mean variance z2: %f' % tf.reduce_mean(vars_z2).numpy(), fff)
    _print('mean stddev z2: %f' % tf.reduce_mean(stddevs_z2).numpy(), fff)

    avg_loss = sum_loss/tot_segs
    _print("average loss = %f" % (avg_loss), fff)
    for i in range(len(sum_vals)):
        avg_vals[i] = sum_vals[i]/tot_segs
        _print("\t average value for %s \t= %f" % (sum_names[i], avg_vals[i]), fff)

    with open(os.path.join(expdir, 'test', 'txt', 'averageloss.txt'), 'w+') as f:
        f.write("average loss = %f\n" % (avg_loss))
        f.write("              x batch_size = %f\n" % (avg_loss*conf['batch_size']))
        for i in range(len(sum_vals)):
            avg_vals[i] = sum_vals[i] / tot_segs
            f.write("\t average value for %s \t= %f\n" % (sum_names[i], avg_vals[i]))

    with open(os.path.join(expdir, 'test', 'txt', 'meanvar.txt'), 'w+') as f:
        f.write('mean variance z1: %f\n' % tf.reduce_mean(vars_z1))
        f.write('mean stddev z1: %f\n' % tf.reduce_mean(stddevs_z1))
        f.write('mean variance z2: %f\n' % tf.reduce_mean(vars_z2))
        f.write('mean stddev z2: %f\n' % tf.reduce_mean(stddevs_z2))

    for n in ['mu1', 'mu2']:
        with open(os.path.join(expdir, 'test', 'txt', '%s_stats.txt' % n), 'w+') as f:
            f.write('%s-table stats \n' % n)
            for stat in mu_stats[n]:
                f.write("avg. norm = %.2f, #%s = %s \n" % (stat[0], n, stat[2]))
                f.write("per dim: %s \n" % (" ".join(["%.2f" % v for v in stat[1]]),))

    return avg_loss, avg_vals


def compute_values_by_phone(model, conf, create_test_dataset, seqs, expdir, fff):

    z1_by_phone = defaultdict(list)
    covar_by_phone = {'cov_det':{}, 'diagcov_det':{}, 'cov_logdet':{}, 'diagcov_logdet':{}}
    diagcovdet_by_phone = dict()
    phone_occs = defaultdict(int)
    mu1_by_phone = dict()

    all_facs = conf['facs'].split(':')
    z1_by_lab_and_phone = dict()
    for fac in all_facs:
        z1_by_lab_and_phone[fac] = dict()
        for i in range(0, len(conf['lab2idx'][fac])):
            lab = str(i)
            z1_by_lab_and_phone[fac][lab] = defaultdict(list)


    complete_Test_Dataset = create_test_dataset(seqs=seqs)

    for _, x, _, c, b, _ in complete_Test_Dataset:
        _, _, _, _, _, _, _, _, qz1_x, _ = model.encoder(x)
        z1_mu = qz1_x[0]
        for idx in range(0, int(b.get_shape().as_list()[0])):
            ph = str(b[idx, 0].numpy())
            z1_by_phone[ph].append(z1_mu[idx, :])
            phone_occs[ph] += 1

            for col in range(0, int(c.get_shape().as_list()[1])):
                fac = all_facs[col]
                lab = str(c[idx, col].numpy())
                z1_by_lab_and_phone[fac][lab][ph].append(z1_mu[idx, :])

    # compute within-class covariance matrix and compute the determinant of this covariance matrix
    # + the determinant of the diagonal of this matrix
    # + logdeterminants
    for ph, all_z1s in z1_by_phone.items():
        cov_mat = np.cov(np.stack(all_z1s, axis=0), rowvar=False)
        if len(cov_mat.shape) == 2:
            covar_by_phone['cov_det'][ph] = np.linalg.det(cov_mat)
            covar_by_phone['diagcov_det'][ph] = np.prod(np.diag(cov_mat))
            (sign, logdet) = np.linalg.slogdet(cov_mat)
            covar_by_phone['cov_logdet'][ph] = logdet
            (sign, logdet) = np.linalg.slogdet(np.diag(np.diag(cov_mat)))
            covar_by_phone['diagcov_logdet'][ph] = logdet

    _print('Average determinant of covariance matrix per phone is %f\n'
          % np.average(list(covar_by_phone['cov_det'].values())), fff)
    _print('Average determinant of diagonal covariance matrix per phone is %f\n'
          % np.average(list(covar_by_phone['diagcov_det'].values())), fff)
    _print('Average log-determinant of covariance matrix per phone is %f\n'
          % np.average(list(covar_by_phone['cov_logdet'].values())), fff)
    _print('Average log-determinant of diagonal covariance matrix per phone is %f\n'
          % np.average(list(covar_by_phone['diagcov_logdet'].values())), fff)

    # posterior map estimation of mu1 according to phone class
    r = np.exp(model.pz1_stddev ** 2) / np.exp(model.pmu1_stddev ** 2)
    for ph, all_z1s in z1_by_phone.items():
        mu1_by_phone[ph] = np.sum(all_z1s, axis=0) / (phone_occs[ph] + r)

    with open(os.path.join(expdir, 'test', 'txt', 'logdet.txt'), 'w+') as f:
        f.write('Average determinant of covariance matrix per phone is %f\n'
              % np.average(list(covar_by_phone['cov_det'].values())))
        f.write('Average determinant of diagonal covariance matrix per phone is %f\n'
              % np.average(list(covar_by_phone['diagcov_det'].values())))
        f.write('Average log-determinant of covariance matrix per phone is %f\n'
              % np.average(list(covar_by_phone['cov_logdet'].values())))
        f.write('Average log-determinant of diagonal covariance matrix per phone is %f\n'
              % np.average(list(covar_by_phone['diagcov_logdet'].values())))

    return z1_by_phone, mu1_by_phone, z1_by_lab_and_phone


def compute_values_by_seq(model, conf, create_test_dataset, seqs, expdir, tt_dset, fff):

    z1_by_seq = defaultdict(list)
    z2_by_seq = defaultdict(list)
    mu2_by_seq = dict()
    mu1_by_seq = dict()  # old s_vector mu1, not from mu1-table
    regpost_by_seq = dict()
    advregpost_by_seq = dict()
    xin_by_seq = defaultdict(list)
    xout_by_seq = defaultdict(list)
    xoutv_by_seq = defaultdict(list)
    z1reg_by_seq = defaultdict(list)
    z2reg_by_seq = defaultdict(list)
    z1advreg_by_seq = defaultdict(list)
    z2advreg_by_seq = defaultdict(list)
    regpost_by_seq_z1 = dict()
    bReg_by_seq = defaultdict(list)
    cReg_by_seq = defaultdict(list)

    z1_0_by_seq = defaultdict(list)
    z2_0_by_seq = defaultdict(list)
    z1_T_by_seq = defaultdict(list)
    z2_T_by_seq = defaultdict(list)

    for seq in seqs:

        Test_Dataset = create_test_dataset(seqs=[seq])

        for _, x, _, c, b, _ in Test_Dataset:
            _, _, z1_sample, z1_sample_0, _, _, z2_sample, z2_sample_0, qz1_x, qz2_x = model.encoder(x)
            z2_by_seq[seq].append(qz2_x[0])
            z1_by_seq[seq].append(qz1_x[0])

            xin_by_seq[seq].append(x)

            _, _, _, _, px_z = model.decoder(x, qz1_x[0], qz2_x[0])
            xout_by_seq[seq].append(px_z[0])
            xoutv_by_seq[seq].append(px_z[1])

            # probabilities of each of the regularisation classes given mean(z1)
            z1_rlogits, z2_rlogits = model.regulariser(qz1_x[0], qz2_x[0])
            # softmax over columns 1:end (skip first column of zeros with unlabeled data)
            z1reg_by_seq[seq] = list(map(_softmax, z1_rlogits))
            z2reg_by_seq[seq] = list(map(_softmax, z2_rlogits))

            # adversarial objective regularisation probabilities
            z1_advrlogits, z2_advrlogits = model.advregulariser(qz1_x[0], qz2_x[0])
            z1advreg_by_seq[seq] = list(map(_softmax, z1_advrlogits))
            z2advreg_by_seq[seq] = list(map(_softmax, z2_advrlogits))

            cReg_by_seq[seq].append(c)
            bReg_by_seq[seq].append(b)

            z1_0_by_seq[seq].append(z1_sample_0)
            z2_0_by_seq[seq].append(z2_sample_0)
            z1_T_by_seq[seq].append(z1_sample)
            z2_T_by_seq[seq].append(z2_sample)

        z1_by_seq[seq] = np.concatenate(z1_by_seq[seq], axis=0)
        z2_by_seq[seq] = np.concatenate(z2_by_seq[seq], axis=0)
        xin_by_seq[seq] = np.concatenate(xin_by_seq[seq], axis=0)
        xout_by_seq[seq] = np.concatenate(xout_by_seq[seq], axis=0)
        xoutv_by_seq[seq] = np.concatenate(xoutv_by_seq[seq], axis=0)
        # z1reg_by_seq[seq] = np.concatenate(z1reg_by_seq[seq], axis=0)

        bReg_by_seq[seq] = np.concatenate(bReg_by_seq[seq], axis=0)
        cReg_by_seq[seq] = np.concatenate(cReg_by_seq[seq], axis=0)

        z1_0_by_seq[seq] = np.concatenate(z1_0_by_seq[seq], axis=0)
        z2_0_by_seq[seq] = np.concatenate(z2_0_by_seq[seq], axis=0)
        z1_T_by_seq[seq] = np.concatenate(z1_T_by_seq[seq], axis=0)
        z2_T_by_seq[seq] = np.concatenate(z2_T_by_seq[seq], axis=0)

        # formula for inferring S-vector mu2 during testing, paper p5 (over all segments from same sequence)
        z2_sum = np.sum(z2_by_seq[seq], axis=0)
        n = len(z2_by_seq[seq])
        r = np.exp(model.pz2_stddev ** 2) / np.exp(model.pmu2_stddev ** 2)
        mu2 = z2_sum / (n+r)
        mu2_by_seq[seq] = np.asarray(mu2).reshape([1, mu2.shape[0]])

        # probabilities of each of the regularisation classes given the computed z2 of above
        _, z2_rlogits = model.regulariser(z1_by_seq[seq], mu2_by_seq[seq])
        regpost_by_seq[seq] = list(map(_softmax, z2_rlogits))

        # formula for inferring alternative S-vector mu1 during testing, paper p7
        z1_sum = np.sum(z1_by_seq[seq], axis=0)
        n = len(z1_by_seq[seq])
        r = np.exp(model.pz1_stddev ** 2)
        mu1 = z1_sum / (n+r)
        mu1_by_seq[seq] = np.asarray(mu1).reshape([1, mu1.shape[0]])

        # same but adverarial objective: probs of classes from z1
        # use mu1 s-vector instead to get summarisy of sequence
        z1_advrlogits, _ = model.advregulariser(mu1_by_seq[seq], z2_by_seq[seq])
        advregpost_by_seq[seq] = list(map(_softmax, z1_advrlogits))

        # probabilities given computed mu1
        z1_rlogits, _ = model.regulariser(mu1_by_seq[seq], z2_by_seq[seq])
        # softmax over columns 1:end, first column is for unlabeled data
        regpost_by_seq_z1[seq] = list(map(_softmax, z1_rlogits))

    # Calculate reconstruction MSE
    with open(os.path.join(expdir, 'test', 'txt', 'reconstruction_MSE.txt'), 'w') as f:
        mse = 0.
        for seq in seqs:
            mse += np.square(np.array(xin_by_seq[seq]) - np.array(xout_by_seq[seq])).mean()
        _print('reconstruction MSE: {:2f}'.format(mse / len(seqs)), fff)
        f.write(str(mse / len(seqs)))

    # save the mean mu2 (neutral)
    if not os.path.exists(os.path.join(expdir, 'test', 'spec', 'mean_mu2.npy')):
        mumu = np.zeros([mu2_by_seq[seqs[1]].size])
        for seq in seqs:
            mumu += mu2_by_seq[seq].flatten()
        mumu /= len(seqs)
        with open(os.path.join(expdir, 'test', 'spec', 'mean_mu2.npy'), "wb") as fnp:
            np.save(fnp, mumu)

    # average mu2 by label
    names = conf['facs'].split(':')
    lab2idx = conf['lab2idx']
    mu2_by_lab = dict()
    for name in names:
        mu2_by_lab[name] = dict()  # store sum first, then store mean
        lab_count = dict()  # amount of speakers per label
        for seq in seqs:
            lab = tt_dset.labs_d[name].seq2lab[seq]
            if lab not in lab_count:
                lab_count[lab] = 1
                mu2_by_lab[name][lab] = mu2_by_seq[seq]
            else:
                lab_count[lab] += 1
                mu2_by_lab[name][lab] += mu2_by_seq[seq]

        for lab in lab_count.keys():
            mu2_by_lab[name][lab] = mu2_by_lab[name][lab] / lab_count[lab]  # compute mean

    return z1_by_seq, z2_by_seq, mu2_by_seq, regpost_by_seq, xin_by_seq, xout_by_seq, xoutv_by_seq, z1reg_by_seq, regpost_by_seq_z1, z2reg_by_seq, bReg_by_seq, cReg_by_seq, z1advreg_by_seq, z2advreg_by_seq, advregpost_by_seq, mu1_by_seq, z1_0_by_seq, z2_0_by_seq, z1_T_by_seq, z2_T_by_seq, mu2_by_lab


def compute_cluster_analysis_mu2(expdir, conf, seqs, tt_dset, mu2_by_seq, fff):

    ''' Calculate intra-cluster variance for each label '''

    names = conf['facs'].split(':')
    if 'spk' not in names:
        names.append('spk')

    lab2idx = conf['lab2idx']
    variances = dict()
    for name in names:

        variances[name] = dict()
        samples_by_lab = dict()
        for seq in seqs:
            if name == 'spk':
                lab = seq.split('_')[0]
            else:
                lab = tt_dset.labs_d[name].seq2lab[seq]
            if lab != "":
                if lab not in samples_by_lab:
                    samples_by_lab[lab] = [mu2_by_seq[seq]]
                else:
                    samples_by_lab[lab].append(mu2_by_seq[seq])

        for lab in samples_by_lab.keys():
            samples_by_lab[lab] = np.array(samples_by_lab[lab])  # shape e.g. 79x1x32 (79=#occs of label)
            variances[name][lab] = _variance(samples_by_lab[lab])

        variances[name]['_mean'] = np.mean(list(variances[name].values()))

    all_mu2s = np.array(list(mu2_by_seq.values()))
    variances['_global variance'] = _variance(all_mu2s)

    with open(os.path.join(expdir, 'test', 'txt', 'intra_cluster_variance.json'), 'w+') as f:
        json.dump(variances, f, sort_keys=True, indent=4)

    return variances


def compute_cluster_analysis_z1(expdir, conf, seqs, tt_dset, mu1_by_phone, z1_by_phone_and_lab, fff):

    names = conf['facs'].split(':')
    fac_z1s = dict()
    variances = dict()
    mean_variances = dict()

    for name in names:
        fac_z1s[name] = defaultdict(list)
        variances[name] = {'global':{}}
        mean_variances[name] = dict()

        # skip empty labels 0 (lab 0 and phone 0)
        for lab in range(1, len(conf['lab2idx'][name])):
            labstr = conf['lab2idx'][name][lab]
            variances[name][labstr] = dict()

            for phone in range(1, conf['num_phones']):
                all_z1s = np.asarray([np.asarray(z1) for z1 in z1_by_phone_and_lab[name][str(lab)][str(phone)]])
                var = _variance(all_z1s)
                # print(name, labstr, str(phone))
                variances[name][labstr][phone] = var

                if len(all_z1s) > 0:
                    if len(all_z1s.shape) == 1:
                        all_z1s = all_z1s[np.newaxis, ...]
                    fac_z1s[name][str(phone)].append(all_z1s)

            mean_variances[name][labstr] = np.mean(list((variances[name][labstr]).values()))

        for phone in range(1, conf['num_phones']):
            global_var = _variance(np.asarray(np.concatenate(fac_z1s[name][str(phone)])))
            variances[name]['global'][phone] = global_var

        mean_variances[name]['global'] = np.mean(list(variances[name]['global'].values()))

    _print('per label mean_of variances of z1_mu per phone\n', fff)
    _print(mean_variances, fff)

    with open(os.path.join(expdir, 'test', 'txt', 'z1_variances_by_phone.json'), 'w+') as f:
        json.dump(variances, f, sort_keys=True, indent=2)
    with open(os.path.join(expdir, 'test', 'txt', 'z1_mean_variances_by_label.json'), 'w+') as f:
        json.dump(mean_variances, f, sort_keys=True, indent=2)

    return variances, mean_variances


def compute_pred_acc_z2(expdir, model, conf, seqs, tt_dset, regpost_by_seq, z2reg_by_seq, cReg_by_seq, fff):

    names = conf['facs'].split(':')
    lab2idx= conf['lab2idx']
    accuracies = [0. for _ in range(len(names))]
    #accuracies_z2reg = [0. for _ in range(len(names))]

    clean_accuracies = [0. for _ in range(len(names))]

    for i, name in enumerate(names):

        ordered_labs = lab2idx[name]
        truelabs = tt_dset.labs_d[name].seq2lab

        total = 0
        correct = 0  #using mu2
        #correct_z2reg = 0  #using z2_rlogits

        clean_total = 0
        clean_correct = 0

        with open("%s/test/txt/label_accuracy/%s_predictions.scp" % (expdir, name), "w") as f:
            f.write("#seq truelabel predictedlabel      for class %s \n" % name)
            for seq in seqs:
                clean = True

                if 'dB' in seq:
                    clean = False

                # when no or unknown label ""
                if len(truelabs[seq]) == 0:
                    continue

                total += 1
                if clean:
                    clean_total += 1

                probs = regpost_by_seq[seq][i]
                # max + 1 since the first label in ordered_labs is the unknown label ""
                pred_lab = ordered_labs[np.argmax(probs)+1]
                if pred_lab == truelabs[seq]:
                    correct += 1
                    if clean:
                        clean_correct += 1

                f.write(seq+" "+str(truelabs[seq])+" "+str(pred_lab)+"\n")

                #probs_z2reg = z2reg_by_seq[seq][i]
                #pred_lab_z2reg = ordered_labs[np.argmax(np.sum(probs_z2reg, axis=0))+1]
                #if pred_lab_z2reg == truelabs[seq]:
                #    correct_z2reg += 1

        accuracies[i] = correct/total

        if clean_total == 0:  # divide by zero error in case all noisy
            clean_total = 1
        clean_accuracies[i] = clean_correct / clean_total

        with open("%s/test/txt/label_accuracy/%s_acc" % (expdir, name), "w") as fid:
            fid.write("%10.3f \n" % accuracies[i])
        _print("prediction accuracy for labels of class %s is %f  (and %f on clean test set)" % (name, accuracies[i], clean_accuracies[i]), fff)

        #accuracies_z2reg[i] = correct_z2reg/total
        #_print("prediction accuracy for labels of class from z2reg_by_seq %s is %f" % (name, accuracies[i]), fff)

    return names, accuracies


def compute_pred_acc_z1(expdir, model, conf, seqs, tt_dset, regpost_by_seq, z1reg_by_seq, bReg_by_seq, fff):

    names = conf['talabs'].split(':')
    talab2idx = conf['train_talab_vals']

    accuracies = [0. for _ in range(len(names))]
    clean_accuracies = [0. for _ in range(len(names))]

    for i, name in enumerate(names):
        with open("%s/test/txt/label_accuracy/%s_predictions.scp" % (expdir, name), "w") as f:
            f.write("#segmentnumber true prediction \n")
            total = 0
            correct = 0
            clean_total = 0
            clean_correct = 0

            for seq in seqs:
                clean = True
                if 'dB' in seq:
                    clean = False

                nsegs = z1reg_by_seq[seq][0].shape[0]
                f.write('Sequence %s with %i segments \n' % (seq, nsegs))

                for j in range(nsegs):
                    truelab = bReg_by_seq[seq][j, i]

                    truelab = list(talab2idx[name].keys())[list(talab2idx[name].values()).index(truelab)]

                    # no or unknown label
                    if len(truelab) == 0:
                        continue

                    total += 1
                    if clean:
                        clean_total += 1

                    # again + 1 because first label is the unknown label "" (not in z1reg)
                    pred_lab = np.argmax(z1reg_by_seq[seq][i][j, :]) + 1

                    pred_lab = list(talab2idx[name].keys())[list(talab2idx[name].values()).index(pred_lab)]

                    if pred_lab == truelab:
                        correct += 1
                        if clean:
                            clean_correct += 1

                    f.write("\t %i \t %s \t %s \n" % (j, str(truelab), str(pred_lab)))

        accuracies[i] = correct/total

        if clean_total == 0:
            clean_total = 1
        clean_accuracies[i] = clean_correct / clean_total

        with open("%s/test/txt/label_accuracy/%s_acc" % (expdir, name), "w") as fid:
            fid.write("%10.3f \n" % accuracies[i])
        _print("prediction accuracy for labels of class %s is %f  (and %f on clean test set)" % (name, accuracies[i], clean_accuracies[i]), fff)

    return names, accuracies


def compute_advpred_acc_z1(expdir, model, conf, seqs, tt_dset, advregpost_by_seq, z1advreg_by_seq, fff):

    names = conf['facs'].split(':')
    lab2idx = conf['lab2idx']
    accuracies = [0. for _ in range(len(names))]
    # accuracies_z1advreg = [0. for _ in range(len(names))]

    clean_accuracies = [0. for _ in range(len(names))]

    for i, name in enumerate(names):

        ordered_labs = lab2idx[name]
        truelabs = tt_dset.labs_d[name].seq2lab

        total = 0
        correct = 0  # using mu2 as input
        # correct_z1advreg = 0  #using z2 as input

        clean_total = 0
        clean_correct = 0

        with open("%s/test/txt/adversarial_accuracy/%s_predictions.scp" % (expdir, name), "w") as f:
            f.write("#seq truelabel predictedlabel      for class %s \n" % name)
            for seq in seqs:
                # when no or unknown label ""
                if len(truelabs[seq]) == 0:
                    continue

                clean = True
                if 'dB' in seq:
                    clean = False

                total += 1
                if clean:
                    clean_total += 1

                probs = advregpost_by_seq[seq][i]
                # max + 1 since the first label in ordered_labs is the unknown label ""
                pred_lab = ordered_labs[np.argmax(probs) + 1]
                if pred_lab == truelabs[seq]:
                    correct += 1
                    if clean:
                        clean_correct += 1
                f.write(seq + " " + str(truelabs[seq]) + " " + str(pred_lab) + "\n")

                # probs_z1advreg = z1advreg_by_seq[seq][i]
                # pred_lab_z1advreg = ordered_labs[np.argmax(np.sum(probs_z1advreg, axis=0))+1]
                # if pred_lab_z1advreg == truelabs[seq]:
                #    correct_z1advreg += 1

        accuracies[i] = correct / total

        if clean_total == 0:
            clean_total = 1
        clean_accuracies[i] = clean_correct / clean_total

        with open("%s/test/txt/adversarial_accuracy/%s_acc" % (expdir, name), "w") as fid:
            fid.write("%10.3f \n" % accuracies[i])
        _print("adversarial prediction accuracy for labels of class %s is %f (and %f on clean test set)" % (name, accuracies[i], clean_accuracies[i]), fff)

        # accuracies_z1advreg[i] = correct_z1advreg/total
        # _print("adversarial prediction accuracy for labels of class from z1advreg_by_seq %s is %f" % (name, accuracies[i]), fff)

    return names, accuracies


def compute_advpred_acc_z2(expdir, model, conf, seqs, tt_dset, z2advreg_by_seq, bReg_by_seq, fff):

    names = conf['talabs'].split(':')
    talab2idx = conf['train_talab_vals']

    accuracies = [0. for _ in range(len(names))]
    clean_accuracies = [0. for _ in range(len(names))]

    for i, name in enumerate(names):
        with open("%s/test/txt/adversarial_accuracy/%s_predictions.scp" % (expdir, name), "w") as f:
            f.write("#segmentnumber true prediction \n")
            total = 0
            correct = 0

            clean_total = 0
            clean_correct = 0

            for seq in seqs:
                clean = True
                nsegs = z2advreg_by_seq[seq][0].shape[0]
                f.write('Sequence %s with %i segments \n' % (seq, nsegs))

                if 'dB' in seq:
                    clean = False

                for j in range(nsegs):
                    truelab = bReg_by_seq[seq][j, i]

                    truelab = list(talab2idx[name].keys())[list(talab2idx[name].values()).index(truelab)]

                    # no or unknown label
                    if len(truelab) == 0:
                        continue

                    total += 1
                    if clean:
                        clean_total += 1

                    # again + 1 because first label is the unknown label "" (not in z1reg)
                    pred_lab = np.argmax(z2advreg_by_seq[seq][i][j, :]) + 1

                    pred_lab = list(talab2idx[name].keys())[list(talab2idx[name].values()).index(pred_lab)]

                    if pred_lab == truelab:
                        correct += 1
                        if clean:
                            clean_correct += 1

                    f.write("\t %i \t %s \t %s \n" % (j, str(truelab), str(pred_lab)))

        accuracies[i] = correct / total

        if clean_total == 0:
            clean_total = 1
        clean_accuracies[i] = clean_correct / clean_total

        with open("%s/test/txt/adversarial_accuracy/%s_acc" % (expdir, name), "w") as fid:
            fid.write("%10.3f \n" % accuracies[i])
        _print("adversarial prediction accuracy for labels of class %s is %f  (and %f on clean test set)" % (name, accuracies[i], clean_accuracies[i]), fff)

    return names, accuracies


def visualize_reg_vals(expdir, model, seqs, tt_dset, conf, z1_by_seq, z2_by_seq, mu2_by_seq, regpost_by_seq, xin_by_seq, xout_by_seq, xoutv_by_seq, z1reg_by_seq, z1_0_by_seq, z2_0_by_seq, z1_T_by_seq, z2_T_by_seq, mu2_by_lab, fff):

    if True:
        # names = ["region", "gender"]
        names = conf['facs'].split(':')
        for i, name in enumerate(names):
            with open("%s/test/txt/label_accuracy/%s.scp" % (expdir, name), "w") as f:
                for seq in seqs:
                    f.write(seq + "  [ ")
                    for e in np.nditer(regpost_by_seq[seq][i]):
                        f.write("%10.3f " % e)
                    f.write("]\n")

    if 'cherrypick' in conf:
        if conf['cherrypick']:
            _print('Using sequences in "%s" for visualization' % conf['cherrypick'], fff)
            seqs = []
            with open(conf['cherrypick']) as f:
                for line in f:
                    seqs.append(line.split()[0].strip())
    else:
        _print("No cherrypick file provided. Using 5 random sequences for visualization", fff)
        seqs = sorted(list(np.random.choice(seqs, 5, replace=False)))
        seq_names = ["%02d_%s" % (i, seq) for i, seq in enumerate(seqs)]
        _print(('seq_names: ', seq_names), fff)


    if True:
        # visualize reconstruction
        _print("visualizing reconstruction", fff)
        plot_x([xin_by_seq[seq] for seq in seqs], seq_names, "%s/test/img/xin.png" % expdir)
        plot_x([xout_by_seq[seq] for seq in seqs], seq_names, "%s/test/img/xout.png" % expdir)
        plot_x([xoutv_by_seq[seq] for seq in seqs], seq_names,
               "%s/test/img/xout_logvar.png" % expdir, clim=(None, None))

    if True:
        # factorization: use the centered segment from each sequence
        _print("visualizing factorization", fff)
        cen_z1 = np.array([z1_by_seq[seq][(np.floor(len(z1_by_seq[seq])/2)).astype(int), :] for seq in seqs])
        cen_z2 = np.array([z2_by_seq[seq][(np.floor(len(z2_by_seq[seq])/2)).astype(int), :] for seq in seqs])
        xfac = []
        for z1 in cen_z1:
            z1 = np.tile(z1, (len(cen_z2), 1))
            _, _, _, _, px_z = model.decoder(\
                np.zeros((len(z1), conf['tr_shape'][0], conf['tr_shape'][1]), dtype=np.float32), z1, cen_z2)
            xfac.append(px_z[0])
        plot_x(xfac, seq_names, "%s/test/img/xfac.png" % expdir, sep=True)

    if True:
        _print("writing out reconstruction", fff)
        # Maybe use dataset instead of dataset_test?
        with open(os.path.join(conf['datadir'], conf['dataset'], 'train', 'mvn.pkl'), "rb") as f:
            mvn_params = pickle.load(f)
        nb_mel = mvn_params["mean"].size
        for src_seq, src_seq_name in zip(seqs, seq_names):
            with open("%s/test/spec/reconstruction/xin_%s.npy" % (expdir, src_seq), "wb") as fnp:
                np.save(fnp, np.reshape(xin_by_seq[src_seq], (-1, nb_mel)) * mvn_params["std"] + mvn_params["mean"])
            with open("%s/test/spec/reconstruction/xout_%s.npy" % (expdir, src_seq), "wb") as fnp:
                np.save(fnp,
                        np.reshape(xout_by_seq[src_seq], (-1, nb_mel)) * mvn_params["std"] + mvn_params["mean"])

    if True:  # sequence neutralisation
        _print("visualizing neutral sequences", fff)
        neu_by_seq = dict()
        with open("%s/test/spec/mean_mu2.npy" % expdir, "rb") as fnp:
            mumu = np.float32(np.load(fnp))  # (32,)
        for src_seq, src_seq_name in zip(seqs, seq_names):
            #mu2_by_seq (1,32) en del_mu2
            del_mu2 = mumu - mu2_by_seq[src_seq]
            src_z1, src_z2 = z1_by_seq[src_seq], z2_by_seq[src_seq]
            # src_z1 en z2 vb (28x32), del_mu2 (1,32)
            neu_by_seq[src_seq] = _seq_translate(
                model, conf['tr_shape'], src_z1, src_z2, del_mu2)
            with open("%s/test/spec/neutral/neu_%s.npy" % (expdir, src_seq), "wb") as fnp:
                np.save(fnp, np.reshape(neu_by_seq[src_seq], (-1, nb_mel)) * mvn_params["std"] + mvn_params["mean"])

        plot_x([neu_by_seq[seq] for seq in seqs], seq_names,
               "%s/test/img/neutral.png" % expdir, False)

    if True:
        # sequence translation
        _print("visualizing sequence translation", fff)
        xtra_by_seq = dict()
        for src_seq, src_seq_name in zip(seqs, seq_names):
            xtra_by_seq[src_seq] = dict()
            src_z1, src_z2 = z1_by_seq[src_seq], z2_by_seq[src_seq]
            for tar_seq in seqs:
                del_mu2 = mu2_by_seq[tar_seq] - mu2_by_seq[src_seq]
                xtra_by_seq[src_seq][tar_seq] = _seq_translate(
                    model, conf['tr_shape'], src_z1, src_z2, del_mu2)
                with open("%s/test/spec/x_tra/src_%s_tar_%s.npy" % (expdir, src_seq, tar_seq), "wb") as fnp:
                    np.save(fnp, np.reshape(xtra_by_seq[src_seq][tar_seq], (-1, nb_mel)) * mvn_params["std"] + mvn_params["mean"])

            plot_x([xtra_by_seq[src_seq][seq] for seq in seqs], seq_names,
                   "%s/test/img/x_tra/%s_tra.png" % (expdir, src_seq_name), True)

        # Write html file for easier comparison
        with open(os.path.join(expdir, 'test', 'wav', 'index.html'), "w+") as file:
            file.write(r'<html><head><style>table{margin:auto}audio{width:150px;display:block;}td.diag{background:pink}</style></head><body><table>')
            for src_seq in seqs:
                file.write('<tr>\n')
                for tar_seq in seqs:
                    if src_seq == tar_seq:
                        file.write('<td class="diag">\n')
                    else:
                        file.write('<td>')
                    file.write('<audio controls src="src_%s_tar_%s.wav"></audio></td>\n' % (src_seq, tar_seq) )
                file.write('</tr>\n')
            file.write('</table>\n</body>\n</html>\n')

    if True:
        # label shift
        _print("visualizing label translation", fff)
        for src_seq, src_seq_name in zip(seqs, seq_names):
            src_z1, src_z2 = z1_by_seq[src_seq], z2_by_seq[src_seq]
            names = conf['facs'].split(':')
            lab2idx= conf['lab2idx']
            # Iterate over all regularization labels
            for name in names:
                src_lab = tt_dset.labs_d[name].seq2lab[src_seq]
                # Iterate over all label values
                for tar_lab in mu2_by_lab[name].keys():
                    # Delta mu is distance between two label means
                    del_mu2 = mu2_by_lab[name][tar_lab] - mu2_by_lab[name][src_lab]
                    x_tra = _seq_translate(
                        model, conf['tr_shape'], src_z1, src_z2, del_mu2)
                    with open("%s/test/spec/tra_lab/%s_%s_src_%s_tar_%s.npy" % (expdir, src_seq, name, src_lab, tar_lab), "wb") as fnp:
                        np.save(fnp, np.reshape(x_tra, (-1, nb_mel)) * mvn_params["std"] +
                                mvn_params["mean"])

    if True:
        # tsne z1 and z2
        _print("t-SNE analysis on latent variables (over entire sequences)", fff)
        n = [len(z1_by_seq[seq]) for seq in seqs]
        z1 = np.concatenate([z1_by_seq[seq] for seq in seqs], axis=0)
        z2 = np.concatenate([z2_by_seq[seq] for seq in seqs], axis=0)

        p = 30
        _print("  perplexity = %s" % p, fff)
        tsne = TSNE(n_components=2, verbose=0, perplexity=p, n_iter=1000)
        z1_tsne = _unflatten(tsne.fit_transform(z1), n)
        scatter_plot(z1_tsne, seq_names, "z1_tsne_%03d" % p,
                     "%s/test/img/z1_tsne_%03d.png" % (expdir, p))
        z2_tsne = _unflatten(tsne.fit_transform(z2), n)
        scatter_plot(z2_tsne, seq_names, "z2_tsne_%03d" % p,
                     "%s/test/img/z2_tsne_%03d.png" % (expdir, p))

    if 'num_flow_steps' in conf:
        if conf['num_flow_steps'] > 0:

            _print("Plotting t-sne of samples before and after flow", fff)

            T = int(conf['num_flow_steps'])

            n = [len(z1_0_by_seq[seq]) for seq in seqs]
            z1_0 = np.concatenate([z1_0_by_seq[seq] for seq in seqs], axis=0)
            z2_0 = np.concatenate([z2_0_by_seq[seq] for seq in seqs], axis=0)
            z1_T = np.concatenate([z1_T_by_seq[seq] for seq in seqs], axis=0)
            z2_T = np.concatenate([z2_T_by_seq[seq] for seq in seqs], axis=0)

            p = 30
            _print("  perplexity = %s" % p, fff)
            tsne = TSNE(n_components=2, verbose=0, perplexity=p, n_iter=1000)
            z1_0_tsne = _unflatten(tsne.fit_transform(z1_0), n)
            scatter_plot(z1_0_tsne, seq_names, "z1_tsne_%03d" % p,
                         "%s/test/img/flow/0_steps_z1_tsne_%03d.png" % (expdir, p))
            z2_0_tsne = _unflatten(tsne.fit_transform(z2_0), n)
            scatter_plot(z2_0_tsne, seq_names, "z2_tsne_%03d" % p,
                         "%s/test/img/flow/0_steps_z2_tsne_%03d.png" % (expdir, p))

            tsne = TSNE(n_components=2, verbose=0, perplexity=p, n_iter=1000)
            z1_T_tsne = _unflatten(tsne.fit_transform(z1_T), n)
            scatter_plot(z1_T_tsne, seq_names, "z1_tsne_%03d" % p,
                         "%s/test/img/flow/%i_steps_z1_tsne_%03d.png" % (expdir, T, p))
            z2_T_tsne = _unflatten(tsne.fit_transform(z2_T), n)
            scatter_plot(z2_T_tsne, seq_names, "z2_tsne_%03d" % p,
                         "%s/test/img/flow/%i_steps_z2_tsne_%03d.png" % (expdir, T, p))


def tsne_by_label(expdir, model, conf, create_test_dataset, seqs, tt_dset, bReg_by_seq, z1_by_phone, fff):

    if len(seqs) > 25:
        seqs = sorted(list(np.random.choice(seqs, 25, replace=False)))

    # infer z1, z2
    z1_by_seq = defaultdict(list)
    z2_by_seq = defaultdict(list)
    for seq in seqs:

        Test_Dataset = create_test_dataset([seq])

        for _, x, _, _, _, _ in Test_Dataset:
            _, _, _, _, _, _, _, _, qz1_x, qz2_x = model.encoder(x)
            z2_by_seq[seq].append(qz2_x[0])
            z1_by_seq[seq].append(qz1_x[0])

        z1_by_seq[seq] = np.concatenate(z1_by_seq[seq], axis=0)
        z2_by_seq[seq] = np.concatenate(z2_by_seq[seq], axis=0)

    # tsne z1 and z2
    _print("t-SNE analysis on latent variables by label", fff)
    n = [len(z1_by_seq[seq]) for seq in seqs]
    z1 = np.concatenate([z1_by_seq[seq] for seq in seqs], axis=0)  # (379x32)
    z2 = np.concatenate([z2_by_seq[seq] for seq in seqs], axis=0)

    p = 30
    tsne = TSNE(n_components=2, verbose=0, perplexity=p, n_iter=1000)
    z1_tsne_by_seq = dict(list(zip(seqs, _unflatten(tsne.fit_transform(z1), n))))  # dict met voor elke sequence de unflattened tsne fit transform (vb (27x2))

    for gen_fac, seq2lab in list(tt_dset.labs_d.items()):  # gender
        _labs, _z1 = _join(z1_tsne_by_seq, seq2lab)  # seq2lab: dict met seq en label (vb 'f')
        # labs = ['f', 'm'] list
        # z1 = list of 2 arrays, both 165x2
        scatter_plot(_z1, _labs, gen_fac,
                     "%s/test/img/tsne_by_label_z1_%s_%03d.png" % (expdir, gen_fac, p))

    z2_tsne_by_seq = dict(list(zip(seqs, _unflatten(tsne.fit_transform(z2), n))))
    for gen_fac, seq2lab in list(tt_dset.labs_d.items()):
        _labs, _z2 = _join(z2_tsne_by_seq, seq2lab)
        scatter_plot(_z2, _labs, gen_fac,
                     "%s/test/img/tsne_by_label_z2_%s_%03d.png" % (expdir, gen_fac, p))

    for gen_talab, seq2talabseq in list(tt_dset.talabseqs_d_new.items()):
        idx = list(conf['b_n'].keys()).index(gen_talab)
        _talabs, _z1 = _join_talab(z1_tsne_by_seq, bReg_by_seq, tt_dset.talab_vals[gen_talab], idx)
        #talabs: vb ['SIL, SON, 'OBS']
        # z1: vb list van 3 arrays, elk (96x2)
        scatter_plot(_z1, _talabs, gen_talab,
                     "%s/test/img/tsne_by_label_z1_%s_%03d.png" % (expdir, gen_talab, p))

    for gen_talab, seq2talabseq in list(tt_dset.talabseqs_d_new.items()):
        idx = list(conf['b_n'].keys()).index(gen_talab)
        _talabs, _z2 = _join_talab(z2_tsne_by_seq, bReg_by_seq, tt_dset.talab_vals[gen_talab], idx)
        scatter_plot(_z2, _talabs, gen_talab,
                     "%s/test/img/tsne_by_label_z2_%s_%03d.png" % (expdir, gen_talab, p))

    # for gen_talab, seq2talabseq in list(tt_dset.talabseqs_d.items()):
    #     _talabs, _z2 = _join_talab(z2_tsne_by_seq, seq2talabseq.seq2talabseq, tt_dset.talab_vals[gen_talab])
    #     scatter_plot(_z2, _talabs, gen_talab,
    #                  "%s/test/img/tsne_by_label_z2_%s_%03d.png" % (expdir, gen_talab, p))


    # plot all z1s of some phones
    phones = list(z1_by_phone.keys())
    if len(phones) > 5:
        phones = sorted(list(np.random.choice(phones, 5, replace=False)))

    n = [len(z1_by_phone[phon]) for phon in phones]
    z1_phon = np.concatenate([z1_by_phone[phon] for phon in phones], axis=0)
    z1_tsne_by_phone = dict(list(zip(phones, _unflatten(tsne.fit_transform(z1_phon), n))))

    _z1_phon = [z1_tsne_by_phone[phon] for phon in phones]
    scatter_plot(_z1_phon, phones, 'phonelist',
                 "%s/test/img/tsne_of_z1_by_phone_%03d.png" % (expdir, p))


def estimate_mu1_mu2_dict(model, dataset, mu1_table, mu2_table, nphones, nsegs, tr_shape):

    for yval, xval, _, _, talab, _ in dataset:
        z1_mu, _, _, _, z2_mu, _, _, _, _, _ = model.encoder(tf.reshape(xval, [-1, tr_shape[0], tr_shape[1]]))

        phon_vecs = tf.one_hot(talab[:, 0], depth=nphones.shape[0], axis=0, dtype=tf.float32)
        mu1_table += tf.matmul(phon_vecs, z1_mu)
        nphones += tf.reduce_sum(phon_vecs, axis=1)

        y_br = tf.one_hot(yval, depth=nsegs.shape[0], axis=0, dtype=tf.float32)
        mu2_table += tf.matmul(y_br, z2_mu)
        nsegs += tf.reduce_sum(y_br, axis=1)

    r_mu1 = tf.constant(np.exp(model.pz1_stddev ** 2) / np.exp(model.pmu1_stddev ** 2), dtype=tf.float32)
    denom_mu1 = nphones + tf.math.scalar_mul(r_mu1, tf.ones_like(nphones))
    mu1_table = tf.divide(mu1_table, tf.expand_dims(denom_mu1, axis=-1))

    r_mu2 = tf.constant(np.exp(model.pz2_stddev ** 2) / np.exp(model.pmu2_stddev ** 2), dtype=tf.float32)
    denom_mu2 = nsegs + tf.math.scalar_mul(r_mu2, tf.ones_like(nsegs))
    mu2_table = tf.divide(mu2_table, tf.expand_dims(denom_mu2, axis=-1))

    return mu1_table, mu2_table, nphones


def _print_mu_stat(mu_table, name, fff):
    # mu2 is no longer dictionary, but a Tensor
    norm_sum = 0.
    dim_norm_sum = 0.
    nseqs = mu_table.get_shape().as_list()[0]
    for y in range(0, nseqs):
        norm_sum += np.linalg.norm(mu_table[y, :])
        dim_norm_sum += np.abs(mu_table[y, :])
    avg_norm = norm_sum / nseqs
    avg_dim_norm = dim_norm_sum / nseqs

    _print('%s-table stats' % name, fff)
    _print("avg. norm = %.2f, #%s = %s" % (avg_norm, name, nseqs), fff)
    _print("per dim: %s" % (" ".join(["%.2f" % v for v in avg_dim_norm]),), fff)

    return [avg_norm, avg_dim_norm, nseqs]


def _softmax(x):
    ## First column are zeros (as added in fix_logits in model, so leave these out and return size-1 tens
    y = np.exp(x[:, 1:])
    return y / np.sum(y, axis=1, keepdims=True)
    # return tf.nn.softmax(x, axis=1)


def _seq_translate(model, tr_shape, src_z1, src_z2, del_mu2):
    # mod_z2: (1,28,32)
    # del_mu2: (1,32) --> del_mu2[np.newaxis, ...] is (1,32,32)
    # src_z2: (28,32)
    # src_z1: (28,32)
    mod_z2 = src_z2 + del_mu2  # [np.newaxis, ...]
    _, _, _, _, px_z = model.decoder(\
        np.zeros((len(src_z1), tr_shape[0], tr_shape[1]), dtype=np.float32), src_z1, mod_z2)

    return px_z[0]


def _unflatten(l_flat, n_l):
    """
    unflatten a list
    """
    l = []
    offset = 0
    for n in n_l:
        l.append(l_flat[offset:offset+n])
        offset += n
    assert(offset == len(l_flat))
    return l


def _join(z_by_seqs, seq2lab):
    d = defaultdict(list)
    for seq, z in list(z_by_seqs.items()):
        d[seq2lab[seq]].append(z)
    for lab in d:
        d[lab] = np.concatenate(d[lab], axis=0)
    return list(d.keys()), list(d.values())


def _join_talab(z_by_seqs, xReg_by_seq, talab_vals, idx):
    d = defaultdict(list)
    for seq, z in list(z_by_seqs.items()):
        n_segs = z.shape[0]
        xReg = xReg_by_seq[seq]

        for seg in range(n_segs):
            talab = xReg[seg, idx]
            talab = list(talab_vals.keys())[list(talab_vals.values()).index(talab)]
            d[talab].append(z[seg, :])
    for lab in d:
        d[lab] = np.stack(d[lab], axis=0)

    return list(d.keys()), list(d.values())


def _variance(x):
    if len(x) == 0:
        return 0.0
    else:
        return np.square(x - x.mean(axis=0)).sum(axis=1).mean().astype(float)


def _dump(expdir, z1_by_seq, z2_by_seq, mu2_by_seq, regpost_by_seq, xin_by_seq, xout_by_seq, xoutv_by_seq, z1reg_by_seq, regpost_by_seq_z1, z2reg_by_seq, bReg_by_seq, cReg_by_seq, z1advreg_by_seq, z2advreg_by_seq, advregpost_by_seq, mu1_by_seq, z1_0_by_seq, z2_0_by_seq, z1_T_by_seq, z2_T_by_seq, z1_by_phone, mu1_by_phone, z1_by_phone_and_lab, mu2_by_lab, fff):

    if True:
        # latent vars and svectors
        with open(os.path.join(expdir, 'test', 'var_dicts', 'lvar', 'z1_by_seq.pkl'), 'wb') as f:
            pickle.dump(z1_by_seq, f)
        with open(os.path.join(expdir, 'test', 'var_dicts', 'lvar', 'z2_by_seq.pkl'), 'wb') as f:
            pickle.dump(z2_by_seq, f)
        with open(os.path.join(expdir, 'test', 'var_dicts', 'lvar', 'mu1_by_seq.pkl'), 'wb') as f:
            pickle.dump(mu1_by_seq, f)
        with open(os.path.join(expdir, 'test', 'var_dicts', 'lvar', 'mu2_by_seq.pkl'), 'wb') as f:
            pickle.dump(mu2_by_seq, f)
        with open(os.path.join(expdir, 'test', 'var_dicts', 'lvar', 'mu2_by_lab.pkl'), 'wb') as f:
            pickle.dump(mu2_by_lab, f)
        with open(os.path.join(expdir, 'test', 'var_dicts', 'lvar', 'mu1_by_phone.pkl'), 'wb') as f:
            pickle.dump(mu1_by_phone, f)
        with open(os.path.join(expdir, 'test', 'var_dicts', 'lvar', 'z1_by_phone.pkl'), 'wb') as f:
            pickle.dump(z1_by_phone, f)
        with open(os.path.join(expdir, 'test', 'var_dicts', 'lvar', 'z1_by_phone_and_lab.pkl'), 'wb') as f:
            pickle.dump(z1_by_phone_and_lab, f)

    if True:
        # reconstructions
        with open(os.path.join(expdir, 'test', 'var_dicts', 'rec', 'xin_by_seq.pkl'), 'wb') as f:
            pickle.dump(xin_by_seq, f)
        with open(os.path.join(expdir, 'test', 'var_dicts', 'rec', 'xout_by_seq.pkl'), 'wb') as f:
            pickle.dump(xout_by_seq, f)
        with open(os.path.join(expdir, 'test', 'var_dicts', 'rec', 'xoutv_seq.pkl'), 'wb') as f:
            pickle.dump(xoutv_by_seq, f)

    if True:
        # regularisations
        with open(os.path.join(expdir, 'test', 'var_dicts', 'reg', 'regpost_by_seq.pkl'), 'wb') as f:
            pickle.dump(regpost_by_seq, f)
        with open(os.path.join(expdir, 'test', 'var_dicts', 'reg', 'regpost_by_seq_z1.pkl'), 'wb') as f:
            pickle.dump(regpost_by_seq_z1, f)
        with open(os.path.join(expdir, 'test', 'var_dicts', 'reg', 'z1reg_by_seq.pkl'), 'wb') as f:
            pickle.dump(z1reg_by_seq, f)
        with open(os.path.join(expdir, 'test', 'var_dicts', 'reg', 'z2reg_by_seq.pkl'), 'wb') as f:
            pickle.dump(z2reg_by_seq, f)

    if True:
        # labels
        with open(os.path.join(expdir, 'test', 'var_dicts', 'lab', 'bReg_by_seq.pkl'), 'wb') as f:
            pickle.dump(bReg_by_seq, f)
        with open(os.path.join(expdir, 'test', 'var_dicts', 'lab', 'cReg_by_seq.pkl'), 'wb') as f:
            pickle.dump(cReg_by_seq, f)

    if True:
        # adversarial regularisations
        with open(os.path.join(expdir, 'test', 'var_dicts', 'adv', 'z1advreg_by_seq.pkl'), 'wb') as f:
            pickle.dump(z1advreg_by_seq, f)
        with open(os.path.join(expdir, 'test', 'var_dicts', 'adv', 'z2advreg_by_seq.pkl'), 'wb') as f:
            pickle.dump(z2advreg_by_seq, f)
        with open(os.path.join(expdir, 'test', 'var_dicts', 'adv', 'advregpost_by_seq.pkl'), 'wb') as f:
            pickle.dump(advregpost_by_seq, f)

    if True:
        # normalising flows
        with open(os.path.join(expdir, 'test', 'var_dicts', 'flow', 'z1_0_by_seq.pkl'), 'wb') as f:
            pickle.dump(z1_0_by_seq, f)
        with open(os.path.join(expdir, 'test', 'var_dicts', 'flow', 'z2_0_by_seq.pkl'), 'wb') as f:
            pickle.dump(z2_0_by_seq, f)
        with open(os.path.join(expdir, 'test', 'var_dicts', 'flow', 'z1_T_by_seq.pkl'), 'wb') as f:
            pickle.dump(z1_T_by_seq, f)
        with open(os.path.join(expdir, 'test', 'var_dicts', 'flow', 'z2_T_by_seq.pkl'), 'wb') as f:
            pickle.dump(z2_T_by_seq, f)


def _init_dirs(expdir):
    os.makedirs(os.path.join(expdir, 'test', 'img', 'x_tra'), exist_ok=True)
    os.makedirs(os.path.join(expdir, 'test', 'img', 'flow'), exist_ok=True)
    os.makedirs(os.path.join(expdir, 'test', 'spec', 'tra_lab'), exist_ok=True)
    os.makedirs(os.path.join(expdir, 'test', 'spec', 'x_tra'), exist_ok=True)
    os.makedirs(os.path.join(expdir, 'test', 'spec', 'reconstruction'), exist_ok=True)
    os.makedirs(os.path.join(expdir, 'test', 'spec', 'neutral'), exist_ok=True)
    os.makedirs(os.path.join(expdir, 'test', 'txt', 'label_accuracy'), exist_ok=True)
    os.makedirs(os.path.join(expdir, 'test', 'txt', 'adversarial_accuracy'), exist_ok=True)
    os.makedirs(os.path.join(expdir, 'test', 'wav'), exist_ok=True)
    os.makedirs(os.path.join(expdir, 'test', 'var_dicts', 'lvar'), exist_ok=True)
    os.makedirs(os.path.join(expdir, 'test', 'var_dicts', 'reg'), exist_ok=True)
    os.makedirs(os.path.join(expdir, 'test', 'var_dicts', 'rec'), exist_ok=True)
    os.makedirs(os.path.join(expdir, 'test', 'var_dicts', 'adv'), exist_ok=True)
    os.makedirs(os.path.join(expdir, 'test', 'var_dicts', 'flow'), exist_ok=True)
    os.makedirs(os.path.join(expdir, 'test', 'var_dicts', 'lab'), exist_ok=True)


def _print(cmd, fileloc):
    print(cmd)
    with open(fileloc, 'a') as ppp:
        print(cmd, file=ppp)
