from __future__ import division
import os
import sys
import time
import pickle
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
from sklearn.manifold import TSNE
import tensorflow as tf
from .plotter import plot_x, scatter_plot
from fhvae.datasets.seq2seg import seq_to_seg_mapper

np.random.seed(123)

def test_reg(expdir, model, conf, tt_iterator_by_seqs, tt_seqs, tt_dset):
    '''
    Compute variational lower bound
    '''

    print("\nRESTORING MODEL")
    starttime = time.time()
    optimizer = tf.keras.optimizers.Adam(learning_rate=conf['lr'], beta_1=conf['beta1'], beta_2=conf['beta2'], epsilon=conf['adam_eps'], amsgrad=False)
    checkpoint_directory = os.path.join(expdir, 'training_checkpoints')
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

    if os.path.exists(os.path.join(checkpoint_directory, 'best_checkpoint')):
        with open(os.path.join(checkpoint_directory, 'best_checkpoint'), 'r') as pid:
            best_checkpoint = (pid.readline()).rstrip()
        status = checkpoint.restore(os.path.join(checkpoint_directory, 'ckpt-' + str(best_checkpoint)))
    else:
        manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_directory, max_to_keep=5)
        status = checkpoint.restore(manager.latest_checkpoint)

    print("restoring model takes %.2f seconds" % (time.time()-starttime))
    status.assert_existing_objects_matched()
    #status.assert_consumed()

    os.makedirs(os.path.join(expdir, 'test', 'img', 'x_tra'), exist_ok=True)
    os.makedirs(os.path.join(expdir, 'test', 'spec'), exist_ok=True)
    os.makedirs(os.path.join(expdir, 'test', 'txt'), exist_ok=True)

    seg_len = tf.constant(tt_dset.seg_len, tf.int64)
    seg_shift = tf.constant(tt_dset.seg_shift, tf.int64)

    def segment_map_fn_test(idx, data):
        # map every sequence to <variable #> segments
        # + use enumeration-number as sequence key such that all keys in range [0, nmu2]
        keys, feats, lens, labs, talabs = data
        keys, feats, lens, labs, talabs, starts = seq_to_seg_mapper(idx, feats, lens, labs, talabs, \
                                                            seg_len, seg_shift, False,
                                                            len(conf['b_n']))
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


    print("\nCOMPUTING AVERAGE VALUES")
    avg_loss, avg_vals = compute_average_values(model, tt_seqs, conf, create_test_dataset)

    print("\nCOMPUTING VALUES BY PHONE CLASS")
    z1_by_phone, mu1_by_phone = compute_values_by_phone(model, conf, create_test_dataset, tt_seqs, expdir)

    print("\nCOMPUTING VALUES BY SEQUENCE")
    z1_by_seq, z2_by_seq, mu2_by_seq, regpost_by_seq, xin_by_seq, xout_by_seq, xoutv_by_seq, z1reg_by_seq, regpost_by_seq_z1, \
        z2reg_by_seq, bReg_by_seq, cReg_by_seq = compute_values_by_seq(model, conf, create_test_dataset, tt_seqs, expdir)

    print("\nCOMPUTING PREDICTION ACCURACIES FOR LABELS FROM Z2")
    labels, accs = compute_pred_acc_z2(expdir, model, conf, tt_seqs, tt_dset, regpost_by_seq, z2reg_by_seq, cReg_by_seq)

    print("\nCOMPUTING PREDICTION ACCURACIES FOR TIME ALIGNED LABELS FROM Z1")
    labels, accs = compute_pred_acc_z1(expdir, model, conf, tt_seqs, tt_dset, regpost_by_seq, z1reg_by_seq, bReg_by_seq)

    print("\nVISUALIZING RESULTS")
    visualize_reg_vals(expdir, model, tt_seqs, conf, z1_by_seq, z2_by_seq, mu2_by_seq, regpost_by_seq, xin_by_seq, \
                       xout_by_seq, xoutv_by_seq, z1reg_by_seq)

    print("\nVISUALIZING TSNE BY LABEL")
    tsne_by_label(expdir, model, conf, create_test_dataset, tt_seqs, tt_dset, bReg_by_seq, z1_by_phone)

    print("\nFINISHED\nResults stored in %s/test" % expdir)


def compute_average_values(model, tt_seqs, conf, create_test_dataset):
    num_seqs = len(tt_seqs)
    Test_Dataset = create_test_dataset(seqs=tt_seqs)

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

    sum_names = ['log_pmu2', 'log_pmu1', 'neg_kld_z2', 'neg_kld_z1', 'log_px_z', 'lb', 'log_qy_mu2', 'log_qy_mu1', 'log_b_loss', 'log_c_loss']
    sum_loss = 0.
    sum_vals = [0. for _ in range(len(sum_names))]
    tot_segs = 0.
    avg_vals = [0. for _ in range(len(sum_names))]

    for yval, xval, nval, cval, bval, _ in Test_Dataset:
        nval = tf.cast(nval, dtype=tf.float32)

        mu2, mu1, qz2_x, z2_sample, qz1_x, z1_sample, px_z, x_sample, z1_rlogits, z2_rlogits = model(xval, yval, bval[:, 0])

        loss, log_pmu2, log_pmu1, neg_kld_z2, neg_kld_z1, log_px_z, lb, log_qy_mu2, log_qy_mu1, log_b_loss, log_c_loss = \
            model.compute_loss(xval, yval, nval, bval, cval, mu2, mu1, qz2_x, z2_sample, qz1_x, z1_sample, px_z, x_sample, z1_rlogits, z2_rlogits, num_seqs)

        results = [log_pmu2, log_pmu1, neg_kld_z2, neg_kld_z1, log_px_z, lb, log_qy_mu2, log_qy_mu1, log_b_loss, log_c_loss]
        for i in range(len(sum_vals)):
            sum_vals[i] += -tf.reduce_mean(results[i])
        sum_loss += loss
        tot_segs += len(xval)

    avg_loss = sum_loss/tot_segs
    print("average loss = %f" % (avg_loss))
    for i in range(len(sum_vals)):
        avg_vals[i] = sum_vals[i]/tot_segs
        print("\t average value for %s \t= %f" % (sum_names[i], avg_vals[i]))

    return avg_loss, avg_vals

def compute_values_by_phone(model, conf, create_test_dataset, seqs, expdir):

    z1_by_phone = defaultdict(list)
    covar_by_phone = {'cov_det':{}, 'diagcov_det':{}, 'cov_logdet':{}, 'diagcov_logdet':{}}
    diagcovdet_by_phone = dict()
    phone_occs = defaultdict(int)
    mu1_by_phone = dict()

    complete_Test_Dataset = create_test_dataset(seqs=seqs)

    for _, x, _, _, b, _ in complete_Test_Dataset:
        _, _, _, _, _, _, qz1_x, _ = model.encoder(x)
        z1_mu = qz1_x[0]
        for idx in range(0, int(b.get_shape().as_list()[0])):
            ph = str(b[idx, 0].numpy())
            z1_by_phone[ph].append(z1_mu[idx, :])
            phone_occs[ph] += 1

    # compute within-class covariance matrix and compute the determinant of this covariance matrix
    # + the determinant of the diagonal of this matrix
    # + logdeterminants
    for ph, all_z1s in z1_by_phone.items():
        cov_mat = np.cov(np.stack(all_z1s, axis=0), rowvar=False)
        covar_by_phone['cov_det'][ph] = np.linalg.det(cov_mat)
        covar_by_phone['diagcov_det'][ph] = np.prod(np.diag(cov_mat))
        (sign, logdet) = np.linalg.slogdet(cov_mat)
        covar_by_phone['cov_logdet'][ph] = logdet
        (sign, logdet) = np.linalg.slogdet(np.diag(np.diag(cov_mat)))
        covar_by_phone['diagcov_logdet'][ph] = logdet

    print('Average determinant of covariance matrix per phone is %f'
          % np.average(list(covar_by_phone['cov_det'].values())))
    print('Average determinant of diagonal covariance matrix per phone is %f'
          % np.average(list(covar_by_phone['diagcov_det'].values())))
    print('Average log-determinant of covariance matrix per phone is %f'
          % np.average(list(covar_by_phone['cov_logdet'].values())))
    print('Average log-determinant of diagonal covariance matrix per phone is %f'
          % np.average(list(covar_by_phone['diagcov_logdet'].values())))

    # posterior map estimation of mu1 according to phone class
    r = np.exp(model.pz1_stddev ** 2) / np.exp(model.pmu1_stddev ** 2)
    for ph, all_z1s in z1_by_phone.items():
        mu1_by_phone[ph] = np.sum(all_z1s, axis=0) / (phone_occs[ph] + r)

    return z1_by_phone, mu1_by_phone

def compute_values_by_seq(model, conf, create_test_dataset, seqs, expdir):

    z1_by_seq = defaultdict(list)
    z2_by_seq = defaultdict(list)
    mu2_by_seq = dict()
    regpost_by_seq = dict()
    xin_by_seq = defaultdict(list)
    xout_by_seq = defaultdict(list)
    xoutv_by_seq = defaultdict(list)
    z1reg_by_seq = defaultdict(list)
    z2reg_by_seq = defaultdict(list)
    regpost_by_seq_z1 = dict()
    bReg_by_seq = defaultdict(list)
    cReg_by_seq = defaultdict(list)

    for seq in seqs:

        Test_Dataset = create_test_dataset(seqs=[seq])

        for _, x, _, c, b, _ in Test_Dataset:

            _, _, _, _, _, _, qz1_x, qz2_x = model.encoder(x)
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

            cReg_by_seq[seq].append(c)
            bReg_by_seq[seq].append(b)

        z1_by_seq[seq] = np.concatenate(z1_by_seq[seq], axis=0)
        z2_by_seq[seq] = np.concatenate(z2_by_seq[seq], axis=0)
        xin_by_seq[seq] = np.concatenate(xin_by_seq[seq], axis=0)
        xout_by_seq[seq] = np.concatenate(xout_by_seq[seq], axis=0)
        xoutv_by_seq[seq] = np.concatenate(xoutv_by_seq[seq], axis=0)
        # z1reg_by_seq[seq] = np.concatenate(z1reg_by_seq[seq], axis=0)

        bReg_by_seq[seq] = np.concatenate(bReg_by_seq[seq], axis=0)
        cReg_by_seq[seq] = np.concatenate(cReg_by_seq[seq], axis=0)


        # formula for inferring S-vector mu2 during testing, paper p5 (over all segments from same sequence)
        z2_sum = np.sum(z2_by_seq[seq], axis=0)
        n = len(z2_by_seq[seq])
        r = np.exp(model.pz2_stddev ** 2) / np.exp(model.pmu2_stddev ** 2)
        mu2_by_seq[seq] = z2_sum / (n+r)

        d2 = mu2_by_seq[seq].shape[0]
        z2 = np.asarray(mu2_by_seq[seq]).reshape([1, d2])

        # probabilities of each of the regularisation classes given the computed z2 of above
        _, z2_rlogits = model.regulariser(z1_by_seq[seq], z2)
        regpost_by_seq[seq] = list(map(_softmax, z2_rlogits))

        # formula for inferring alternative S-vector mu1 during testing, paper p7
        z1_sum = np.sum(z1_by_seq[seq], axis=0)
        n = len(z1_by_seq[seq])
        r = np.exp(model.pz1_stddev ** 2)
        z1 = z1_sum / (n+r)
        z1 = np.asarray(z1).reshape([1, z1.shape[0]])

        # probabilities given computed mu1
        z1_rlogits, _ = model.regulariser(z1, z2_by_seq[seq])
        # softmax over columns 1:end, first column is for unlabeled data
        regpost_by_seq_z1[seq] = list(map(_softmax, z1_rlogits))

    # # save the mu2
    # with open(os.path.join(expdir, 'test', 'mu2_by_seq.txt'),"w"):
    #     for seq in seqs:
    #         f.write( ' '.join (map(str,mu2_by_seq[seq])) )
    #         f.write('\n')

    # save the mean mu2
    if not os.path.exists(os.path.join(expdir, 'test', 'mu2_by_seq.npy')):
        mumu = np.zeros([mu2_by_seq[seqs[1]].size])
        for seq in seqs:
            mumu += mu2_by_seq[seq]
        mumu /= len(seqs)
        with open(os.path.join(expdir, 'test', 'mu2_by_seq.npy'), "wb") as fnp:
            np.save(fnp, mumu)

    return z1_by_seq, z2_by_seq, mu2_by_seq, regpost_by_seq, xin_by_seq, xout_by_seq, xoutv_by_seq, z1reg_by_seq, regpost_by_seq_z1, z2reg_by_seq, bReg_by_seq, cReg_by_seq


def compute_pred_acc_z2(expdir, model, conf, seqs, tt_dset, regpost_by_seq, z2reg_by_seq, cReg_by_seq):

    names = conf['facs'].split(':')
    lab2idx= conf['lab2idx']
    accuracies = [0. for _ in range(len(names))]
    #accuracies_z2reg = [0. for _ in range(len(names))]

    for i, name in enumerate(names):

        ordered_labs = lab2idx[name]
        truelabs = tt_dset.labs_d[name].seq2lab

        total = 0

        correct = 0  #using mu2
        #correct_z2reg = 0  #using z2_rlogits

        with open("%s/test/txt/%s_predictions.scp" % (expdir, name), "w") as f:
            f.write("#seq truelabel predictedlabel      for class %s \n" % name)
            for seq in seqs:
                # when no or unknown label ""
                if len(truelabs[seq]) == 0:
                    continue
                total += 1
                probs = regpost_by_seq[seq][i]
                # max + 1 since the first label in ordered_labs is the unknown label ""
                pred_lab = ordered_labs[np.argmax(probs)+1]
                if pred_lab == truelabs[seq]:
                    correct += 1
                f.write(seq+" "+str(truelabs[seq])+" "+str(pred_lab)+"\n")

                #probs_z2reg = z2reg_by_seq[seq][i]
                #pred_lab_z2reg = ordered_labs[np.argmax(np.sum(probs_z2reg, axis=0))+1]
                #if pred_lab_z2reg == truelabs[seq]:
                #    correct_z2reg += 1

        accuracies[i] = correct/total
        with open("%s/test/txt/%s_acc" % (expdir, name), "w") as fid:
            fid.write("%10.3f \n" % accuracies[i])
        print("prediction accuracy for labels of class %s is %f" % (name, accuracies[i]))

        #accuracies_z2reg[i] = correct_z2reg/total
        #print("prediction accuracy for labels of class from z2reg_by_seq %s is %f" % (name, accuracies[i]))

    return names, accuracies


def compute_pred_acc_z1(expdir, model, conf, seqs, tt_dset, regpost_by_seq, z1reg_by_seq, bReg_by_seq):

    names = conf['talabs'].split(':')
    talab2idx = conf['train_talab_vals']

    accuracies = [0. for _ in range(len(names))]

    for i, name in enumerate(names):
        with open("%s/test/txt/%s_predictions.scp" % (expdir, name), "w") as f:
            f.write("#segmentnumber true prediction \n")
            total = 0
            correct = 0

            for seq in seqs:
                nsegs = z1reg_by_seq[seq][0].shape[0]
                f.write('Sequence %s with %i segments \n' % (seq, nsegs))

                for j in range(nsegs):
                    truelab = bReg_by_seq[seq][j, i]

                    truelab = list(talab2idx[name].keys())[list(talab2idx[name].values()).index(truelab)]

                    # no or unknown label
                    if len(truelab) == 0:
                        continue

                    total += 1

                    # again + 1 because first label is the unknown label "" (not in z1reg)
                    pred_lab = np.argmax(z1reg_by_seq[seq][i][j, :]) + 1

                    pred_lab = list(talab2idx[name].keys())[list(talab2idx[name].values()).index(pred_lab)]

                    if pred_lab == truelab:
                        correct += 1

                    f.write("\t %i \t %s \t %s \n" % (j, str(truelab), str(pred_lab)))

        accuracies[i] = correct/total
        with open("%s/test/txt/%s_acc" % (expdir, name), "w") as fid:
            fid.write("%10.3f \n" % accuracies[i])
        print("prediction accuracy for labels of class %s is %f" % (name, accuracies[i]))

    return names, accuracies


def visualize_reg_vals(expdir, model, seqs, conf, z1_by_seq, z2_by_seq, mu2_by_seq, regpost_by_seq, xin_by_seq, xout_by_seq, xoutv_by_seq, z1reg_by_seq):

    if True:
        # names = ["region", "gender"]
        names = conf['facs'].split(':')
        for i, name in enumerate(names):
            with open("%s/test/txt/%s.scp" % (expdir, name), "w") as f:
                for seq in seqs:
                    f.write(seq + "  [ ")
                    for e in np.nditer(regpost_by_seq[seq][i]):
                        f.write("%10.3f " % e)
                    f.write("]\n")

    if True:
        names = ["pho"]
        #names = conf['talabs']  #.split(':')
        for i, name in enumerate(names):
            os.makedirs("%s/test/txt/%s" % (expdir, name), exist_ok=True)
            for seq in seqs:
                np.save("%s/test/txt/%s/%s" % (expdir, name, seq), z1reg_by_seq[seq][i])

    print("only using 10 random sequences for visualization")
    seqs = sorted(list(np.random.choice(seqs, 10, replace=False)))
    seq_names = ["%02d_%s" % (i, seq) for i, seq in enumerate(seqs)]

    if True:
        # visualize reconstruction
        print("visualizing reconstruction")
        plot_x([xin_by_seq[seq] for seq in seqs], seq_names, "%s/test/img/xin.png" % expdir)
        plot_x([xout_by_seq[seq] for seq in seqs], seq_names, "%s/test/img/xout.png" % expdir)
        plot_x([xoutv_by_seq[seq] for seq in seqs], seq_names,
               "%s/test/img/xout_logvar.png" % expdir, clim=(None, None))

    if True:
        # factorization: use the centered segment from each sequence
        print("visualizing factorization")
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
        # Maybe use dataset instead of dataset_test?
        with open(os.path.join(conf['datadir'], conf['dataset'], 'train', 'mvn.pkl'), "rb") as f:
            mvn_params = pickle.load(f)
        nb_mel = mvn_params["mean"].size
        for src_seq, src_seq_name in zip(seqs, seq_names):
            with open("%s/test/spec/xin_%s.npy" % (expdir, src_seq), "wb") as fnp:
                np.save(fnp, np.reshape(xin_by_seq[src_seq], (-1, nb_mel)) * mvn_params["std"] + mvn_params["mean"])
            with open("%s/test/spec/xout_%s.npy" % (expdir, src_seq), "wb") as fnp:
                np.save(fnp,
                        np.reshape(xout_by_seq[src_seq], (-1, nb_mel)) * mvn_params["std"] + mvn_params["mean"])

    if True:
        # sequence neutralisation
        print("visualizing neutral sequences")
        neu_by_seq = dict()
        with open("%s/test/mu2_by_seq.npy" % expdir, "rb") as fnp:
            mumu = np.float32(np.load(fnp))
        for src_seq, src_seq_name in zip(seqs, seq_names):
            del_mu2 = mumu - mu2_by_seq[src_seq]
            src_z1, src_z2 = z1_by_seq[src_seq], z2_by_seq[src_seq]
            neu_by_seq[src_seq] = _seq_translate(
                model, conf['tr_shape'], src_z1, src_z2, del_mu2)
            with open("%s/test/spec/neu_%s.npy" % (expdir, src_seq), "wb") as fnp:
                np.save(fnp, np.reshape(neu_by_seq[src_seq], (-1, nb_mel)) * mvn_params["std"] + mvn_params["mean"])

        plot_x([neu_by_seq[seq] for seq in seqs], seq_names,
               "%s/test/img/neutral.png" % expdir, False)

    if True:
        # sequence translation
        print("visualizing sequence translation")
        xtra_by_seq = dict()
        for src_seq, src_seq_name in zip(seqs, seq_names):
            xtra_by_seq[src_seq] = dict()
            src_z1, src_z2 = z1_by_seq[src_seq], z2_by_seq[src_seq]
            for tar_seq in seqs:
                del_mu2 = mu2_by_seq[tar_seq] - mu2_by_seq[src_seq]
                xtra_by_seq[src_seq][tar_seq] = _seq_translate(
                    model, conf['tr_shape'], src_z1, src_z2, del_mu2)
                with open("%s/test/spec/src_%s_tar_%s.npy" % (expdir, src_seq, tar_seq), "wb") as fnp:
                    np.save(fnp, np.reshape(xtra_by_seq[src_seq][tar_seq], (-1, nb_mel)) * mvn_params["std"] +
                            mvn_params["mean"])

            plot_x([xtra_by_seq[src_seq][seq] for seq in seqs], seq_names,
                   "%s/test/img/x_tra/%s_tra.png" % (expdir, src_seq_name), True)

    if True:
        # tsne z1 and z2
        print("t-SNE analysis on latent variables (over entire sequences)")
        n = [len(z1_by_seq[seq]) for seq in seqs]
        z1 = np.concatenate([z1_by_seq[seq] for seq in seqs], axis=0)
        z2 = np.concatenate([z2_by_seq[seq] for seq in seqs], axis=0)

        p = 30
        print("  perplexity = %s" % p)
        tsne = TSNE(n_components=2, verbose=0, perplexity=p, n_iter=1000)
        z1_tsne = _unflatten(tsne.fit_transform(z1), n)
        scatter_plot(z1_tsne, seq_names, "z1_tsne_%03d" % p,
                     "%s/test/img/z1_tsne_%03d.png" % (expdir, p))
        z2_tsne = _unflatten(tsne.fit_transform(z2), n)
        scatter_plot(z2_tsne, seq_names, "z2_tsne_%03d" % p,
                     "%s/test/img/z2_tsne_%03d.png" % (expdir, p))


def tsne_by_label(expdir, model, conf, create_test_dataset, seqs, tt_dset, bReg_by_seq, z1_by_phone):

    if len(seqs) > 25:
        seqs = sorted(list(np.random.choice(seqs, 25, replace=False)))

    # infer z1, z2
    z1_by_seq = defaultdict(list)
    z2_by_seq = defaultdict(list)
    for seq in seqs:

        Test_Dataset = create_test_dataset([seq])

        for _, x, _, _, _, _ in Test_Dataset:
            _, _, _, _, _, _, qz1_x, qz2_x = model.encoder(x)
            z2_by_seq[seq].append(qz2_x[0])
            z1_by_seq[seq].append(qz1_x[0])

        z1_by_seq[seq] = np.concatenate(z1_by_seq[seq], axis=0)
        z2_by_seq[seq] = np.concatenate(z2_by_seq[seq], axis=0)

    # tsne z1 and z2
    print("t-SNE analysis on latent variables by label")
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
        z1_mu, _, _, z2_mu, _, _, _, _ = model.encoder(tf.reshape(xval, [-1, tr_shape[0], tr_shape[1]]))

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


def _print_mu2_stat(mu2_dict):
    norm_sum = 0.
    dim_norm_sum = 0.
    for y in sorted(mu2_dict.keys()):
        norm_sum += np.linalg.norm(mu2_dict[y])
        dim_norm_sum += np.abs(mu2_dict[y])
    avg_norm = norm_sum / len(mu2_dict)
    avg_dim_norm = dim_norm_sum / len(mu2_dict)
    print("avg. norm = %.2f, #mu2 = %s" % (avg_norm, len(mu2_dict)))
    print("per dim: %s" % (" ".join(["%.2f" % v for v in avg_dim_norm]),))


def _softmax(x):
    ## First column are zeros (as added in fix_logits in model, so leave these out and return size-1 tens
    y = np.exp(x[:, 1:])
    return y / np.sum(y, axis=1, keepdims=True)
    # return tf.nn.softmax(x, axis=1)


def _seq_translate(model, tr_shape, src_z1, src_z2, del_mu2):
    mod_z2 = src_z2 + del_mu2[np.newaxis, ...]
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


# def _join_talab(z_by_seqs, seq2talabseq, talab_vals):
#     d = defaultdict(list)
#     for seq, z in list(z_by_seqs.items()):
#         n_segs = z.shape[0]
#         seq_talabs = seq2talabseq[seq].talabs
#         for seg in range(n_segs):
#             idx = seq_talabs[seg].lab
#             talab = list(talab_vals.keys())[list(talab_vals.values()).index(idx)]
#             d[talab].append(z[seg, :])
#     for lab in d:
#         d[lab] = np.stack(d[lab], axis=0)
#     return list(d.keys()), list(d.values())
