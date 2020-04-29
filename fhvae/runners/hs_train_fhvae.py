from __future__ import division
import os
import sys
import time
import math
import numpy as np
import tensorflow as tf
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fhvae.datasets.seq2seg import seq_to_seg_mapper


def hs_train_reg(exp_dir, model, conf, tr_iterator_by_seqs, dt_iterator, tr_dset, dt_dset, num_phones):
    """
    train fhvae with hierarchical sampling
    """

    # setup the optimizer
    if conf['lr'] == 'custom':
        learning_rate = CustomSchedule(conf['d_model'], warmup_steps=conf['warmup_steps'], k=conf['k'])
    else:
        learning_rate = conf['lr']

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=conf['beta1'], beta_2=conf['beta2'], epsilon=conf['adam_eps'], amsgrad=True)
    train_loss = tf.keras.metrics.Mean(name='train_loss')

    # save model checkpoints
    checkpoint_directory = os.path.join(exp_dir, 'training_checkpoints')
    os.makedirs(checkpoint_directory, exist_ok=True)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_directory, max_to_keep=conf['n_patience']+2)

    # write logfiles
    logdir = os.path.join(exp_dir, "logdir")
    os.makedirs(logdir, exist_ok=True)
    writer = tf.summary.create_file_writer(logdir)
    writer.set_as_default()

    ## Enable following lines before and after a @tf.function function
    ## If you want to visualize the graph created, with cmd 'tensorboard --logdir logdir'
    # tf.summary.trace_on(graph=True)
    # %% tf.function def %%
    # with writer.as_default():
    #   tf.summary.trace_export(
    #        name="my_func_trace",
    #        step=epoch - 1,
    #        profiler_outdir=logdir)

    mean_losses = []
    valid_losses = []

    # print all losses to text files
    lossdir = os.path.join(exp_dir, "losses")
    os.makedirs(lossdir, exist_ok=True)
    meanloss_file = os.path.join(lossdir, 'result_mean_loss.txt')
    validloss_file = os.path.join(lossdir, 'result_valid_loss.txt')
    comploss_file = os.path.join(lossdir, 'result_comp_loss_0.txt')

    if os.path.exists(meanloss_file):
        os.remove(meanloss_file)
    if os.path.exists(comploss_file):
        os.remove(comploss_file)
    if os.path.exists(validloss_file):
        os.remove(validloss_file)

    # initial values
    best_epoch, best_valid_loss = 0, np.inf
    tr_seg_len = tf.constant(tr_dset.seg_len, tf.int64)
    tr_seg_shift = tf.constant(tr_dset.seg_shift, tf.int64)
    dt_seg_len = tf.constant(dt_dset.seg_len, tf.int64)
    dt_seg_shift = tf.constant(dt_dset.seg_shift, tf.int64)
    flag = True
    start = time.time()

    def segment_map_fn(idx, data):
        # map every sequence to <variable #> segments
        # + use enumeration-number as sequence key such that all keys in range [0, nmu2]
        keys, feats, lens, labs, talabs = data
        keys, feats, lens, labs, talabs, starts = seq_to_seg_mapper(idx, feats, lens, labs, talabs,
                                                            tr_seg_len, tr_seg_shift, tr_dset.rand_seg, len(conf['b_n']))
        return keys, feats, lens, labs, talabs, starts

    def segment_map_fn_validation(idx, data):
        # map every sequence to <variable #> segments
        # + use enumeration-number as sequence key such that all keys in range [0, nmu2]
        keys, feats, lens, labs, talabs = data
        keys, feats, lens, labs, talabs, starts = seq_to_seg_mapper(idx, feats, lens, labs, talabs,
                                                            dt_seg_len, dt_seg_shift, dt_dset.rand_seg, len(conf['b_n']))
        return keys, feats, lens, labs, talabs, starts

    def create_train_dataset(seqs):
        # build a tf.data.dataset for the #nmu2 seqs provided in s_seqs
        Sequence_Dataset = tf.data.Dataset.from_generator(
            lambda: tr_iterator_by_seqs(s_seqs=seqs, bs=conf['batch_size'], seg_rem=True), \
            output_shapes=((), (None, conf['tr_shape'][1]), (), (len(conf['c_n']),), (len(conf['b_n']), 3, None)), \
            output_types=(tf.int64, tf.float32, tf.int64, tf.int64, tf.int64)) \
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE) \
            .enumerate(start=0) \
            .cache()

        # parallellise the sequence to segment mapping
        Segment_Dataset = Sequence_Dataset \
            .map(segment_map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
            .unbatch() \
            .shuffle(buffer_size=5000) \
            .batch(batch_size=conf['batch_size'], drop_remainder=True) \
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return Segment_Dataset

    def create_validation_dataset():

        # initialize the validation dataset, assumed to have #sequences < nmu2
        # otherwise work with num_steps and split in sequence/segment dataset like with training dataset
        Validation_Dataset = tf.data.Dataset.from_generator(
                                lambda: dt_iterator(bs=conf['batch_size']), \
                                output_shapes=((), (None, conf['tr_shape'][1]), (), (len(conf['c_n']),), (len(conf['b_n']), 3, None)), \
                                output_types=(tf.int64, tf.float32, tf.int64, tf.int64, tf.int64)) \
                                .enumerate(start=0) \
                                .map(segment_map_fn_validation, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
                                .unbatch() \
                                .batch(batch_size=conf['batch_size']) \
                                .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return Validation_Dataset

    Validation_Dataset = create_validation_dataset()

    print('NOTE TO USER: The first epoch will be slower, due to building/tracing the graphs')
    for epoch in range(1, conf['n_epochs']+1):

        print("EPOCH %i\n" % epoch)
        epoch_start = time.time()
        train_loss.reset_states()
        comploss_file = os.path.join(lossdir, 'result_comp_loss.txt')

        # shuffle the sequence list
        seqs = list(tr_dset.seqlist)
        num_seqs = len(seqs)  # for in loss-function
        np.random.shuffle(seqs)

        # hierarchical sampling: train on #nmu2 sequences per step
        num_steps = len(seqs) // conf['nmu2']

        for step in range(0, num_steps + 1):

            # get #nmu2 sequences (already shuffled) and build the dataset
            sample_seqs = seqs[step*conf['nmu2']:min((step+1)*conf['nmu2'], len(seqs))]
            Train_Dataset = create_train_dataset(sample_seqs)

            start_mu_tables = time.time()

            # initialize
            mu1_table = tf.zeros([num_phones, conf['z1_dim']], dtype=tf.float32)
            mu2_table = tf.zeros([conf['nmu2'], conf['z2_dim']], dtype=tf.float32)
            nsegs = tf.zeros([conf['nmu2']])
            phone_occs = tf.zeros([num_phones])

            # calculate mu1-dict and mu2-dict
            mu1_table, mu2_table, phone_occs = estimate_mu1_mu2_dict(model, Train_Dataset, mu1_table, mu2_table, phone_occs, nsegs, conf['tr_shape'])
            model.mu1_table.assign(mu1_table)
            model.mu2_table.assign(mu2_table)
            model.phone_occs.assign(phone_occs)

            print('calculating mu1_dict and mu2_dict took {} seconds \n'.format(time.time() - start_mu_tables))

            # seq-keys, feats, nsegs_of_seq, labs, talabs
            for yval, xval, nval, cval, bval, _ in Train_Dataset:

                step_loss, log_pmu2, log_pmu1, neg_kld_z2, neg_kld_z1, log_px_z, lb, log_qy_mu2, log_qy_mu1, log_b_loss, log_c_loss = train_step(model, xval, yval, tf.cast(nval, dtype=tf.float32), bval, cval, optimizer, train_loss, num_seqs)

                train_loss(step_loss)

                # print the model variables (only once)
                if flag:
                    for v in model.trainable_variables:
                        print((v.name, v.shape))
                    flag = False

                # print separate losses in terminal during first epoch for debugging purposes
                if epoch < 2:
                    print("Loss: %f \t \t lb=%f \t log_qy_mu2=%f \t log_qy_mu1=%f \t log_b=%f \t log_c=%f" % (step_loss, -tf.reduce_mean(lb), -tf.reduce_mean(log_qy_mu2), -tf.reduce_mean(log_qy_mu1), -tf.reduce_mean(log_b_loss), -tf.reduce_mean(log_c_loss)))

                    print("\t lower bound components: \t log_pmu2=%f \t log_pmu1=%f \t neg_kld_z2=%f \t neg_kld_z1=%f \t log_px_z=%f" % (-tf.reduce_mean(log_pmu2), -tf.reduce_mean(log_pmu1), -tf.reduce_mean(neg_kld_z2), -tf.reduce_mean(neg_kld_z1), -tf.reduce_mean(log_px_z)))

                    # print the results to a file
                    with open(comploss_file, "a+") as pid:
                        pid.write("Loss: %f \t \t lb=%f \t log_qy_mu2=%f \t log_qy_mu1=%f \t log_b=%f \t log_c=%f \n" % (step_loss, -tf.reduce_mean(lb), -tf.reduce_mean(log_qy_mu2), -tf.reduce_mean(log_qy_mu1), -tf.reduce_mean(log_b_loss), -tf.reduce_mean(log_c_loss)))
                        pid.write("\t lower bound components: \t log_pmu2=%f \t log_pmu1=%f \t neg_kld_z2=%f \t neg_kld_z1=%f \t log_px_z=%f \n" % (-tf.reduce_mean(log_pmu2), -tf.reduce_mean(log_pmu1), -tf.reduce_mean(neg_kld_z2), -tf.reduce_mean(neg_kld_z1), -tf.reduce_mean(log_px_z)))

        # write components of first loss only in later epochs
        with open(comploss_file, "a+") as pid:
            pid.write("EPOCH: %i \n" % int(epoch))
            pid.write("Loss: %f \t \t lb=%f \t log_qy_mu2=%f \t log_qy_mu1=%f \t log_b=%f \t log_c=%f \n" % (step_loss, -tf.reduce_mean(lb), -tf.reduce_mean(log_qy_mu2), -tf.reduce_mean(log_qy_mu1), -tf.reduce_mean(log_b_loss), -tf.reduce_mean(log_c_loss)))
            pid.write("\t lower bound components: \t log_pmu2=%f \t log_pmu1=%f \t neg_kld_z2=%f \t neg_kld_z1=%f \t log_px_z=%f \n" % (-tf.reduce_mean(log_pmu2), -tf.reduce_mean(log_pmu1), -tf.reduce_mean(neg_kld_z2), -tf.reduce_mean(neg_kld_z1), -tf.reduce_mean(log_px_z)))

        # write mean loss of epoch
        print('Resulting mean-loss of epoch {} is {:.4f}, which took {} seconds to run'.format(epoch, train_loss.result(), time.time()-epoch_start))
        mean_losses.append(float(train_loss.result()))
        with open(meanloss_file, "a+") as fid:
            fid.write('Resulting mean-loss of epoch {} is {:.4f}, which took {} seconds to run\n'.format(epoch, train_loss.result(), time.time()-epoch_start))

        tf.summary.scalar('train_loss', train_loss.result(), step=epoch)

        manager.save()

        start_val = time.time()

        # validation step
        valid_loss, normalloss = validation_step(model, Validation_Dataset, conf, num_phones, len(dt_dset.seqlist))

        valid_losses.append(valid_loss)
        print('Validation loss of epoch {} is {:.4f}, and {:.4f} when calculated differently, and took {} seconds \n'.format(epoch, valid_loss, normalloss, time.time()-start_val))
        with open(validloss_file, "a+") as fid:
            fid.write('Validation loss of epoch {} is {:.4f}, and {:.4f} when calculated differently \n'.format(epoch, valid_loss, normalloss))

        # early stopping
        best_epoch, best_valid_loss, is_finished = check_finished(conf, epoch, best_epoch, valid_loss, best_valid_loss)
        with open(os.path.join(checkpoint_directory, 'best_checkpoint'), 'w+') as lid:
            lid.write(str(best_epoch))
        if is_finished:
            break

    print('Complete run over {} epochs took {} seconds\n'.format(conf['n_epochs'], time.time()-start))
    print('Best run was in epoch {} with a validation loss of {} \n'.format(best_epoch, best_valid_loss))

    # make plots of loss after training
    plt.figure('result-meanloss')
    plt.plot(mean_losses)
    plt.xlabel('Epochs #')
    plt.ylabel('Mean Loss')
    plt.savefig(os.path.join(exp_dir, 'result_mean_loss.pdf'), format='pdf')

    plt.figure('result-validloss')
    plt.plot(valid_losses)
    plt.xlabel('Epochs #')
    plt.ylabel('Mean Loss')
    plt.savefig(os.path.join(exp_dir, 'result_valid_loss.pdf'), format='pdf')


@tf.function
def train_step(model, x, y, n, bReg, cReg, optimizer, train_loss, num_seqs):
    """
    train fhvae step by step and compute the gradients from the losses
    """
    print('tracing...')

    with tf.GradientTape() as tape:

        mu2, mu1, qz2_x, z2_sample, qz1_x, z1_sample, px_z, x_sample, z1_rlogits, z2_rlogits = model(x, y, bReg[:, 0])

        loss, log_pmu2, log_pmu1, neg_kld_z2, neg_kld_z1, log_px_z, lb, log_qy_mu2, log_qy_mu1, log_b_loss, log_c_loss = model.compute_loss(x, y, n, bReg, cReg, mu2, mu1, qz2_x, z2_sample, qz1_x, z1_sample, px_z, x_sample, z1_rlogits, z2_rlogits, num_seqs)

    # apply gradients
    gradients = tape.gradient(loss, model.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.NONE)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # update keras mean loss metric
    train_loss(loss)

    return loss, log_pmu2, log_pmu1, neg_kld_z2, neg_kld_z1, log_px_z, lb, log_qy_mu2, log_qy_mu1, log_b_loss, log_c_loss


def estimate_mu1_mu2_dict(model, dataset, mu1_table, mu2_table, nphones, nsegs, tr_shape):
    '''
    Calculate mu1/mu2 dict tables for every phone/sequence with posterior inference formulae
    '''

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


def validation_step(model, val_dataset, conf, num_phones, num_val_seqs):
    """
    calculate loss on development set
    """
    validloss = 0.0
    tot_segs = 0.0
    normalloss = 0.0

    mu1_table = tf.zeros([num_phones, conf['z1_dim']], dtype=tf.float32)
    mu2_table = tf.zeros([conf['nmu2'], conf['z2_dim']], dtype=tf.float32)
    nsegs = tf.zeros([conf['nmu2']])
    phone_occs = tf.zeros([num_phones])

    # calculate mu1-dict and mu2-dict
    mu1_table, mu2_table, phone_occs = estimate_mu1_mu2_dict(model, val_dataset, mu1_table, mu2_table, phone_occs, nsegs, conf['tr_shape'])
    model.mu1_table.assign(mu1_table)
    model.mu2_table.assign(mu2_table)
    model.phone_occs.assign(phone_occs)

    for yval, xval, nval, cval, bval, _ in val_dataset:

        mu2, mu1, qz2_x, z2_sample, qz1_x, z1_sample, px_z, x_sample, z1_rlogits, z2_rlogits = model(xval, yval, bval[:, 0])

        loss, log_pmu2, log_pmu1, neg_kld_z2, neg_kld_z1, log_px_z, lb, log_qy_mu2, log_qy_mu1, log_b_loss, log_c_loss = model.compute_loss(tf.cast(xval, dtype=tf.float32), yval, tf.cast(nval, dtype=tf.float32), bval, cval, mu2, mu1, qz2_x,
                               z2_sample, qz1_x, z1_sample, px_z, x_sample, z1_rlogits, z2_rlogits, num_val_seqs)

        validloss += loss * len(yval)
        tot_segs += len(yval)
        normalloss += loss

    return validloss / tot_segs, normalloss


def check_finished(conf, epoch, best_epoch, val_loss, best_val_loss):
    """
    stop if validation loss doesnt improve after n_patience epochs
    """
    is_finished = False
    if val_loss < best_val_loss:
        best_epoch = epoch
        best_val_loss = val_loss

    if (epoch - best_epoch) > conf['n_patience']:
        is_finished = True

    if math.isnan(val_loss):
        is_finished = True

    return best_epoch, best_val_loss, is_finished


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    # define custom learning rate schedule as according to transformer paper
    def __init__(self, d_model, warmup_steps=7000, k=10):
        super(CustomSchedule, self).__init__()

        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.k = tf.cast(k, tf.float32)

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return self.k * tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
