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


def hs_train_reg(exp_dir, model, optimizer, conf, tr_iterator_by_seqs, dt_iterator, tr_dset, dt_dset, num_phones, noise_training):
    """
    train fhvae with hierarchical sampling
    """
    
    # quick fix
    if not noise_training:
        conf['num_noisy_versions'] = 0

    # save model checkpoints
    checkpoint_directory = os.path.join(exp_dir, 'training_checkpoints')
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

    # restore a crashed or evicted training job
    if os.path.exists(checkpoint_directory):
        if len(os.listdir(checkpoint_directory)) > 0:
            manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_directory, max_to_keep=5)
            status = checkpoint.restore(manager.latest_checkpoint)  # manager.latest_checkpoint e.g. <ckptdir>/ckpt-3
            curr_epoch = int((os.path.basename(manager.latest_checkpoint)).split('-')[1])

            if os.path.exists(os.path.join(checkpoint_directory, 'best_checkpoint')):
                with open(os.path.join(checkpoint_directory, 'best_checkpoint'), 'r') as pid:
                    best_epoch = int((pid.readline()).rstrip())
            else:
                best_epoch = curr_epoch

            if os.path.exists(os.path.join(checkpoint_directory, 'best_valid_loss')):
                with open(os.path.join(checkpoint_directory, 'best_valid_loss'), 'r') as pid:
                    best_valid_loss = float((pid.readline()).rstrip())
            else:
                best_valid_loss = np.inf
    else:
        os.makedirs(checkpoint_directory, exist_ok=True)
        manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_directory, max_to_keep=conf['n_patience']+2)
        best_epoch, best_valid_loss = 0, np.inf
        curr_epoch = 0

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

    train_loss = tf.keras.metrics.Mean(name='train_loss')

    # print all losses to text files
    lossdir = os.path.join(exp_dir, "losses")
    os.makedirs(lossdir, exist_ok=True)
    meanloss_file = os.path.join(lossdir, 'result_mean_loss.txt')
    validloss_file = os.path.join(lossdir, 'result_valid_loss.txt')
    comploss_file = os.path.join(lossdir, 'result_comp_loss_0.txt')

    # initial values
    #best_epoch, best_valid_loss = 0, np.inf
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
        keys, feats, lens, labs, talabs, starts = seq_to_seg_mapper(idx, keys, feats, lens, labs, talabs,
                                                            tr_seg_len, tr_seg_shift, tr_dset.rand_seg, len(conf['b_n']), int(conf['num_noisy_versions']))
        return keys, feats, lens, labs, talabs, starts

    def segment_map_fn_validation(idx, data):
        # map every sequence to <variable #> segments
        # + use enumeration-number as sequence key such that all keys in range [0, nmu2]
        keys, feats, lens, labs, talabs = data
        keys, feats, lens, labs, talabs, starts = seq_to_seg_mapper(idx, keys, feats, lens, labs, talabs,
                                                            dt_seg_len, dt_seg_shift, dt_dset.rand_seg, len(conf['b_n']), int(conf['num_noisy_versions']))
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

    def create_train_dataset_mixed(cleanseqs, noisyseqs, epoch, num_noisy_versions):

        Clean_Sequence_Dataset = tf.data.Dataset.from_generator(
            lambda: tr_iterator_by_seqs(s_seqs=cleanseqs, bs=conf['batch_size'], seg_rem=True), \
            output_shapes=((), (None, conf['tr_shape'][1]), (), (len(conf['c_n']),), (len(conf['b_n']), 3, None)), \
            output_types=(tf.int64, tf.float32, tf.int64, tf.int64, tf.int64)) \
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE) \
            .enumerate(start=0) \
            .cache()

        Clean_Segment_Dataset = Clean_Sequence_Dataset \
            .map(segment_map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
            .unbatch() \
            .shuffle(buffer_size=5000, seed=epoch)
            # \
            # .batch(batch_size=conf['batch_size'], drop_remainder=True) \
            # .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        Noisy_Sequence_Dataset = tf.data.Dataset.from_generator(
            lambda: tr_iterator_by_seqs(s_seqs=noisyseqs, bs=conf['batch_size'], seg_rem=True), \
            output_shapes=((), (None, conf['tr_shape'][1]), (), (len(conf['c_n']),), (len(conf['b_n']), 3, None)), \
            output_types=(tf.int64, tf.float32, tf.int64, tf.int64, tf.int64)) \
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE) \
            .enumerate(start=0) \
            .cache()

        Noisy_Segment_Dataset = Noisy_Sequence_Dataset \
            .map(segment_map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
            .unbatch() \
            .shuffle(buffer_size=5000, seed=epoch)
            # \
            # .batch(batch_size=conf['batch_size'], drop_remainder=True) \
            # .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        datasets = [Clean_Segment_Dataset, Noisy_Segment_Dataset]
        choices = tf.data.Dataset.from_tensor_slices(tf.cast([0]+num_noisy_versions*[1], tf.int64)).repeat()
        Mixed_Dataset = tf.data.experimental.choose_from_datasets(datasets, choices) \
                            .batch(batch_size=conf['batch_size'], drop_remainder=True) \
                            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return Mixed_Dataset

    def create_validation_dataset(seqs):

        # initialize the validation dataset, assumed to have #sequences < nmu2
        # otherwise work with num_steps and split in sequence/segment dataset like with training dataset
        Validation_Dataset = tf.data.Dataset.from_generator(
                                lambda: dt_iterator(s_seqs=seqs, bs=conf['batch_size']), \
                                output_shapes=((), (None, conf['tr_shape'][1]), (), (len(conf['c_n']),), (len(conf['b_n']), 3, None)), \
                                output_types=(tf.int64, tf.float32, tf.int64, tf.int64, tf.int64)) \
                                .enumerate(start=0) \
                                .map(segment_map_fn_validation, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
                                .unbatch() \
                                .batch(batch_size=conf['batch_size']) \
                                .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return Validation_Dataset

    def create_validation_dataset_mixed(clean_valseqs, noisy_valseqs, num_noisy_versions):

        # initialize the validation dataset, assumed to have #sequences < nmu2
        # otherwise work with num_steps and split in sequence/segment dataset like with training dataset
        Clean_Validation_Dataset = tf.data.Dataset.from_generator(
                                lambda: dt_iterator(s_seqs=clean_valseqs, bs=conf['batch_size']), \
                                output_shapes=((), (None, conf['tr_shape'][1]), (), (len(conf['c_n']),), (len(conf['b_n']), 3, None)), \
                                output_types=(tf.int64, tf.float32, tf.int64, tf.int64, tf.int64)) \
                                .enumerate(start=0) \
                                .map(segment_map_fn_validation, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
                                .unbatch()

        Noisy_Validation_Dataset = tf.data.Dataset.from_generator(
                                lambda: dt_iterator(s_seqs=noisy_valseqs, bs=conf['batch_size']), \
                                output_shapes=((), (None, conf['tr_shape'][1]), (), (len(conf['c_n']),), (len(conf['b_n']), 3, None)), \
                                output_types=(tf.int64, tf.float32, tf.int64, tf.int64, tf.int64)) \
                                .enumerate(start=0) \
                                .map(segment_map_fn_validation, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
                                .unbatch()

        val_datasets = [Clean_Validation_Dataset, Noisy_Validation_Dataset]
        choices = tf.data.Dataset.from_tensor_slices(tf.cast([0]+num_noisy_versions*[1], tf.int64)).repeat()
        Mixed_Validation_Dataset = tf.data.experimental.choose_from_datasets(val_datasets, choices) \
                                .batch(batch_size=conf['batch_size'], drop_remainder=False) \
                                .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return Mixed_Validation_Dataset

    if noise_training:
        assert (conf['batch_size'] % (
                    conf['num_noisy_versions'] + 1)) == 0, "batch size should be a multiple of (num_noisy_versions+1)"

    print('NOTE TO USER: The first epoch will be slower, due to building/tracing the graphs')
    for epoch in range(curr_epoch+1, conf['n_epochs']+1):

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

        if noise_training:
            indices = list(range(len(tr_dset.clean_seqlist)))
            np.random.shuffle(indices)
            clean_seqs = [tr_dset.clean_seqlist[i] for i in indices]
            noisy_seqs = [seq for i in indices for seq in tr_dset.noisy_seqlist[i*conf['num_noisy_versions']:(i+1)*conf['num_noisy_versions']]]

        for step in range(0, 1):  # num_steps + 1):

            # get #nmu2 sequences (already shuffled) and build the dataset
            sample_seqs = seqs[step*conf['nmu2']:min((step+1)*conf['nmu2'], len(seqs))]
            Train_Dataset = create_train_dataset(sample_seqs)

            if noise_training:
                clean_sample_seqs = clean_seqs[int(step * conf['nmu2'] / (conf['num_noisy_versions']+1)):int(
                    min((step + 1) * conf['nmu2'] / (conf['num_noisy_versions']+1), len(clean_seqs)))]
                noisy_sample_seqs = noisy_seqs[int(step * conf['num_noisy_versions'] * conf['nmu2'] / (conf['num_noisy_versions']+1)):int(
                    min((step + 1) * conf['num_noisy_versions'] * conf['nmu2'] / (conf['num_noisy_versions']+1), len(noisy_seqs)))]
                Train_Dataset = create_train_dataset_mixed(clean_sample_seqs, noisy_sample_seqs, epoch, conf['num_noisy_versions'])

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

                step_loss, log_pmu2, log_pmu1, neg_kld_z2, neg_kld_z1, log_px_z, lb, log_qy_mu2, log_qy_mu1, log_b_loss, log_c_loss, advlog_b_loss, advlog_c_loss, noise_loss = train_step(model, xval, yval, tf.cast(nval, dtype=tf.float32), bval, cval, optimizer, train_loss, num_seqs)

                train_loss(step_loss)

                # print the model variables (only once)
                if flag:
                    for v in model.trainable_variables:
                        print((v.name, v.shape))
                    flag = False

                # print separate losses in terminal during first epoch for debugging purposes
                if epoch < 2:
                    print("Loss: %f \t \t lb=%f \t log_qy_mu2=%f \t log_qy_mu1=%f \t log_b=%f \t log_c=%f \t advlog_b=%f \t advlog_c=%f \t noise_loss=%f" % (step_loss, -tf.reduce_mean(lb), -tf.reduce_mean(log_qy_mu2), -tf.reduce_mean(log_qy_mu1), -tf.reduce_mean(log_b_loss), -tf.reduce_mean(log_c_loss), -tf.reduce_mean(advlog_b_loss), -tf.reduce_mean(advlog_c_loss), -tf.reduce_mean(noise_loss)))

                    print("\t lower bound components: \t log_pmu2=%f \t log_pmu1=%f \t neg_kld_z2=%f \t neg_kld_z1=%f \t log_px_z=%f" % (-tf.reduce_mean(log_pmu2), -tf.reduce_mean(log_pmu1), -tf.reduce_mean(neg_kld_z2), -tf.reduce_mean(neg_kld_z1), -tf.reduce_mean(log_px_z)))

                    # print the results to a file
                    with open(comploss_file, "a+") as pid:
                        pid.write("Loss: %f \t \t lb=%f \t log_qy_mu2=%f \t log_qy_mu1=%f \t log_b=%f \t log_c=%f \t advlog_b=%f \t advlog_c=%f \t noise_loss=%f\n" % (step_loss, -tf.reduce_mean(lb), -tf.reduce_mean(log_qy_mu2), -tf.reduce_mean(log_qy_mu1), -tf.reduce_mean(log_b_loss), -tf.reduce_mean(log_c_loss), -tf.reduce_mean(advlog_b_loss), -tf.reduce_mean(advlog_c_loss), -tf.reduce_mean(noise_loss)))
                        pid.write("\t lower bound components: \t log_pmu2=%f \t log_pmu1=%f \t neg_kld_z2=%f \t neg_kld_z1=%f \t log_px_z=%f \n" % (-tf.reduce_mean(log_pmu2), -tf.reduce_mean(log_pmu1), -tf.reduce_mean(neg_kld_z2), -tf.reduce_mean(neg_kld_z1), -tf.reduce_mean(log_px_z)))

        # write components of first loss only in later epochs
        with open(comploss_file, "a+") as pid:
            pid.write("EPOCH: %i \n" % int(epoch))
            pid.write("Loss: %f \t \t lb=%f \t log_qy_mu2=%f \t log_qy_mu1=%f \t log_b=%f \t log_c=%f \t advlog_b=%f \t advlog_c=%f \t noise_loss=%f\n" % (step_loss, -tf.reduce_mean(lb), -tf.reduce_mean(log_qy_mu2), -tf.reduce_mean(log_qy_mu1), -tf.reduce_mean(log_b_loss), -tf.reduce_mean(log_c_loss), -tf.reduce_mean(advlog_b_loss), -tf.reduce_mean(advlog_c_loss), -tf.reduce_mean(noise_loss)))
            pid.write("\t lower bound components: \t log_pmu2=%f \t log_pmu1=%f \t neg_kld_z2=%f \t neg_kld_z1=%f \t log_px_z=%f \n" % (-tf.reduce_mean(log_pmu2), -tf.reduce_mean(log_pmu1), -tf.reduce_mean(neg_kld_z2), -tf.reduce_mean(neg_kld_z1), -tf.reduce_mean(log_px_z)))

        # write mean loss of epoch
        print('Resulting mean-loss of epoch {} is {:.4f}, which took {} seconds to run'.format(epoch, train_loss.result(), time.time()-epoch_start))
        mean_losses.append(float(train_loss.result()))
        with open(meanloss_file, "a+") as fid:
            fid.write('Resulting mean-loss of epoch {} is {:.4f}, which took {} seconds to run\n'.format(epoch, train_loss.result(), time.time()-epoch_start))

        tf.summary.scalar('train_loss', train_loss.result(), step=epoch)

        manager.save()

        curr_epoch += 1

        start_val = time.time()

        # validation step
        if len(dt_dset.seqlist) < conf['nmu2']:
            Validation_Dataset = create_validation_dataset(dt_dset.seqlist)
            if noise_training:
                Validation_Dataset = create_validation_dataset_mixed(dt_dset.clean_seqlist, dt_dset.noisy_seqlist,
                                                                     conf['num_noisy_versions'])
            valid_loss, normalloss = validation_step(model, Validation_Dataset, conf, num_phones, len(dt_dset.seqlist))
        else:
            valid_loss_t, normalloss_t = 0.0, 0.0
            num_val_steps = len(dt_dset.seqlist) // conf['nmu2']
            for valstep in range(0, num_val_steps+1):
                if not noise_training:
                    valseqs = list(dt_dset.seqlist)
                    sample_val_seqs = valseqs[step * conf['nmu2']:min((step + 1) * conf['nmu2'], len(valseqs))]
                    Validation_Dataset = create_validation_dataset(sample_val_seqs)
                    valid_loss, normalloss = validation_step(model, Validation_Dataset, conf, num_phones, len(dt_dset.seqlist))
                    valid_loss_t += valid_loss
                    normalloss_t += normalloss
                else:
                    clean_val_seqs = list(dt_dset.clean_seqlist)
                    noisy_val_seqs = list(dt_dset.noisy_seqlist)
                    clean_sample_val_seqs = clean_val_seqs[int(valstep * conf['nmu2'] / (conf['num_noisy_versions'] + 1)):int(
                        min((valstep + 1) * conf['nmu2'] / (conf['num_noisy_versions'] + 1), len(clean_val_seqs)))]
                    noisy_sample_val_seqs = noisy_val_seqs[int(
                        valstep * conf['num_noisy_versions'] * conf['nmu2'] / (conf['num_noisy_versions'] + 1)):int(
                        min((valstep + 1) * conf['num_noisy_versions'] * conf['nmu2'] / (conf['num_noisy_versions'] + 1),
                            len(noisy_val_seqs)))]
                    Validation_Dataset = create_validation_dataset_mixed(clean_sample_val_seqs, noisy_sample_val_seqs,
                                                               conf['num_noisy_versions'])
                    valid_loss, normalloss = validation_step(model, Validation_Dataset, conf, num_phones, len(dt_dset.seqlist))
                    valid_loss_t += valid_loss
                    normalloss_t += normalloss
            valid_loss = valid_loss_t / num_val_steps
            normalloss = normalloss_t / num_val_steps
        valid_losses.append(valid_loss)
        print('Validation loss of epoch {} is {:.4f}, and {:.4f} when calculated differently, and took {} seconds \n'.format(epoch, valid_loss, normalloss, time.time()-start_val))
        with open(validloss_file, "a+") as fid:
            fid.write('Validation loss of epoch {} is {:.4f}, and {:.4f} when calculated differently \n'.format(epoch, valid_loss, normalloss))

        # early stopping
        best_epoch, best_valid_loss, is_finished = check_finished(conf, epoch, best_epoch, valid_loss, best_valid_loss)
        with open(os.path.join(checkpoint_directory, 'best_checkpoint'), 'w+') as lid:
            lid.write(str(best_epoch))
        with open(os.path.join(checkpoint_directory, 'best_valid_loss'), 'w+') as lid:
            lid.write("%.3f" % best_valid_loss)
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

    # # clean up checkpoints
    # if os.path.exists(os.path.join(checkpoint_directory, 'best_checkpoint')):
    #     with open(os.path.join(checkpoint_directory, 'best_checkpoint'), 'r') as pid:
    #         best_checkpoint = (pid.readline()).rstrip()
    #
    #     # delete other checkpoints to save space
    #     cp_list = os.listdir(checkpoint_directory)
    #     for cp in cp_list:
    #         if (not cp.startswith('ckpt-' + str(best_checkpoint))) and (not cp == 'best_checkpoint') and (
    #         not cp == 'checkpoint') and (not cp == 'best_valid_loss'):
    #             os.remove(os.path.join(checkpoint_directory, cp))


@tf.function
def train_step(model, x, y, n, bReg, cReg, optimizer, train_loss, num_seqs):
    """
    train fhvae step by step and compute the gradients from the losses
    """
    print('tracing...')

    with tf.GradientTape() as tape:

        mu2, mu1, qz2_x, z2_sample, z2_sample_0, qz1_x, z1_sample, z1_sample_0, px_z, x_sample, z1_rlogits, z2_rlogits, z1_advrlogits, z2_advrlogits = model(x, y, bReg[:, 0])

        loss, log_pmu2, log_pmu1, neg_kld_z2, neg_kld_z1, log_px_z, lb, log_qy_mu2, log_qy_mu1, log_b_loss, log_c_loss, advlog_b_loss, advlog_c_loss, noise_loss = model.compute_loss(x, y, n, bReg, cReg, mu2, mu1, qz2_x, z2_sample, qz1_x, z1_sample, px_z, x_sample, z1_rlogits, z2_rlogits, z1_advrlogits, z2_advrlogits, num_seqs)

    # apply gradients
    gradients = tape.gradient(loss, model.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.NONE)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    #print(gradients, model.trainable_variables)

    # update keras mean loss metric
    train_loss(loss)

    return loss, log_pmu2, log_pmu1, neg_kld_z2, neg_kld_z1, log_px_z, lb, log_qy_mu2, log_qy_mu1, log_b_loss, log_c_loss, advlog_b_loss, advlog_c_loss, noise_loss


def estimate_mu1_mu2_dict(model, dataset, mu1_table, mu2_table, nphones, nsegs, tr_shape):
    '''
    Calculate mu1/mu2 dict tables for every phone/sequence with posterior inference formulae
    '''

    for yval, xval, _, _, talab, _ in dataset:
        z1_mu, _, _, _, z2_mu, _, _, _, _, _ = model.encoder(tf.reshape(xval, [-1, tr_shape[0], tr_shape[1]]))

        # phon_vecs: [num_phones x batch_size],  z1_mu: [batch_size x z1_dim]
        phon_vecs = tf.one_hot(talab[:, 0], depth=nphones.shape[0], axis=0, dtype=tf.float32)
        mu1_table += tf.matmul(phon_vecs, z1_mu)
        nphones += tf.reduce_sum(phon_vecs, axis=1)

        # y_br: [#nmu2_seqs x batch_size]
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

        mu2, mu1, qz2_x, z2_sample, z2_sample_0, qz1_x, z1_sample, z1_sample_0, px_z, x_sample, z1_rlogits, z2_rlogits, z1_advrlogits, z2_advrlogits = model(xval, yval, bval[:, 0])

        loss, log_pmu2, log_pmu1, neg_kld_z2, neg_kld_z1, log_px_z, lb, log_qy_mu2, log_qy_mu1, log_b_loss, log_c_loss, advlog_b_loss, advlog_c_loss, noise_loss = model.compute_loss(tf.cast(xval, dtype=tf.float32), yval, tf.cast(nval, dtype=tf.float32), bval, cval, mu2, mu1, qz2_x,
                               z2_sample, qz1_x, z1_sample, px_z, x_sample, z1_rlogits, z2_rlogits, z1_advrlogits, z2_advrlogits, num_val_seqs)

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

