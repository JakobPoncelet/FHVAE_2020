from __future__ import division
import os
import sys
import time
import pickle
import numpy as np
import tensorflow as tf
from scripts.train.hs_train_loaders import load_data_reg
from fhvae.datasets.seq2seg import seq_to_seg_mapper

def save_all_vars(expdir, model, conf, trainconf, tt_iterator_by_seqs, tt_dset):

    # whether to save the latent variables for the test set only or for train+dev as well
    # test and validation set are assumed to have < #nmu2 sequences...
    write_test = True
    write_dev = True
    write_train = True

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

    if write_train or write_dev:
        tr_nseqs, tr_shape, tr_iterator_by_seqs, dt_iterator, dt_dset, tr_dset = \
            load_data_reg(trainconf['dataset'], trainconf['fac_root'], trainconf['facs'], trainconf['talabs'])

        print('Seg_len and Seg_shifts.... (save_latent_vars.py)')
        print('     train seg_len: ',tr_dset.seg_len)
        print('     train seg_shift: ', tr_dset.seg_shift)
        print('     dev seg_len: ', dt_dset.seg_len)
        print('     dev seg_shift: ', dt_dset.seg_shift)
        print('     test seg_len: ', tt_dset.seg_len)
        print('     test seg_shift: ', tt_dset.seg_shift)

    if write_train:
        def segment_map_fn_train(idx, data):
            # map every sequence to <variable #> segments
            # + use enumeration-number as sequence key such that all keys in range [0, nmu2]
            keys, feats, lens, labs, talabs = data
            keys, feats, lens, labs, talabs, starts = seq_to_seg_mapper(idx, feats, lens, labs, talabs, \
                                                                tr_dset.seg_len, tr_dset.seg_shift, False, len(conf['b_n']))

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
                .map(segment_map_fn_train, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
                .unbatch() \
                .batch(batch_size=conf['batch_size'], drop_remainder=True) \
                .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

            return Segment_Dataset
        train_save_loc_z1 = os.path.join(expdir, 'latent_vars', 'train', 'z1')
        train_save_loc_z2 = os.path.join(expdir, 'latent_vars', 'train', 'z2')
        train_save_loc_segs = os.path.join(expdir, 'latent_vars', 'train')
        os.makedirs(train_save_loc_z1, exist_ok=True)
        os.makedirs(train_save_loc_z2, exist_ok=True)

    if write_dev:
        def segment_map_fn_dev(idx, data):
            # map every sequence to <variable #> segments
            # + use enumeration-number as sequence key such that all keys in range [0, nmu2]
            keys, feats, lens, labs, talabs = data
            keys, feats, lens, labs, talabs, starts = seq_to_seg_mapper(idx, feats, lens, labs, talabs, \
                                                                dt_dset.seg_len, dt_dset.seg_shift, False, len(conf['b_n']))
            return keys, feats, lens, labs, talabs, starts
        def create_dev_dataset():
            # initialize the validation dataset, assumed to have #sequences < nmu2
            # otherwise work with num_steps and split in sequence/segment dataset like with training dataset
            Validation_Dataset = tf.data.Dataset.from_generator(
                lambda: dt_iterator(bs=conf['batch_size']), \
                output_shapes=((), (None, conf['tr_shape'][1]), (), (len(conf['c_n']),), (len(conf['b_n']), 3, None)), \
                output_types=(tf.int64, tf.float32, tf.int64, tf.int64, tf.int64)) \
                .enumerate(start=0) \
                .map(segment_map_fn_dev, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
                .unbatch() \
                .batch(batch_size=conf['batch_size']) \
                .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

            return Validation_Dataset
        dev_save_loc_z1 = os.path.join(expdir, 'latent_vars', 'dev', 'z1')
        dev_save_loc_z2 = os.path.join(expdir, 'latent_vars', 'dev', 'z2')
        dev_save_loc_segs = os.path.join(expdir, 'latent_vars', 'dev')
        os.makedirs(dev_save_loc_z1, exist_ok=True)
        os.makedirs(dev_save_loc_z2, exist_ok=True)


    if write_test:
        def segment_map_fn_test(idx, data):
            # map every sequence to <variable #> segments
            # + use enumeration-number as sequence key such that all keys in range [0, nmu2]
            keys, feats, lens, labs, talabs = data
            keys, feats, lens, labs, talabs, starts = seq_to_seg_mapper(idx, feats, lens, labs, talabs, \
                                                                tt_dset.seg_len, tt_dset.seg_shift, False,
                                                                len(conf['b_n']))
            return keys, feats, lens, labs, talabs, starts
        def create_test_dataset():

            Test_Dataset = tf.data.Dataset.from_generator(
                lambda: tt_iterator_by_seqs(tt_dset.seqlist, bs=conf['batch_size']), \
                output_shapes=((), (None, conf['tr_shape'][1]), (), (len(conf['c_n']),), (len(conf['b_n']), 3, None)), \
                output_types=(tf.int64, tf.float32, tf.int64, tf.int64, tf.int64)) \
                .enumerate(start=0) \
                .map(segment_map_fn_test, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
                .unbatch() \
                .batch(batch_size=conf['batch_size']) \
                .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

            return Test_Dataset
        test_save_loc_z1 = os.path.join(expdir, 'latent_vars', 'test', 'z1')
        test_save_loc_z2 = os.path.join(expdir, 'latent_vars', 'test', 'z2')
        test_save_loc_segs = os.path.join(expdir, 'latent_vars', 'test')
        os.makedirs(test_save_loc_z1, exist_ok=True)
        os.makedirs(test_save_loc_z2, exist_ok=True)

    if write_train:
        print('Writing latent variables of trainset...')
        num_steps = len(tr_dset.seqlist) // conf['nmu2']

        seg_starts = dict()
        seg_talabs = dict()

        for step in range(0, num_steps + 1):
            print('step=',step)
            seqs = list(tr_dset.seqlist[step * conf['nmu2']:min((step + 1) * conf['nmu2'], len(tr_dset.seqlist))])
            Train_Dataset = create_train_dataset(seqs)

            # initialize
            mu1_table = tf.zeros([conf['num_phones'], conf['z1_dim']], dtype=tf.float32)
            mu2_table = tf.zeros([conf['nmu2'], conf['z2_dim']], dtype=tf.float32)
            nsegs = tf.zeros([conf['nmu2']])
            phone_occs = tf.zeros([conf['num_phones']])

            # calculate mu1-dict and mu2-dict
            mu1_table, mu2_table, phone_occs = estimate_mu1_mu2_dict(model, Train_Dataset, mu1_table, mu2_table, phone_occs, nsegs, conf['tr_shape'])
            model.mu1_table.assign(mu1_table)
            model.mu2_table.assign(mu2_table)
            model.phone_occs.assign(phone_occs)

            seq_cnt = 0
            seg_cnt = 0

            names = []
            z1 = []
            z2 = []

            for yval, xval, _, cval, bval, start in Train_Dataset:

                mu2, mu1, qz2_x, z2_sample, qz1_x, z1_sample, px_z, x_sample, z1_rlogits, z2_rlogits = model(xval, yval, bval[:, 0])

                for seg in range(0, yval.get_shape().as_list()[0]):
                    z1.append(z1_sample[seg, :])
                    z2.append(z2_sample[seg, :])
                    seg_start = start[seg]

                    if len(bval.shape) > 1:  # if multiple talabs used, only use phones
                        seg_talab = bval[seg, 0]
                    else:
                        seg_talab = bval[seg]

                    if yval[seg] > seq_cnt:
                        seq_cnt += 1
                        seg_cnt = 0

                    seg_name = str(tr_dset.seqlist[seq_cnt+step*conf['nmu2']]) + '_' + str(seg_cnt)
                    names.append(seg_name)

                    seg_starts[seg_name] = seg_start
                    seg_talabs[seg_name] = seg_talab

                    seg_cnt += 1

            for idx, name in enumerate(names):
                with open(os.path.join(train_save_loc_z1, name + '_z1.npy'), 'wb') as fnp:
                    np.save(fnp, z1[idx])
                with open(os.path.join(train_save_loc_z2, name + '_z2.npy'), 'wb') as fnp:
                    np.save(fnp, z2[idx])

        with open(os.path.join(train_save_loc_segs, 'segments.txt'), "w") as fid:
            fid.write('Segment_name Segment_start Centered_talab\n')
        for seg in seg_starts:
            tf.print(str(seg), seg_starts[seg], seg_talabs[seg], output_stream='file://'+str(os.path.join(train_save_loc_segs, 'segments.txt')))

    if write_dev:
        print('Writing latent variables of validation set...')
        Validation_Dataset = create_dev_dataset()

        mu1_table = tf.zeros([conf['num_phones'], conf['z1_dim']], dtype=tf.float32)
        mu2_table = tf.zeros([conf['nmu2'], conf['z2_dim']], dtype=tf.float32)
        nsegs = tf.zeros([conf['nmu2']])
        phone_occs = tf.zeros([conf['num_phones']])

        # calculate mu1-dict and mu2-dict
        mu1_table, mu2_table, phone_occs = estimate_mu1_mu2_dict(model, Validation_Dataset, mu1_table, mu2_table, phone_occs,
                                                                 nsegs, conf['tr_shape'])
        model.mu1_table.assign(mu1_table)
        model.mu2_table.assign(mu2_table)
        model.phone_occs.assign(phone_occs)

        seq_cnt = 0
        seg_cnt = 0

        names = []
        z1 = []
        z2 = []

        seg_starts = dict()
        seg_talabs = dict()

        for yval, xval, _, cval, bval, start in Validation_Dataset:

            mu2, mu1, qz2_x, z2_sample, qz1_x, z1_sample, px_z, x_sample, z1_rlogits, z2_rlogits = model(xval, yval, bval[:, 0])

            for seg in range(0, yval.get_shape().as_list()[0]):
                z1.append(z1_sample[seg, :])
                z2.append(z2_sample[seg, :])
                seg_start = start[seg]

                if len(bval.shape) > 1:  # if multiple talabs used, only use phones
                    seg_talab = bval[seg, 0]
                else:
                    seg_talab = bval[seg]

                if yval[seg] > seq_cnt:
                    seq_cnt += 1
                    seg_cnt = 0

                seg_name = str(dt_dset.seqlist[seq_cnt]) + '_' + str(seg_cnt)
                names.append(seg_name)

                seg_starts[seg_name] = seg_start
                seg_talabs[seg_name] = seg_talab

                seg_cnt += 1

        for idx, name in enumerate(names):
            with open(os.path.join(dev_save_loc_z1, name + '_z1.npy'), 'wb') as fnp:
                np.save(fnp, z1[idx])
            with open(os.path.join(dev_save_loc_z2, name + '_z2.npy'), 'wb') as fnp:
                np.save(fnp, z2[idx])

        with open(os.path.join(dev_save_loc_segs, 'segments.txt'), "w") as fid:
            fid.write('Segment_name Segment_start Centered_talab\n')

        for seg in seg_starts:
            tf.print(str(seg), seg_starts[seg], seg_talabs[seg], output_stream='file://'+str(os.path.join(dev_save_loc_segs, 'segments.txt')))

    if write_test:
        print('Writing latent variables of test set...')
        Test_Dataset = create_test_dataset()

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

        seq_cnt = 0
        seg_cnt = 0

        names = []
        z1 = []
        z2 = []

        seg_starts = dict()
        seg_talabs = dict()

        for yval, xval, _, cval, bval, start in Test_Dataset:

            mu2, mu1, qz2_x, z2_sample, qz1_x, z1_sample, px_z, x_sample, z1_rlogits, z2_rlogits = model(xval, yval, bval[:, 0])

            for seg in range(0, yval.get_shape().as_list()[0]):
                z1.append(z1_sample[seg, :])
                z2.append(z2_sample[seg, :])
                seg_start = start[seg]

                if len(bval.shape) > 1:  # if multiple talabs used, only use phones
                    seg_talab = bval[seg, 0]
                else:
                    seg_talab = bval[seg]

                if yval[seg] > seq_cnt:
                    seq_cnt += 1
                    seg_cnt = 0

                seg_name = str(tt_dset.seqlist[seq_cnt])+'_'+str(seg_cnt)
                names.append(seg_name)

                seg_starts[seg_name] = seg_start
                seg_talabs[seg_name] = seg_talab

                seg_cnt += 1

        for idx, name in enumerate(names):
            with open(os.path.join(test_save_loc_z1, name+'_z1.npy'), 'wb') as fnp:
                np.save(fnp, z1[idx])
            with open(os.path.join(test_save_loc_z2, name+'_z2.npy'), 'wb') as fnp:
                np.save(fnp, z2[idx])

        with open(os.path.join(test_save_loc_segs, 'segments.txt'), "w") as fid:
            fid.write('Segment_name Segment_start Centered_talab\n')

        for seg in seg_starts:
            tf.print(str(seg), seg_starts[seg], seg_talabs[seg], output_stream='file://'+str(os.path.join(test_save_loc_segs, 'segments.txt')))


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
