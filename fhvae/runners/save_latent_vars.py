from __future__ import division
import os
import sys
import time
import pickle
import numpy as np
import tensorflow as tf
import json
from scripts.train.hs_train_loaders import load_data_reg
from fhvae.datasets.seq2seg import seq_to_seg_mapper

def save_all_vars(expdir, model, conf, trainconf, tt_iterator_by_seqs, tt_dset):

###################################################################################################################
    # whether to save the latent variables for the test set only or for train/dev as well
    write_train_z1 = True
    write_train_z2 = False

    write_dev_z1 = True
    write_dev_z2 = False

    write_test_z1 = True
    write_test_z2 = False

    # which variable to write out per segment
    to_save = ['zmu']  # ['zsample', 'z0', 'zmu']

    # don't write out latent_vars of augmented files
    skip_aug_segs = False


    if os.path.exists(os.path.join(expdir, "training_checkpoints_finetuning")):
        checkpoint_directory = os.path.join(expdir, 'training_checkpoints_finetuning')
    else:
        checkpoint_directory = os.path.join(expdir, 'training_checkpoints')
    ###################################################################################################################

    print("\nRESTORING MODEL")
    starttime = time.time()
    optimizer = tf.keras.optimizers.Adam(learning_rate=conf['lr'], beta_1=conf['beta1'], beta_2=conf['beta2'], epsilon=conf['adam_eps'], amsgrad=False)

    checkpoint = tf.train.Checkpoint(model=model)
    # checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)

    if os.path.exists(os.path.join(checkpoint_directory, 'best_checkpoint')):
        with open(os.path.join(checkpoint_directory, 'best_checkpoint'), 'r') as pid:
            best_checkpoint = (pid.readline()).rstrip()
        status = checkpoint.restore(os.path.join(checkpoint_directory, 'ckpt-' + str(best_checkpoint)))
    else:
        manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_directory, max_to_keep=5)
        status = checkpoint.restore(manager.latest_checkpoint)

    print("restoring model takes %.2f seconds" % (time.time()-starttime))
    status.assert_existing_objects_matched()
    # status.assert_consumed()

    if write_train_z1 or write_train_z2 or write_dev_z1 or write_dev_z2:
        tr_nseqs, tr_shape, tr_iterator_by_seqs, dt_iterator, dt_dset, tr_dset = \
            load_data_reg(trainconf['dataset'], trainconf['fac_root'], trainconf['facs'], trainconf['talabs'], mvn=conf['mvn'])

        print('Seg_len and Seg_shifts.... (save_latent_vars.py)')
        print('     train seg_len: ', tr_dset.seg_len)
        print('     train seg_shift: ', tr_dset.seg_shift)
        print('     dev seg_len: ', dt_dset.seg_len)
        print('     dev seg_shift: ', dt_dset.seg_shift)
        print('     test seg_len: ', tt_dset.seg_len)
        print('     test seg_shift: ', tt_dset.seg_shift)

    if write_train_z1 or write_train_z2:

        def segment_map_fn_train(idx, data):
            # map every sequence to <variable #> segments
            # + use enumeration-number as sequence key such that all keys in range [0, nmu2]
            keys, feats, lens, labs, talabs = data
            keys, feats, lens, labs, talabs, starts = seq_to_seg_mapper(idx, keys, feats, lens, labs, talabs, \
                                                                tr_dset.seg_len, tr_dset.seg_shift, False, len(conf['b_n']), 0)

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

    if write_dev_z1 or write_dev_z2:

        def segment_map_fn_dev(idx, data):
            # map every sequence to <variable #> segments
            # + use enumeration-number as sequence key such that all keys in range [0, nmu2]
            keys, feats, lens, labs, talabs = data
            keys, feats, lens, labs, talabs, starts = seq_to_seg_mapper(idx, keys, feats, lens, labs, talabs, \
                                                                dt_dset.seg_len, dt_dset.seg_shift, False, len(conf['b_n']), 0)
            return keys, feats, lens, labs, talabs, starts

        def create_dev_dataset(seqs):
            # initialize the validation dataset, assumed to have #sequences < nmu2
            # otherwise work with num_steps and split in sequence/segment dataset like with training dataset
            Validation_Dataset = tf.data.Dataset.from_generator(
                lambda: dt_iterator(s_seqs=seqs, bs=conf['batch_size']), \
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

    if write_test_z1 or write_test_z2:

        def segment_map_fn_test(idx, data):
            # map every sequence to <variable #> segments
            # + use enumeration-number as sequence key such that all keys in range [0, nmu2]
            keys, feats, lens, labs, talabs = data
            keys, feats, lens, labs, talabs, starts = seq_to_seg_mapper(idx, keys, feats, lens, labs, talabs, \
                                                                tt_dset.seg_len, tt_dset.seg_shift, False,
                                                                len(conf['b_n']), 0)
            return keys, feats, lens, labs, talabs, starts

        def create_test_dataset(seqs):

            Test_Dataset = tf.data.Dataset.from_generator(
                lambda: tt_iterator_by_seqs(s_seqs=seqs, bs=conf['batch_size']), \
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

    if write_train_z1:
        print('Writing Z1 latent variables of trainset...')
        num_steps = len(tr_dset.seqlist) // conf['nmu2']

        with open(os.path.join(train_save_loc_segs, 'segments.txt'), "w") as fid:
            fid.write('Segment_name Segment_start Centered_talab\n')

        z1_dict = dict()
        z2_dict = dict()
        z1_0_dict = dict()
        z2_0_dict = dict()
        z1_mu_dict = dict()
        z2_mu_dict = dict()

        for step in range(0, num_steps + 1):
            print('step=', step)
            seqs = list(tr_dset.seqlist[step * conf['nmu2']:min((step + 1) * conf['nmu2'], len(tr_dset.seqlist))])
            Train_Dataset = create_train_dataset(seqs)

            seq_cnt = 0
            seg_cnt = 0

            names = []
            z1 = []
            z2 = []
            z1_0 = []
            z2_0 = []
            z1_mu = []
            z2_mu = []

            for yval, xval, _, cval, bval, start in Train_Dataset:

                _, _, z1_sample, z1_sample_0, _, _, _, _, qz1_x, _ = model.encoder(xval)

                for seg in range(0, yval.get_shape().as_list()[0]):

                    if yval[seg] > seq_cnt:
                        seq_cnt += 1
                        seg_cnt = 0

                    seg_name = str(tr_dset.seqlist[seq_cnt + step * conf['nmu2']]) + '_' + str(seg_cnt)

                    if skip_aug_segs:
                        if 'dB' in seg_name:
                            seg_cnt += 1
                            continue

                    if 'zsample' in to_save:
                        z1.append(z1_sample[seg, :])
                    if 'z0' in to_save:
                        z1_0.append(z1_sample_0[seg, :])
                    if 'zmu' in to_save:
                        z1_mu.append(qz1_x[0][seg, :])

                    seg_start = start[seg]

                    if len(bval.shape) > 1:  # if multiple talabs used, only use phones
                        seg_talab = bval[seg, 0]
                    else:
                        seg_talab = bval[seg]

                    names.append(seg_name)

                    tf.print(str(seg_name), seg_start, seg_talab,
                             output_stream='file://' + str(os.path.join(train_save_loc_segs, 'segments.txt')))

                    seg_cnt += 1

            for idx, name in enumerate(names):
                if 'zsample' in to_save:
                    z1_dict[name] = z1[idx]
                if 'z0' in to_save:
                    z1_0_dict[name] = z1_0[idx]
                if 'zmu' in to_save:
                    z1_mu_dict[name] = z1_mu[idx]

        if 'zsample' in to_save:
            with open(os.path.join(train_save_loc_z1, 'z1_dict.pickle'), 'wb') as handle_z1:
                pickle.dump(z1_dict, handle_z1, protocol=pickle.HIGHEST_PROTOCOL)

        if 'z0' in to_save:
            with open(os.path.join(train_save_loc_z1, 'z1_0_dict.pickle'), 'wb') as handle_z1:
                pickle.dump(z1_0_dict, handle_z1, protocol=pickle.HIGHEST_PROTOCOL)

        if 'zmu' in to_save:
            with open(os.path.join(train_save_loc_z1, 'z1_mu_dict.pickle'), 'wb') as handle_z1:
                pickle.dump(z1_mu_dict, handle_z1, protocol=pickle.HIGHEST_PROTOCOL)

    if write_train_z2:
        print('Writing Z2 latent variables of trainset...')
        num_steps = len(tr_dset.seqlist) // conf['nmu2']

        z1_dict = dict()
        z2_dict = dict()
        z1_0_dict = dict()
        z2_0_dict = dict()
        z1_mu_dict = dict()
        z2_mu_dict = dict()

        for step in range(0, num_steps + 1):
            print('step=', step)
            seqs = list(tr_dset.seqlist[step * conf['nmu2']:min((step + 1) * conf['nmu2'], len(tr_dset.seqlist))])
            Train_Dataset = create_train_dataset(seqs)

            seq_cnt = 0
            seg_cnt = 0

            names = []
            z1 = []
            z2 = []
            z1_0 = []
            z2_0 = []
            z1_mu = []
            z2_mu = []

            for yval, xval, _, _, _, _ in Train_Dataset:

                _, _, _, _, _, _, z2_sample, z2_sample_0, _, qz2_x = model.encoder(xval)

                for seg in range(0, yval.get_shape().as_list()[0]):

                    if yval[seg] > seq_cnt:
                        seq_cnt += 1
                        seg_cnt = 0

                    seg_name = str(tr_dset.seqlist[seq_cnt + step * conf['nmu2']]) + '_' + str(seg_cnt)

                    if skip_aug_segs:
                        if 'dB' in seg_name:
                            seg_cnt += 1
                            continue

                    if 'zsample' in to_save:
                        z2.append(z2_sample[seg, :])
                    if 'z0' in to_save:
                        z2_0.append(z2_sample_0[seg, :])
                    if 'zmu' in to_save:
                        z2_mu.append(qz2_x[0][seg, :])

                    names.append(seg_name)

                    seg_cnt += 1

            for idx, name in enumerate(names):
                if 'zsample' in to_save:
                    z2_dict[name] = z2[idx]
                if 'z0' in to_save:
                    z2_0_dict[name] = z2_0[idx]
                if 'zmu' in to_save:
                    z2_mu_dict[name] = z2_mu[idx]

        if 'zsample' in to_save:
            with open(os.path.join(train_save_loc_z2, 'z2_dict.pickle'), 'wb') as handle_z2:
                pickle.dump(z2_dict, handle_z2, protocol=pickle.HIGHEST_PROTOCOL)

        if 'z0' in to_save:
            with open(os.path.join(train_save_loc_z2, 'z2_0_dict.pickle'), 'wb') as handle_z2:
                pickle.dump(z2_0_dict, handle_z2, protocol=pickle.HIGHEST_PROTOCOL)

        if 'zmu' in to_save:
            with open(os.path.join(train_save_loc_z2, 'z2_mu_dict.pickle'), 'wb') as handle_z2:
                pickle.dump(z2_mu_dict, handle_z2, protocol=pickle.HIGHEST_PROTOCOL)

    if write_dev_z1:
        print('Writing Z1 latent variables of validation set...')
        num_steps = len(dt_dset.seqlist) // conf['nmu2']

        with open(os.path.join(dev_save_loc_segs, 'segments.txt'), "w") as fid:
            fid.write('Segment_name Segment_start Centered_talab\n')

        z1_dict = dict()
        z2_dict = dict()
        z1_0_dict = dict()
        z2_0_dict = dict()
        z1_mu_dict = dict()
        z2_mu_dict = dict()

        for step in range(0, num_steps+1):
            print('step=', step)
            seqs = list(dt_dset.seqlist[step * conf['nmu2']:min((step + 1) * conf['nmu2'], len(dt_dset.seqlist))])
            Validation_Dataset = create_dev_dataset(seqs)
            seq_cnt = 0
            seg_cnt = 0

            names = []
            z1 = []
            z2 = []
            z1_0 = []
            z2_0 = []
            z1_mu = []
            z2_mu = []

            for yval, xval, _, cval, bval, start in Validation_Dataset:

                _, _, z1_sample, z1_sample_0, _, _, _, _, qz1_x, _ = model.encoder(xval)

                for seg in range(0, yval.get_shape().as_list()[0]):

                    if yval[seg] > seq_cnt:
                        seq_cnt += 1
                        seg_cnt = 0

                    seg_name = str(dt_dset.seqlist[seq_cnt + step * conf['nmu2']]) + '_' + str(seg_cnt)

                    if skip_aug_segs:
                        if 'dB' in seg_name:
                            seg_cnt += 1
                            continue

                    if 'zsample' in to_save:
                        z1.append(z1_sample[seg, :])
                    if 'z0' in to_save:
                        z1_0.append(z1_sample_0[seg, :])
                    if 'zmu' in to_save:
                        z1_mu.append(qz1_x[0][seg, :])

                    seg_start = start[seg]

                    if len(bval.shape) > 1:  # if multiple talabs used, only use phones
                        seg_talab = bval[seg, 0]
                    else:
                        seg_talab = bval[seg]

                    names.append(seg_name)

                    tf.print(str(seg_name), seg_start, seg_talab,
                             output_stream='file://' + str(os.path.join(dev_save_loc_segs, 'segments.txt')))

                    seg_cnt += 1

            for idx, name in enumerate(names):
                if 'zsample' in to_save:
                    z1_dict[name] = z1[idx]
                if 'z0' in to_save:
                    z1_0_dict[name] = z1_0[idx]
                if 'zmu' in to_save:
                    z1_mu_dict[name] = z1_mu[idx]

        if 'zsample' in to_save:
            with open(os.path.join(dev_save_loc_z1, 'z1_dict.pickle'), 'wb') as handle_z1:
                pickle.dump(z1_dict, handle_z1, protocol=pickle.HIGHEST_PROTOCOL)
        if 'z0' in to_save:
            with open(os.path.join(dev_save_loc_z1, 'z1_0_dict.pickle'), 'wb') as handle_z1:
                pickle.dump(z1_0_dict, handle_z1, protocol=pickle.HIGHEST_PROTOCOL)
        if 'zmu' in to_save:
            with open(os.path.join(dev_save_loc_z1, 'z1_mu_dict.pickle'), 'wb') as handle_z1:
                pickle.dump(z1_mu_dict, handle_z1, protocol=pickle.HIGHEST_PROTOCOL)


    if write_dev_z2:
        print('Writing Z2 latent variables of validation set...')
        num_steps = len(dt_dset.seqlist) // conf['nmu2']

        z1_dict = dict()
        z2_dict = dict()
        z1_0_dict = dict()
        z2_0_dict = dict()
        z1_mu_dict = dict()
        z2_mu_dict = dict()

        for step in range(0, num_steps + 1):
            print('step=', step)
            seqs = list(dt_dset.seqlist[step * conf['nmu2']:min((step + 1) * conf['nmu2'], len(dt_dset.seqlist))])

            Validation_Dataset = create_dev_dataset(seqs)

            seq_cnt = 0
            seg_cnt = 0

            names = []
            z1 = []
            z2 = []
            z1_0 = []
            z2_0 = []
            z1_mu = []
            z2_mu = []

            for yval, xval, _, _, _, _ in Validation_Dataset:

                _, _, _, _, _, _, z2_sample, z2_sample_0, _, qz2_x = model.encoder(xval)

                for seg in range(0, yval.get_shape().as_list()[0]):

                    if yval[seg] > seq_cnt:
                        seq_cnt += 1
                        seg_cnt = 0

                    seg_name = str(dt_dset.seqlist[seq_cnt + step * conf['nmu2']]) + '_' + str(seg_cnt)

                    if skip_aug_segs:
                        if 'dB' in seg_name:
                            seg_cnt += 1
                            continue

                    if 'zsample' in to_save:
                        z2.append(z2_sample[seg, :])
                    if 'z0' in to_save:
                        z2_0.append(z2_sample_0[seg, :])
                    if 'zmu' in to_save:
                        z2_mu.append(qz2_x[0][seg, :])

                    names.append(seg_name)

                    seg_cnt += 1

            for idx, name in enumerate(names):
                if 'zsample' in to_save:
                    z2_dict[name] = z2[idx]
                if 'z0' in to_save:
                    z2_0_dict[name] = z2_0[idx]
                if 'zmu' in to_save:
                    z2_mu_dict[name] = z2_mu[idx]

        if 'zsample' in to_save:
            with open(os.path.join(dev_save_loc_z2, 'z2_dict.pickle'), 'wb') as handle_z2:
                pickle.dump(z2_dict, handle_z2, protocol=pickle.HIGHEST_PROTOCOL)

        if 'z0' in to_save:
            with open(os.path.join(dev_save_loc_z2, 'z2_0_dict.pickle'), 'wb') as handle_z2:
                pickle.dump(z2_0_dict, handle_z2, protocol=pickle.HIGHEST_PROTOCOL)

        if 'zmu' in to_save:
            with open(os.path.join(dev_save_loc_z2, 'z2_mu_dict.pickle'), 'wb') as handle_z2:
                pickle.dump(z2_mu_dict, handle_z2, protocol=pickle.HIGHEST_PROTOCOL)


    if write_test_z1:
        print('Writing Z1 latent variables of test set...')
        num_steps = len(tt_dset.seqlist) // conf['nmu2']

        with open(os.path.join(test_save_loc_segs, 'segments.txt'), "w") as fid:
            fid.write('Segment_name Segment_start Centered_talab\n')

        z1_dict = dict()
        z2_dict = dict()
        z1_0_dict = dict()
        z2_0_dict = dict()
        z1_mu_dict = dict()
        z2_mu_dict = dict()

        for step in range(0, num_steps + 1):
            print('step=', step)
            seqs = list(tt_dset.seqlist[step * conf['nmu2']:min((step + 1) * conf['nmu2'], len(tt_dset.seqlist))])

            Test_Dataset = create_test_dataset(seqs)

            seq_cnt = 0
            seg_cnt = 0

            names = []
            z1 = []
            z2 = []
            z1_0 = []
            z2_0 = []
            z1_mu = []
            z2_mu = []

            for yval, xval, _, cval, bval, start in Test_Dataset:

                _, _, z1_sample, z1_sample_0, _, _, _, _, qz1_x, _ = model.encoder(xval)

                for seg in range(0, yval.get_shape().as_list()[0]):

                    if yval[seg] > seq_cnt:
                        seq_cnt += 1
                        seg_cnt = 0

                    seg_name = str(tt_dset.seqlist[seq_cnt+step*conf['nmu2']])+'_'+str(seg_cnt)

                    if skip_aug_segs:
                        if 'dB' in seg_name:
                            seg_cnt += 1
                            continue

                    if 'zsample' in to_save:
                        z1.append(z1_sample[seg, :])
                    if 'z0' in to_save:
                        z1_0.append(z1_sample_0[seg, :])
                    if 'zmu' in to_save:
                        z1_mu.append(qz1_x[0][seg, :])

                    seg_start = start[seg]

                    if len(bval.shape) > 1:  # if multiple talabs used, only use phones
                        seg_talab = bval[seg, 0]
                    else:
                        seg_talab = bval[seg]

                    names.append(seg_name)

                    tf.print(str(seg_name), seg_start, seg_talab,
                             output_stream='file://' + str(os.path.join(test_save_loc_segs, 'segments.txt')))

                    seg_cnt += 1

            for idx, name in enumerate(names):
                if 'zsample' in to_save:
                    z1_dict[name] = z1[idx]
                if 'z0' in to_save:
                    z1_0_dict[name] = z1_0[idx]
                if 'zmu' in to_save:
                    z1_mu_dict[name] = z1_mu[idx]

        if 'zsample' in to_save:
            with open(os.path.join(test_save_loc_z1, 'z1_dict.pickle'), 'wb') as handle_z1:
                pickle.dump(z1_dict, handle_z1, protocol=pickle.HIGHEST_PROTOCOL)
        if 'z0' in to_save:
            with open(os.path.join(test_save_loc_z1, 'z1_0_dict.pickle'), 'wb') as handle_z1:
                pickle.dump(z1_0_dict, handle_z1, protocol=pickle.HIGHEST_PROTOCOL)
        if 'zmu' in to_save:
            with open(os.path.join(test_save_loc_z1, 'z1_mu_dict.pickle'), 'wb') as handle_z1:
                pickle.dump(z1_mu_dict, handle_z1, protocol=pickle.HIGHEST_PROTOCOL)

    if write_test_z2:
        print('Writing Z2 latent variables of test set...')
        num_steps = len(tt_dset.seqlist) // conf['nmu2']

        z1_dict = dict()
        z2_dict = dict()
        z1_0_dict = dict()
        z2_0_dict = dict()
        z1_mu_dict = dict()
        z2_mu_dict = dict()

        for step in range(0, num_steps + 1):
            print('step=', step)
            seqs = list(tt_dset.seqlist[step * conf['nmu2']:min((step + 1) * conf['nmu2'], len(tt_dset.seqlist))])

            Test_Dataset = create_test_dataset(seqs)

            seq_cnt = 0
            seg_cnt = 0

            names = []
            z1 = []
            z2 = []
            z1_0 = []
            z2_0 = []
            z1_mu = []
            z2_mu = []

            for yval, xval, _, _, _, _ in Test_Dataset:

                _, _, _, _, _, _, z2_sample, z2_sample_0, _, qz2_x = model.encoder(xval)

                for seg in range(0, yval.get_shape().as_list()[0]):

                    if yval[seg] > seq_cnt:
                        seq_cnt += 1
                        seg_cnt = 0

                    seg_name = str(tt_dset.seqlist[seq_cnt + step * conf['nmu2']]) + '_' + str(seg_cnt)

                    if skip_aug_segs:
                        if 'dB' in seg_name:
                            seg_cnt += 1
                            continue

                    if 'zsample' in to_save:
                        z2.append(z2_sample[seg, :])
                    if 'z0' in to_save:
                        z2_0.append(z2_sample_0[seg, :])
                    if 'zmu' in to_save:
                        z2_mu.append(qz2_x[0][seg, :])

                    names.append(seg_name)

                    seg_cnt += 1

            for idx, name in enumerate(names):
                if 'zsample' in to_save:
                    z2_dict[name] = z2[idx]
                if 'z0' in to_save:
                    z2_0_dict[name] = z2_0[idx]
                if 'zmu' in to_save:
                    z2_mu_dict[name] = z2_mu[idx]

        if 'zsample' in to_save:
            with open(os.path.join(test_save_loc_z2, 'z2_dict.pickle'), 'wb') as handle_z2:
                pickle.dump(z2_dict, handle_z2, protocol=pickle.HIGHEST_PROTOCOL)
        if 'z0' in to_save:
            with open(os.path.join(test_save_loc_z2, 'z2_0_dict.pickle'), 'wb') as handle_z2:
                pickle.dump(z2_0_dict, handle_z2, protocol=pickle.HIGHEST_PROTOCOL)
        if 'zmu' in to_save:
            with open(os.path.join(test_save_loc_z2, 'z2_mu_dict.pickle'), 'wb') as handle_z2:
                pickle.dump(z2_mu_dict, handle_z2, protocol=pickle.HIGHEST_PROTOCOL)


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
