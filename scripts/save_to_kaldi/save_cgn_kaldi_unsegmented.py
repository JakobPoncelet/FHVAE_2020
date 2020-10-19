from __future__ import division
import os
import sys
import time
import pickle
import math
import numpy as np
import kaldiio
import bisect
from kaldiio import WriteHelper
from kaldiio import open_like_kaldi
import tensorflow as tf
from scripts.train.hs_train_loaders import load_data_reg
from fhvae.datasets.seq2seg import seq_to_seg_mapper


def save_cgn_kaldi_unsegmented(expdir, model, conf, datadir, suff, seg_len, seg_shift, lvar, speed_factor, save_numpy):

    #####
    # frame spacing of features in s
    hop_t = 0.010
    ####

    print("\nRESTORING MODEL")
    starttime = time.time()

    if os.path.exists(os.path.join(expdir,'training_checkpoints_finetuning')):
        checkpoint_directory = os.path.join(expdir, 'training_checkpoints_finetuning')
    else:
        checkpoint_directory = os.path.join(expdir, 'training_checkpoints')

    checkpoint = tf.train.Checkpoint(model=model)
    print("Checkpoint Directory: ", checkpoint_directory)
    
    if os.path.exists(os.path.join(checkpoint_directory, 'best_checkpoint')):
        print("Loading best checkpoint")
        with open(os.path.join(checkpoint_directory, 'best_checkpoint'), 'r') as pid:
            best_checkpoint = (pid.readline()).rstrip()
        status = checkpoint.restore(os.path.join(checkpoint_directory, 'ckpt-' + str(best_checkpoint)))
        print(best_checkpoint, checkpoint_directory)
    else:
        print("Loading latest checkpoint")
        manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_directory, max_to_keep=5)
        status = checkpoint.restore(manager.latest_checkpoint)

    print("restoring model takes %.2f seconds" % (time.time()-starttime))
    status.assert_existing_objects_matched()

    tr_nseqs, tr_shape, tr_iterator_by_seqs, dt_iterator, dt_dset, tr_dset = \
        load_data_reg(datadir, seg_len=seg_len, seg_shift=seg_shift, rand_seg=False, mvn=conf['mvn'])

    def segment_map_fn(idx, data):
        # map every sequence to <variable #> segments
        # + use enumeration-number as sequence key such that all keys in range [0, nmu2]
        keys, feats, lens, labs, talabs = data
        keys, feats, lens, labs, talabs, starts = seq_to_seg_mapper(idx, keys, feats, lens, labs, talabs, \
                                                            tr_dset.seg_len, tr_dset.seg_shift, False, len(conf['b_n']), 0, False)

        return keys, feats, lens, labs, talabs, starts

    def create_dataset(seqs):
        # build a tf.data.dataset for the #nmu2 seqs provided in s_seqs
        Dataset = tf.data.Dataset.from_generator(
            lambda: tr_iterator_by_seqs(s_seqs=seqs, bs=conf['batch_size'], seg_rem=True), \
            output_shapes=((), (None, conf['tr_shape'][1]), (), (0,), (0,)), \
            output_types=(tf.int64, tf.float32, tf.int64, tf.int64, tf.int64)) \
            .enumerate(start=0) \
            .map(segment_map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
            .unbatch() \
            .batch(batch_size=conf['batch_size']) \
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return Dataset

    if save_numpy:
        os.makedirs('%s/output_features_%s_%s/%s_features' % (expdir, lvar, suff, lvar), exist_ok=True)

    print('Writing latent variable %s' % lvar)

    #seqs = []
    #seqs.append(tr_dset.seqlist[-1])
    seqs = tr_dset.seqlist
    if 'test' in suff:
        nbest = True
    else:
        nbest = False

    Train_Dataset = create_dataset(seqs)
    curr_seq = 0  # start
    seg_mat = []

    for yval, xval, _, _, _, start in Train_Dataset:

        z1_mu, _, _, _, z2_mu, _, _, _, _, _ = model.encoder(xval)

        if lvar == 'z1':
            z = z1_mu
        elif lvar == 'z2':
            z = z2_mu
        elif lvar == 'z1z2':
            z = np.concatenate((z1_mu, z2_mu), axis=1)

        bs = yval.get_shape().as_list()[0]
        curr_st = 0

        while True:
            if np.searchsorted(yval, curr_seq+1) == bs:
                seg_mat.append(z)
                break

            else:
                loc = np.searchsorted(yval, curr_seq+1)
                seg_mat.append(z[curr_st:loc])
                curr_st = loc

                if all(len(el)==0 for el in seg_mat):
                    print("Discarded sequence")
                else:
                    seg_mat = np.concatenate(seg_mat, axis=0)
                    if nbest:
                        seq_name = str(seqs[curr_seq])
                    else:
                        seq_name = (str(seqs[curr_seq])).split('_')[0]
                    kaldiio.save_ark('%s/output_features_%s_%s/%s_feats.ark' % (expdir, lvar, suff, lvar),
                                     {seq_name: seg_mat},
                                     scp='%s/output_features_%s_%s/%s_feats.scp' % (expdir, lvar, suff, lvar),
                                     append=True)
                    if save_numpy:
                        with open('%s/output_features_%s_%s/%s_features/%s.npy' % (expdir, lvar, suff, lvar, seq_name), 'wb') as f:
                            np.save(f, seg_mat)

                print("Saved features of %i / %i files" % (curr_seq+1, len(seqs)))
                curr_seq += 1
                seg_mat = []

    # last batch
    if all(len(el) == 0 for el in seg_mat):
        print("Discarded sequence")
    else:
        seg_mat = np.concatenate(seg_mat, axis=0)
        if nbest:
            seq_name = str(seqs[curr_seq])
        else:
            seq_name = (str(seqs[curr_seq])).split('_')[0]
        kaldiio.save_ark('%s/output_features_%s_%s/%s_feats.ark' % (expdir, lvar, suff, lvar),
                         {seq_name: seg_mat},
                         scp='%s/output_features_%s_%s/%s_feats.scp' % (expdir, lvar, suff, lvar),
                         append=True)
        if save_numpy:
            with open('%s/output_features_%s_%s/%s_features/%s.npy' % (expdir, lvar, suff, lvar, seq_name), 'wb') as f:
                np.save(f, seg_mat)

    print("Saved features of %i / %i files" % (curr_seq + 1, len(seqs)))
    curr_seq += 1
    seg_mat = []

    # for i in range(0, bs):
    #     if yval[i] == curr_seq:
    #         seg_mat.append(z[i, :])
    #     else:
    #         if all(len(el)==0 for el in seg_mat):
    #             print("Discarded sequence")
    #         else:
    #             seg_mat = np.concatenate(seg_mat, axis=0)
    #             if nbest:
    #                 seq_name = str(seqs[curr_seq])
    #             else:
    #                 seq_name = (str(seqs[curr_seq])).split('_')[0]
    #             kaldiio.save_ark('%s/output_features_%s_%s/%s_feats.ark' % (expdir, lvar, suff, lvar),
    #                              {seq_name: seg_mat},
    #                              scp='%s/output_features_%s_%s/%s_feats.scp' % (expdir, lvar, suff, lvar),
    #                              append=True)
    #         print("Saved features of %i / %i files" % (curr_seq+1, len(seqs)))
    #         curr_seq += 1
    #         seg_mat = []
