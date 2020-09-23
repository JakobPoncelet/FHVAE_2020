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


def save_cgn_kaldi(expdir, model, conf, datadir, segments, suff, seg_len, seg_shift, lvar, speed_factor):

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
    manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_directory, max_to_keep=5)
    
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

    print("Loading segmentation")
    segm_dict = process_segmentation(segments, seg_len, seg_shift, hop_t, speed_factor)
    valid_files = set(segm_dict.keys())  # follow train/dev/set split

    print('Writing latent variable %s' % lvar)

    # CHOOSE COMPONENT
    #seqs = [seq for seq in tr_dset.seqlist if seq.split('_')[2] == 'm' and seq.split('_')[0] != 'fv400010']
    # seqs = list(tr_dset.seqlist)
    if 'nbest' in segments:
        nbest = True
        seqs = list(tr_dset.seqlist)
    else:
        nbest = False
        seqs = [seq for seq in tr_dset.seqlist if seq.split('_')[0] in valid_files]

    Train_Dataset = create_dataset(seqs)
    curr_seq = 0  # start
    curr_segidx = 0
    seg_mat = []
    init_flag = True
    new_seq_flag = False

    for yval, xval, _, _, _, start in Train_Dataset:

        z1_mu, _, _, _, z2_mu, _, _, _, _, _ = model.encoder(xval)

        if lvar == 'z1':
            z = z1_mu
        elif lvar == 'z2':
            z = z2_mu
        elif lvar == 'z1z2':
            z = np.concatenate((z1_mu, z2_mu), axis=1)

        bs = yval.get_shape().as_list()[0]

        if init_flag:
            if nbest:
                seq_name = str(seqs[curr_seq])
            else:
                seq_name = (str(seqs[curr_seq])).split('_')[0]
            curr_st = segm_dict[seq_name]['start'][curr_segidx]
            curr_dur = segm_dict[seq_name]['dur'][curr_segidx]
            curr_segid = segm_dict[seq_name]['segid'][curr_segidx]
            init_flag = False

        while True:
            if new_seq_flag:  # start of new sequence
                if np.searchsorted(yval, curr_seq) == bs:  # not in current batch
                    break
                else:
                    curr_st += np.searchsorted(yval, curr_seq)
                    new_seq_flag = False

            if curr_st > bs:  # skip batch, not in segmentation
                curr_st -= bs
                break

            elif curr_dur + curr_st > bs:  # segment over multiple batches
                seg_mat.append(z[curr_st:bs,:])
                curr_dur -= bs - curr_st
                curr_st = 0
                break

            else:
                # completed this segment in this batch
                if curr_dur > 0:
                    seg_mat.append(z[curr_st:curr_st+curr_dur,:])

                if all(len(el)==0 for el in seg_mat):  # VERY small segment... --> no feature = Discard
                    print("Very short segment is discarded from feats: segment %s of file %s has duration %.3f" % (seq_name, curr_segid, float(segm_dict[seq_name]['dur'][curr_segidx])))
                    #seg_mat = np.array(seg_mat)  # or save as empty array (unindent kaldiio.save)
                else:
                    seg_mat = np.concatenate(seg_mat, axis=0)
                    #print(seg_mat.shape)
                    kaldiio.save_ark('%s/output_features_%s_%s/%s_feats.ark' % (expdir, lvar, suff, lvar), {curr_segid: seg_mat}, scp='%s/output_features_%s_%s/%s_feats.scp' % (expdir, lvar, suff, lvar), append=True)
                    #kaldiio.save_ark('%s/output_features_z1_%s/z1_feats_text.ark' % (expdir, suff), {curr_segid: seg_mat}, scp='%s/output_features_z1_%s/z1_feats_text.scp' % (expdir, suff), append=True, text=True)
                    # print("     Processed %i / %i segments for file %s" % (curr_segidx+1, len(segm_dict[seq_name]['start']), seq_name))
                seg_mat = []
                curr_segidx += 1

                # next file
                if curr_segidx == len(segm_dict[seq_name]['start']):
                    curr_seq += 1
                    print("Saved features of %i / %i files  -  file %s with %i segments" % (curr_seq, len(seqs), seq_name, curr_segidx))
                    curr_segidx = 0
                    if curr_seq == len(seqs):
                        print("DONE")
                        exit(0)

                    if nbest:
                        seq_name = str(seqs[curr_seq])
                    else:
                        seq_name = (str(seqs[curr_seq])).split('_')[0]

                if curr_segidx == 0:  # first segment of NEXT sequence --> wait till start
                    curr_st =  segm_dict[seq_name]['start'][curr_segidx]
                    new_seq_flag = True
                else:  #subtract start+dur=end of previous
                    curr_st = segm_dict[seq_name]['start'][curr_segidx] \
                             - segm_dict[seq_name]['start'][curr_segidx-1] \
                             - segm_dict[seq_name]['dur'][curr_segidx-1] \
                             + curr_st + curr_dur  # init row from previous segm

                curr_dur = segm_dict[seq_name]['dur'][curr_segidx]
                curr_segid = segm_dict[seq_name]['segid'][curr_segidx]


def process_segmentation(segments_file, seg_len, seg_shift, hop_t, factor):
    ''' Read Kaldi-style segments file and save start/end times etc.
        Save start and duration in number of z1-segments (rows in a batch)
         instead of real time. '''

    segm_dict = {}
    seg_t = seg_shift * hop_t  # in seconds
    fc = seg_len * hop_t / 2  # center of first segment in seconds

    with open(segments_file, 'r') as pd:
        line = pd.readline()
        while line:
            if factor is None:
                segid, fname, st, et = line.rstrip().split(' ')
                st = int(math.ceil((float(st) - fc) / seg_t))
                et = int(math.ceil((float(et) - fc) / seg_t))
                dur = int(et - st)

                if fname not in segm_dict:
                    segm_dict[fname] = {'start': [st], 'dur': [dur], 'segid': [segid]}
                else:
                    ins_pnt = bisect.bisect_left(segm_dict[fname]['start'], st)
                    segm_dict[fname]['start'].insert(ins_pnt, st)
                    segm_dict[fname]['dur'].insert(ins_pnt, dur)
                    segm_dict[fname]['segid'].insert(ins_pnt, segid)

            elif float(factor) == 0.9 or float(factor) == 1.1:
                segid, fname, st, et = line.rstrip().split(' ')
                st = float(st) / factor
                et = float(et) / factor
                if et <= st + 0.01:  # follow KALDI rules
                    line = pd.readline()
                    continue
                st = int(math.ceil((st - fc) / seg_t))
                et = int(math.ceil((et - fc) / seg_t))
                dur = int(et - st)

                fname = "sp%s-%s" % (str(factor), fname)
                segid = "sp%s-%s" % (str(factor), segid)
                if fname not in segm_dict:
                    segm_dict[fname] = {'start': [st], 'dur': [dur], 'segid': [segid]}
                else:
                    ins_pnt = bisect.bisect_left(segm_dict[fname]['start'], st)
                    segm_dict[fname]['start'].insert(ins_pnt, st)
                    segm_dict[fname]['dur'].insert(ins_pnt, dur)
                    segm_dict[fname]['segid'].insert(ins_pnt, segid)

            else:
                msg = "Speed perturbation factor should be 0.9 or 1.1"
                raise ValueError(msg)

            line = pd.readline()

    return segm_dict

    # Train_Dataset = create_dataset(seqs)
    # curr_seq = -1  # start
    # seg_vars = []
    # curr_segid = ''
    #
    # for yval, xval, _, _, _, start in Train_Dataset:
    #
    #     z1_mu, _, z1_sample, z1_sample_0, _, _, _, _, _, _ = model.encoder(xval)
    #
    #     seg = 0
    #     while seg < yval.get_shape().as_list()[0]:
    #
    #         if yval[seg] != curr_seq:
    #             curr_seq = yval[seg]
    #             seq_name = (str(seqs[curr_seq])).split('_')[0]
    #             seq_starts = list(segm_dict[seq_name]['start'])
    #             seq_ends = list(segm_dict[seq_name]['end'])
    #             seq_segids = list(segm_dict[seq_name]['segid'])
    #             curr_segid = seq_segids[0]
    #
    #             print("Saved features of %i / %i files" % (curr_seq, len(seqs)))
    #
    #         seg_start = tf.cast(start[seg], tf.float32)
    #         seg_center = seg_start * seg_shift * hop_t + (seg_len * hop_t / 2)
    #
    #         seg_idx = bisect.bisect(seq_starts, seg_center)
    #         if seg_idx == 0:  # before first start --> not in a segment (cut out)
    #             n_skip = math.ceil((seq_starts[0] - seg_center) / (seg_shift * hop_t))
    #             seg += int(n_skip)
    #             continue
    #         if seg_center > seq_ends[seg_idx-1]:  # after end of segment = between segments (cut out)
    #             if seg_idx == len(seq_starts):
    #                 nextseq_startseg = np.argmax(yval>curr_seq)  # where next sequence starts
    #                 if nextseq_startseg == 0:  # doesn't happen in this batch
    #                     break
    #                 else:
    #                     seg = nextseq_startseg
    #                     continue
    #             n_skip = math.ceil((seq_starts[seg_idx] - seg_center) / (seg_shift * hop_t))
    #             seg += int(n_skip)
    #             continue
    #
    #         if seq_segids[seg_idx-1] != curr_segid:  # completed segid
    #             seg_mat = np.stack(seg_vars)
    #             print(seg_mat.shape)
    #             kaldiio.save_ark('%s/output_features_z1/z1_feats.ark' % expdir, {curr_segid: seg_mat}, scp='%s/output_features_z1/z1_feats.scp' % expdir, append=True)
    #             print("     Processed %i / %i segments for file %s" % (seg_idx-1, len(seq_segids), seq_name))
    #             curr_segid = seq_segids[seg_idx-1]
    #             seg_vars = []
    #
    #         seg_vars.append(z1_mu[seg, :])
    #         seg += 1


# def process_segmentation(segments_file):
#     ''' Read Kaldi-style segments file and save start/end times etc.'''
#
#     segm_dict = {}
#
#     with open(segments_file, 'r') as pd:
#         line = pd.readline()
#         while line:
#             segid, fname, st, et = line.rstrip().split(' ')
#             st, et = float(st), float(et)
#
#             if fname not in segm_dict:
#                 segm_dict[fname] = {'start': [st], 'end': [et], 'segid': [segid]}
#             else:
#                 ins_pnt = bisect.bisect_left(segm_dict[fname]['start'], st)
#                 segm_dict[fname]['start'].insert(ins_pnt, st)
#                 segm_dict[fname]['end'].insert(ins_pnt, et)
#                 segm_dict[fname]['segid'].insert(ins_pnt, segid)
#
#             line = pd.readline()
#
#     return segm_dict

# read segmentation input # {segid : (filename, start, end)}
# d = kaldiio.load_scp(wavscp, segments=segments)
# segm = d._segments_dict
# with WriteHelper('ark,scp:file.ark,file.scp') as writer:
# with open_like_kaldi('| gzip %s/output_features_z1/z1_feats.ark.gz' % expdir, 'w') as f:
# kaldiio.save_ark(f, {seq_name: seq_mat}, append=True)
