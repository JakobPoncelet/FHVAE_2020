import numpy as np
from .seq_dataset import *
from collections import defaultdict


class Segment(object):
    def __init__(self, seq, start, end, lab, talab):
        self.seq = seq
        self.start = start
        self.end = end
        self.lab = lab
        self.talab = talab

    def __str__(self):
        return "(%s, %s, %s, %s, %s)" % (
            self.seq, self.start, self.end, self.lab, self.talab)

    def __repr__(self):
        return str(self)


class SegmentDataset(object):
    def __init__(self, seq_d, seg_len=20, seg_shift=8, rand_seg=False):
        """
        Args:
            seq_d(SequenceDataset): SequenceDataset or its child class
            seg_len(int): segment length
            seg_shift(int): segment shift if seg_rand is False; otherwise
                randomly extract floor(seq_len/seg_shift) segments per sequence
            rand_seg(bool): if randomly extract segments or not
        """
        self.seq_d = seq_d
        self.seg_len = seg_len
        self.seg_shift = seg_shift
        self.rand_seg = rand_seg

        self.seqlist = self.seq_d.seqlist
        self.feats = self.seq_d.feats
        self.lens = self.seq_d.lens
        self.labs_d = self.seq_d.labs_d
        self.talabseqs_d = self.seq_d.talabseqs_d
        self.talab_vals = self.seq_d.talab_vals
        self.talabseqs_d_new = self.seq_d.talabseqs_d_new

        self.lab_names = self.seq_d.lab_names
        self.talab_names = self.seq_d.talab_names
        self.seq2idx = self.seq_d.seq2idx
        self.seq2regidx = self.seq_d.seq2regidx

        self.clean_seqlist = self.seq_d.clean_seqlist
        self.noisy_seqlist = self.seq_d.noisy_seqlist

    def seq_iterator(self, bs, seqs=None, mapper=None):
        return self.seq_d.iterator(
            bs, seqs, mapper)

    def iterator(self, seg_bs, seq_bs=-1, seq_mapper=None, seqs=None):
        """
        Args:
            seg_bs(int): segment batch size
            seg_shift(int): use self.seg_shift if not set (None)
            rand_seg(bool): use self.rand_seg if not set (None)
            seg_shuffle(bool): shuffle segment list if True
            seg_rem(bool): yield remained segment batch if True
            seq_bs(int): -1 for loading all sequences. otherwise only
                blocked randomization for segments available
            lab_names(list): see SequenceDataset
            talab_names(list): see SequenceDataset
            seqs(list): see SequenceDataset
            seq_shuffle(bool): shuffle sequence list if True. this is
                unnecessary if seq_bs == -1 and seg_shuffle == True
            seq_rem(bool): yield remained sequence batch if True
            seq_mapper(callable): see SequenceDataset
        """
        seqs = self.seqlist if seqs is None else seqs
        seq_bs = len(seqs) if seq_bs == -1 else seg_bs

        seq_iterator = self.seq_iterator(seq_bs, seqs, seq_mapper)

        for seq, feats, lens in seq_iterator:
            if 'dB' in seq:  # load lab/talab of clean seq
                if seq.startswith('nbest'):
                    seq = '_'.join(seq.split('_')[0:-3]+['clean'])
                elif seq.startswith('v') or seq.startswith('n'):  # cgn
                    seq = '_'.join(seq.split('_')[0:-2])
                else:  # timit
                    seq = '_'.join(seq.split('_')[0:-3])

            seq_talabs = [self.talabseqs_d_new[name][seq] for name in self.talab_names]
            yield self.seq2idx[seq], feats, lens, self.seq2regidx[seq], seq_talabs

    def lab2nseg(self, lab_name, seg_shift=None):
        lab2nseg = defaultdict(int)
        seg_shift = self.seg_shift if seg_shift is None else seg_shift
        for seq in self.seqlist:
            nseg = (self.lens[seq] - self.seg_len) // seg_shift + 1
            lab = self.labs_d[lab_name][seq]
            lab2nseg[lab] += nseg
        return lab2nseg

    def get_shape(self):
        seq_shape = self.seq_d.get_shape()
        return (self.seg_len,) + seq_shape[1:]

class NumpySegmentDataset(SegmentDataset):
    def __init__(self, feat_scp, len_scp, lab_specs=[], talab_specs=[], min_len=1,
                 preload=False, mvn_path=None, seg_len=20, seg_shift=8, rand_seg=False, copy_from=None, train_talabs=None, num_noisy_versions=None):
        seq_d = NumpyDataset(feat_scp, len_scp, lab_specs, talab_specs,
                             min_len, preload, mvn_path, copy_from, train_talabs, num_noisy_versions)
        super(NumpySegmentDataset, self).__init__(
            seq_d, seg_len, seg_shift, rand_seg)
