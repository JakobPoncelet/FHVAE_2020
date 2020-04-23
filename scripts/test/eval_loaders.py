import sys
import numpy as np
from fhvae.datasets.seg_dataset import NumpySegmentDataset

np.random.seed(123)


def load_data_reg(name, set_name, seqlist_path=None, lab_names=None, talab_names=None, train_talab_vals=None):
    root = "./datasets/%s" % name
    mvn_path = "%s/train/mvn.pkl" % root
    seg_len = 20  # 15
    Dataset = NumpySegmentDataset

    if lab_names is not None:
        lab_specs = [(lab, seqlist_path % lab) for lab in lab_names.split(':')]
    else:
        lab_specs = list()
    if talab_names is not None:
        talab_specs = [(talab, seqlist_path % talab) for talab in talab_names.split(':')]
    else:
        talab_specs = list()

    tt_dset = Dataset(
        "%s/%s/feats.scp" % (root, set_name), "%s/%s/len.scp" % (root, set_name),
        lab_specs=lab_specs, talab_specs=talab_specs,
        min_len=seg_len, preload=False, mvn_path=mvn_path,
        seg_len=seg_len, seg_shift=seg_len, rand_seg=False, copy_from=None, train_talabs=train_talab_vals)

    return _load_reg(tt_dset) + (tt_dset,)

def _load_reg(tt_dset):
    # def _make_batch(seqs, feats, nsegs, seq2idx, seq2reg, talabs):
    #     # what the iterator returns --> xval, yval, nval, cval
    #     x = feats
    #     y = np.asarray([seq2idx[seq] for seq in seqs])
    #     n = np.asarray(nsegs)
    #     c = np.asarray([seq2reg[seq] for seq in seqs])
    #     b = np.asarray(talabs)
    #     return x, y, n, c, b

    def tt_iterator_by_seqs(s_seqs, bs=256):
        # seq2idx = dict([(seq, i) for i, seq in enumerate(s_seqs)])
        # lab_names = list(tt_dset.labs_d.keys())
        # talab_names = list(tt_dset.talabseqs_d.keys())
        # ii = list()
        # for k in s_seqs:
        #     itm = []
        #     for name in lab_names:
        #         if k not in tt_dset.labs_d[name].seq2lab:  # unsupervised data without labels
        #             itm.append("")
        #         else:
        #             itm.append(tt_dset.labs_d[name].lablist.index(tt_dset.labs_d[name].seq2lab[k]))
        #     #itm = [tt_dset.labs_d[name].lablist.index(tt_dset.labs_d[name].seq2lab[k]) for name in lab_names]
        #     ii.append(np.asarray(itm))
        # seq2regidx = dict(list(zip(s_seqs, ii)))
        _iterator = tt_dset.iterator(bs, seqs=s_seqs)

        # for seqs, feats, nsegs, labs, talabs in _iterator:
        #     yield _make_batch(seqs, feats, nsegs, seq2idx, seq2regidx, talabs)

        for key, feats, lens, lab, talabs in _iterator:
            yield np.asarray(key), np.asarray(feats), np.asarray(lens), np.asarray(lab), np.asarray(talabs)

    def tt_iterator(bs=256):
        # seq2idx = dict([(seq, i) for i, seq in enumerate(tt_dset.seqlist)])
        # lab_names = list(tt_dset.labs_d.keys())
        # talab_names = list(tt_dset.talabseqs_d.keys())
        # ii = list()
        # for k in tt_dset.seqlist:
        #     itm = []
        #     for name in lab_names:
        #         if k not in tt_dset.labs_d[name].seq2lab:  # unsupervised data without labels
        #             itm.append("")
        #         else:
        #             itm.append(tt_dset.labs_d[name].lablist.index(tt_dset.labs_d[name].seq2lab[k]))
        #     #itm = [tt_dset.labs_d[name].lablist.index(tt_dset.labs_d[name].seq2lab[k]) for name in lab_names]
        #     ii.append(np.asarray(itm))
        # seq2regidx = dict(list(zip(tt_dset.seqlist, ii)))
        _iterator = tt_dset.iterator(bs, seqs=tt_dset.seqlist)
        # for seqs, feats, nsegs, labs, talabs in _iterator:
        #     yield _make_batch(seqs, feats, nsegs, seq2idx, seq2regidx, talabs)

        for key, feats, lens, lab, talabs in _iterator:
            yield np.asarray(key), np.asarray(feats), np.asarray(lens), np.asarray(lab), np.asarray(talabs)

    return tt_iterator, tt_iterator_by_seqs, tt_dset.seqlist
