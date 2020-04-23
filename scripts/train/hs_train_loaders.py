import sys
import numpy as np
from fhvae.datasets.seg_dataset import NumpySegmentDataset
import tensorflow as tf

def load_data_reg(name, seqlist_path=None, lab_names=None, talab_names=None):
    # lab_names e.g. region:gender then loaded from (seqlist_path % lab_name) as scp file
    root = "./datasets/%s" % name
    mvn_path = "%s/train/mvn.pkl" % root
    seg_len = 20  # 15
    seg_shift = 8  # 5

    Dataset = NumpySegmentDataset

    if lab_names is not None:
        lab_specs = [(lab, seqlist_path % lab) for lab in lab_names.split(':')]
    else:
        lab_specs = list()
    if talab_names is not None:
        talab_specs = [(talab, seqlist_path % talab) for talab in talab_names.split(':')]
    else:
        talab_specs = list()

    # initialize the datasets
    tr_dset = Dataset(
        "%s/train/feats.scp" % root, "%s/train/len.scp" % root,
        lab_specs=lab_specs, talab_specs=talab_specs,
        min_len=seg_len, preload=False, mvn_path=mvn_path,
        seg_len=seg_len, seg_shift=seg_shift, rand_seg=True)

    dt_dset = Dataset(
        "%s/dev/feats.scp" % root, "%s/dev/len.scp" % root,
        lab_specs=lab_specs, talab_specs=talab_specs,
        min_len=seg_len, preload=False, mvn_path=mvn_path,
        seg_len=seg_len, seg_shift=seg_len, rand_seg=False,
        copy_from=tr_dset)

    return _load_reg(tr_dset, dt_dset) + (tr_dset,)


def _load_reg(tr_dset, dt_dset):

    def tr_iterator_by_seqs(s_seqs=tr_dset.seqlist, bs=256, seg_rem=False):
        # build an iterator over the dataset, into batches
        _iterator = tr_dset.iterator(bs, seqs=s_seqs)

        for key, feats, lens, lab, talabs in _iterator:
            yield np.asarray(key), np.asarray(feats), np.asarray(lens), np.asarray(lab), np.asarray(talabs)


    def dt_iterator(bs=256):
        # development set iterator for validation step
        _iterator = dt_dset.iterator(bs, seqs=dt_dset.seqlist)

        for key, feats, lens, lab, talabs in _iterator:
            yield np.asarray(key), np.asarray(feats), np.asarray(lens), np.asarray(lab), np.asarray(talabs)


    tr_nseqs = len(tr_dset.seqlist)
    tr_shape = tr_dset.get_shape()

    return tr_nseqs, tr_shape, tr_iterator_by_seqs, dt_iterator, dt_dset
