import os
import numpy as np
import bisect
import pickle
import librosa
from collections import OrderedDict
from collections import Counter
from scipy.stats import entropy
from .audio_utils import *


def scp2dict(path, dtype=str, seqlist=None, lab=False):
    with open(path) as f:
        # s = [line.rstrip() for line in f]
        l = [line.rstrip().split(None, 1) for line in f]
    d = OrderedDict([(k, dtype(v)) for k, v in l])
    if seqlist is not None:
            d = subset_d(d, seqlist, lab)
    return d


# Regularized version
def load_lab(spec, seqlist=None, copy_from=None):
    if len(spec) == 3:
        name, nclass, path = spec
        seq2lab = scp2dict(path, int, seqlist, lab=True)
    else:
        name, path = spec
        seq2lab = scp2dict(path, str, seqlist, lab=True)
        # clean up unused or underrepresented labels
        if copy_from is None:
            u_lab = Counter(list(seq2lab.values()))
            # ignore classes ending in "X"
            for lab in list(u_lab.keys()):
                if len(lab) == 0 or lab[-1].upper() == "X":
                    del u_lab[lab]
            # ignore classes below frequency threshold
            ntot = sum(u_lab.values())
            pp = np.exp(entropy(list(u_lab.values())))  # perplexity
            if pp < 100.0:
                # delete classes which are underrepresented, except for e.g. speaker ID (pp > 100)
                for lab in list(u_lab.keys()):
                    if u_lab[lab] < ntot / (10.0 * pp):
                        del u_lab[lab]
            print(u_lab)
            u_lab = list(u_lab.keys())
        else:
            u_lab = copy_from.labs_d[name].lablist

        # remove eliminated labels from seq2lab
        for k in list(seq2lab.keys()):
            if seq2lab[k] not in u_lab:
                seq2lab[k] = ""

        nclass = len(u_lab)
        if "" not in u_lab:
            nclass += 1  # signal empty label should be added (unknown class)
    return name, nclass, seq2lab


def subset_d(d, l, lab=False):
    """
    retain keys in l. raise KeyError if some key is missed
    + add "" if no label present
    """
    new_d = OrderedDict()

    # if labels are not present (unsupervised), put the "" (unknown) label
    if lab:
        for k in l:
            if k not in d:
                new_d[k] = ""
            else:
                new_d[k] = d[k]
    else:
        # check for missing features
        for k in l:
            new_d[k] = d[k]
    return new_d


# regularized version
def load_talab(spec, lens, seqlist=None, copy_from=None, train_talabs=None):

    if len(spec) == 3:
        name, nclass, path = spec
        seq2lab = scp2dict(path, int, seqlist, lab=True)
    else:
        name, path = spec

    # keep dictionary with all labels and transform them to integers
    if train_talabs == None:
        talab_vals = OrderedDict()
        talab_vals[""] = 0  # add empty label
        talab_cnt = 1
    else:
        # in testing phase, keep the same numbering for the phones as during training
        talab_vals = train_talabs[name]

    if copy_from == None:
        with open(path) as f:
            toks_l = [line.rstrip().split() for line in f]
        assert (len(toks_l) > 0 and len(toks_l[0]) == 1)
        seq2talab_newform = OrderedDict()
        seq = toks_l[0][0]
        talabs_newform = [[], [], []]  #starts, stops, labs
        for toks in toks_l[1:]:
            if len(toks) == 1:  # new utt
                seq2talab_newform[seq] = talabs_newform
                seq = toks[0]
                talabs_newform = [[], [], []]
            elif len(toks) == 3:  # format start stop label
                if not isinstance(toks[2], int):
                    if toks[2] not in talab_vals:
                        talab_vals[toks[2]] = talab_cnt
                        talab_cnt += 1
                    lab = int(talab_vals[toks[2]])
                else:
                    lab = int(toks[2])

                talabs_newform[0].append(int(toks[0]))
                talabs_newform[1].append(int(toks[1]))
                talabs_newform[2].append(lab)

            else:
                raise ValueError("invalid line %s" % str(toks))
        seq2talab_newform[seq] = talabs_newform

    else:
        talab_vals = copy_from.talab_vals[name]
        seq2talab_newform = copy_from.talabseqs_d_new[name]


    for seq in seqlist:
        if seq not in seq2talab_newform:
            # no talabs for this seq -> put unknown "" label over entire length
            seq2talab_newform[seq] = [[0], [lens[seq]-1], [0]]

    nclass = len(talab_vals)

    return name, nclass, talab_vals, seq2talab_newform


class TimeAlignedLabel(object):
    """
    time-aligned label
    """

    def __init__(self, lab, start, stop):
        assert (start >= 0)
        self.lab = lab
        self.start = start
        self.stop = stop

    def __str__(self):
        return "(lab=%s, start=%s, stop=%s)" % (self.lab, self.start, self.stop)

    def __repr__(self):
        return str(self)

    @property
    def center(self):
        return (self.start + self.stop) / 2

    def __len__(self):
        return self.stop - self.start

    def centered_talab(self, slice_len):
        start = self.center - slice_len / 2
        stop = self.center + slice_len / 2
        return TimeAlignedLabel(self, self.lab, start, stop)


class TimeAlignedLabelSeq(object):
    """
    time-aligned labels for one sequence
    """

    def __init__(self, talabs, noov=True, nosp=False):
        """
        talabs(list): list of TimeAlignedLabel
        noov(bool): check no overlapping between TimeAlignedLabels
        nosp(bool): check no spacing between TimeAlignedLabels
        """
        talabs = sorted(talabs, key=lambda x: x.start)
        if noov and nosp:
            assert (talabs[0].start == 0)
            for i in range(len(talabs) - 1):
                if talabs[i].stop != talabs[i + 1].start:
                    raise ValueError(talabs[i], talabs[i + 1])
        elif noov:
            for i in range(len(talabs) - 1):
                if talabs[i].stop > talabs[i + 1].start:
                    raise ValueError(talabs[i], talabs[i + 1])
        elif nosp:
            assert (talabs[0].start == 0)
            for i in range(len(talabs) - 1):
                if talabs[i].stop < talabs[i + 1].start:
                    raise ValueError(talabs[i], talabs[i + 1])

        self.talabs = talabs
        self.noov = noov
        self.nosp = nosp
        self.max_stop = max([l.stop for l in talabs])

    def __str__(self):
        return "\n".join([str(l) for l in self.talabs])

    def __len__(self):
        return len(self.talabs)

    def to_seq(self):
        return [l.lab for l in self.talabs]

    def center_lab(self, start=-1, stop=-1, strict=False):
        """
        return the centered label in a sub-sequence
        Args:
            start(int)
            stop(int)
            strict(bool): raise error if center is not defined
        """
        if not self.noov:
            raise ValueError("center() only available in noov mode")

        start = 0 if start == -1 else start
        stop = self.max_stop if stop == -1 else stop
        center = (start + stop) / 2
        idx_l = bisect.bisect_right([l.start for l in self.talabs], center) - 1
        idx_r = bisect.bisect_right([l.stop for l in self.talabs], center)
        if not strict:
            return self.talabs[idx_l].lab
        elif idx_r != idx_l and strict:
            msg = "spacing detected at %s; " % center
            msg += "neigbors: %s, %s" % (self.talabs[idx_l], self.talabs[idx_r])
            raise ValueError(msg)

    @property
    def lablist(self):
        if not hasattr(self, "_lablist"):
            self._lablist = sorted(np.unique([l.lab for l in self.talabs]))
        return self._lablist


class TimeAlignedLabelSeqs(object):
    """
    time-aligned label sequences(TimeAlignedLabelSeq) for a set of sequences
    """

    def __init__(self, name, nclass, seq2talabseq):
        self.name = name
        self.nclass = nclass
        self.seq2talabseq = seq2talabseq

    def __getitem__(self, seq):
        return self.seq2talabseq[seq]

    def __str__(self):
        return "name=%s, nclass=%s, nseqs=%s" % (
            self.name, self.nclass, len(self.seq2talabseq))

    @property
    def lablist(self):
        if not hasattr(self, "_lablist"):
            labs = np.concatenate(
                [talabseq.lablist for talabseq in list(self.seq2talabseq.values())])
            self._lablist = sorted(np.unique(labs))
        return self._lablist


class Labels(object):
    """
    labels(int) for a set of sequences
    """

    def __init__(self, name, nclass, seq2lab):
        self.name = name
        self.nclass = nclass
        self.seq2lab = seq2lab

    def __getitem__(self, seq):
        return self.seq2lab[seq]

    @property
    def lablist(self):
        if not hasattr(self, "_lablist"):
            self._lablist = sorted(np.unique(list(self.seq2lab.values())))
            if len(self._lablist) < self.nclass:
                self._lablist.insert(0, "")  # assumes "" always ends up first if it is already in lablist
        return self._lablist


class SequenceDataset(object):
    def __init__(self, feat_scp, len_scp, lab_specs=[], talab_specs=[], min_len=1, copy_from=None, train_talabs=None):
        """
        Args:
            feat_scp(str): feature scp path
            len_scp(str): sequence-length scp path
            lab_specs(list): list of label specifications. each is
                (name, number of classes, scp path)
            talab_specs(list): list of time-aligned label specifications.
                each is (name, number of classes, ali path)
            min_len(int): keep sequence no shorter than min_len
            copy_from: a SequenceDataset from which to copy the labs_d dictionary.
                Used such that train and dev set use same labels.
        """
        feats = scp2dict(feat_scp)
        lens = scp2dict(len_scp, int, list(feats.keys()))

        self.seqlist = [k for k in list(feats.keys()) if lens[k] >= min_len]
        self.feats = OrderedDict([(k, feats[k]) for k in self.seqlist])
        self.lens = OrderedDict([(k, lens[k]) for k in self.seqlist])
        print(("%s: %s out of %s kept, min_len = %d" % (
            self.__class__.__name__, len(self.feats), len(feats), min_len)))

        self.labs_d = OrderedDict()
        for lab_spec in lab_specs:
            if lab_spec[0] == "spk":  # don't copy speaker info: dev speakers don't overlap with trn speakers in mu2-table
                name, nclass, seq2lab = load_lab(lab_spec, self.seqlist)
            else:
                name, nclass, seq2lab = load_lab(lab_spec, self.seqlist, copy_from)
            #name, nclass, seq2lab = load_lab(lab_spec, self.seqlist)
            self.labs_d[name] = Labels(name, nclass, seq2lab)
        self.talabseqs_d = OrderedDict()
        self.talabseqs_d_new = OrderedDict()
        self.talab_vals = OrderedDict()
        for talab_spec in talab_specs:
            name, nclass, talab_vals, seq2talab_newform = load_talab(talab_spec, self.lens, self.seqlist, copy_from, train_talabs)
            self.talab_vals[name] = talab_vals
            self.talabseqs_d_new[name] = seq2talab_newform.copy()

        self.seq2idx = dict([(seq, i) for i, seq in enumerate(self.seqlist)])
        self.lab_names = list(self.labs_d.keys())
        self.talab_names = list(self.talabseqs_d_new.keys())

        ## If you want no spk detection but filter on spk, not tested in a while
        # if "spk" in tr_dset.labs_d:
        #     s_seqs = tr_dset.seqlist
        #     spklist = tr_dset.labs_d["spk"].lablist
        #     seq2lab = tr_dset.labs_d["spk"].seq2lab
        #     seq2idx = dict([(seq, spklist.index(seq2lab[seq])) for seq in s_seqs])
        #     lab_names = tr_dset.labs_d.keys()
        #     lab_names.remove("spk")
        # if "spk" in dt_dset.labs_d:
        #     lab_names = dt_dset.labs_d.keys()
        #     lab_names.remove("spk")

        ii = list()
        for k in self.seqlist:
            itm = []
            for name in self.lab_names:
                if k not in self.labs_d[name].seq2lab:  # unsupervised data without labels
                    itm.append(0)
                else:
                    itm.append(self.labs_d[name].lablist.index(self.labs_d[name].seq2lab[k]))
            ii.append(np.asarray(itm))
        self.seq2regidx = dict(list(zip(self.seqlist, ii)))

    def iterator(self, bs, seqs=None, mapper=None):
        """
        Args:
            bs(int): batch size
            lab_names(list): list of names of labels to include
            talab_names(list): list of names of time-aligned labels to include
            seqs(list): list of sequences to iterate. iterate all if seqs is None
            shuffle(bool): shuffle sequence order if true
            rem(bool): yield remained batch if true
            mapper(callable): feat is mapped by mapper if not None
        Return:
            keys(list): list of sequences(str)
            feats(list): list of feats(str/mapper(str))
            lens(list): list of sequence lengths(int)
            labs(list): list of included labels(int list)
            talabs(list): list of included time-aligned labels(talabel list)
        """
        seqs = self.seqlist if seqs is None else seqs
        mapper = (lambda x: x) if mapper is None else mapper

        for idx, seq in enumerate(seqs):  # e.g. all of the training set
            yield seq, mapper(self.feats[seq]), self.lens[seq]

    def seqs_of_lab(self, lab_name, lab):
        return [seq for seq in self.seqlist if self.labs_d[lab_name][seq] == lab]

    def seqs_of_talab(self, talab_name, lab):
        return [seq for seq in self.seqlist \
                if lab in self.talabseqs_d[talab_name][seq].lablist]

    def get_shape(self, mapper=None):
        raise NotImplementedError

class NumpyDataset(SequenceDataset):
    def __init__(self, feat_scp, len_scp, lab_specs=[], talab_specs=[],
                 min_len=1, preload=False, mvn_path=None, copy_from=None, train_talabs=None):
        super(NumpyDataset, self).__init__(
            feat_scp, len_scp, lab_specs, talab_specs, min_len, copy_from, train_talabs)
        if preload:
            feats = OrderedDict()
            for seq in self.seqlist:
                with open(self.feats[seq], "rb") as f:
                    feats[seq] = np.load(f)
            self.feats = feats
            print("preloaded features")
        else:
            self.feats = self.feat_getter(self.feats)

        if mvn_path is not None:
            if not os.path.exists(mvn_path):
                self.mvn_params = self.compute_mvn()
                with open(mvn_path, "wb") as f:
                    pickle.dump(self.mvn_params, f)
            else:
                with open(mvn_path, "rb") as f:
                    self.mvn_params = pickle.load(f)
        else:
            self.mvn_params = None

    class feat_getter:
        def __init__(self, feats):
            self.feats = dict(feats)

        def __getitem__(self, seq):
            with open(self.feats[seq], "rb") as f:
                feat = np.load(f)
            return feat

    def compute_mvn(self):
        n, x, x2 = 0., 0., 0.
        for seq in self.seqlist:
            feat = self.feats[seq]  #calls feat_getter.__getitem__
            x += np.sum(feat, axis=0, keepdims=True)
            x2 += np.sum(feat ** 2, axis=0, keepdims=True)
            n += feat.shape[0]
        mean = x / n
        std = np.sqrt(x2 / n - mean ** 2)
        return {"mean": mean, "std": std}

    def apply_mvn(self, feats):
        if self.mvn_params is None:
            return feats
        else:
            return (feats - self.mvn_params["mean"]) / self.mvn_params["std"]

    def undo_mvn(self, feats):
        if self.mvn_params is None:
            return feats
        else:
            return feats * self.mvn_params["std"] + self.mvn_params["mean"]

    def iterator(self, bs, seqs=None, mapper=None):
        if mapper is None:
            new_mapper = self.apply_mvn
        else:
            new_mapper = lambda x: mapper(self.apply_mvn(x))
        return super(NumpyDataset, self).iterator(
            bs, seqs, new_mapper)

    def get_shape(self):
        seq_shape = self.feats[self.seqlist[0]].shape
        return (None,) + tuple(seq_shape[1:])
