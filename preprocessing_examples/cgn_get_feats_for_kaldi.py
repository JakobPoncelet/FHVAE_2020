from __future__ import absolute_import
import os
import sys
print(sys.path)
print(os.getcwd())
sys.path.insert(0, os.getcwd())
import wave
import argparse
import subprocess
from sphfile import SPHFile
from fhvae.datasets.audio_utils import *

# os.environ["CUDA_VISIBLE_DEVICES"]="0"

'''
Compute mfcc features of full-length cgn wav-files
--> then you can extract z1 features for these files using a trained model
(trained on split files with phone annot) and use the z1 features as input for kaldi
'''

###############
#cgndir = '/users/spraak/spchdata/cgn/wav'
#cgndir = '/esat/spchdisk/scratch/bbagher/kaldi/egs/CGN/data-nl/local/data/wav-data/users/spraak/spchdata/cgn/sam/'
#cgndir = '/esat/spchtemp/scratch/jponcele/nbest_augmented'
cgndir = '/esat/spchdisk/scratch/bbagher/kaldi/egs/CGN/data-vl/local/data/wav-data/users/spraak/spchdata/cgn/sam/'
#componentslist = 'abefghijklmno'
componentslist = 'cd'
#lang = 'nl'
lang = 'vl'
#outdir = '/esat/spchdisk/scratch/jponcele/fhvae_jakob/datasets/cgn_nl_kaldi_feats'
#outdir = '/esat/spchdisk/scratch/jponcele/fhvae_jakob/datasets/cgn_kaldi_feats'
#outdir = '/esat/spchdisk/scratch/jponcele/fhvae_jakob/datasets/nbest_kaldi_feats'
#outdir = '/esat/spchdisk/scratch/jponcele/fhvae_jakob/datasets/cgn_kaldi_feats_sp'
#outdir = '/esat/spchdisk/scratch/jponcele/fhvae_jakob/datasets/nbest_augmented_kaldi_feats'
outdir = '/esat/spchdisk/scratch/jponcele/fhvae_jakob/datasets/cgn_tel_kaldi_feats'

suffix = 'a0'
speed_perturb = False  # also generate feats at 0.9 and 1.1 speed (for kaldi DNN training)

#outdir = '/esat/spchdisk/scratch/jponcele/fhvae_jakob/datasets/nbest_cts_kaldi_feats'

###############

def maybe_makedir(d):
    try:
        os.makedirs(d)
    except OSError:
        pass

print("making wav.scp")

tr_scp = "%s/train/wav.scp" % (outdir)
dt_scp = "%s/dev/wav.scp" % (outdir)
tt_scp = "%s/test/wav.scp" % (outdir)

tr_sp0_scp = "%s/train/wav_sp0.scp" % (outdir)
tr_sp1_scp = "%s/train/wav_sp1_scp" % (outdir)

# maybe_makedir(wav_dir)
maybe_makedir(os.path.dirname(tr_scp))
maybe_makedir(os.path.dirname(dt_scp))
maybe_makedir(os.path.dirname(tt_scp))

tr_f = open(tr_scp, "w")
dt_f = open(dt_scp, "w")
tt_f = open(tt_scp, "w")

tr_sp0_f = open(tr_sp0_scp, "w")
tr_sp1_f = open(tr_sp1_scp, "w")


all_files = set()
for root, _, fnames in sorted(os.walk(cgndir)):

    # regio = root.split("/")[-1].lower()
    # if regio != lang:
    #     continue
    #
    # comp = root.split("/")[-2].lower()
    # if comp.split("-")[1] not in componentslist:
    #     continue

    for fname in fnames:
        if (fname.endswith(".wav") or fname.endswith(".WAV")) and not "converted" in fname:

            fileid = fname.split(".")[0].lower()

            # # if suffix, only keep _A0
            # if '_' in fileid:
            #     fileid, suff = fileid.split('_')
            #     if suff != suffix:
            #         continue
            #
            # if fileid in all_files:  # no duplicates
            #     continue
            #
            # all_files.add(fileid)
            #
            # seqid = "%s_%i_%s" % (fileid, 0, comp.split("-")[1])

            seqid = "%s" % fileid
            path = "%s/%s" % (root, fname)
            tr_f.write("%s %s\n" % (seqid, path))

            tr_sp0_f.write("sp0.9-%s %s\n" % (seqid, path))
            tr_sp1_f.write("sp1.1-%s %s\n" % (seqid, path))

            ## DOESNT WORK:
            #tr_sp0_f.write("sp0.9-%s sox -t wav %s -b 16 -t wav - remix - | sox -t wav - -t wav - speed 0.9 | \n" % (seqid, path))
            #tr_sp1_f.write("sp1.1-%s sox -t wav %s -b 16 -t wav - remix - | sox -t wav - -t wav - speed 1.1 | \n" % (seqid, path))

tr_f.close()
dt_f.close()
tt_f.close()
tr_sp0_f.close()
tr_sp1_f.close()

if speed_perturb:
    with open(tr_scp, 'a') as td:
        with open(tr_sp0_scp, 'r') as pd:
            td.write(pd.read())
        with open(tr_sp1_scp, 'r') as rd:
            td.write(rd.read())

os.remove(tr_sp0_scp)
os.remove(tr_sp1_scp)

print("computing features")

# compute feature
featdir = os.path.abspath("%s/%s" % (outdir, 'fbank'))
maybe_makedir(featdir)


def compute_feature(name):
    cmd = ["python", "./scripts/preprocess/prepare_numpy_data.py", "--ftype=fbank"]
    cmd += ["%s/%s/wav.scp" % (outdir, name), featdir]
    cmd += ["%s/%s/feats.scp" % (outdir, name)]
    cmd += ["%s/%s/len.scp" % (outdir, name)]
    cmd += ["--speed_perturb=%s" % str(speed_perturb)]

    p = subprocess.Popen(cmd)
    if p.wait() != 0:
        raise RuntimeError("Non-zero (%d) return code for `%s`" % (p.returncode, " ".join(cmd)))


compute_feature("train")

for dset in ["train", "dev", "test"]:
    for f in ["wav.scp", "feats.scp", "len.scp"]:
        fname = os.path.join(outdir, dset, f)
        with open(fname, 'a'):
            pass

print("DONE")
