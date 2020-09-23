"""
prepare CGN data for FHVAE

## COMMAND PARAMETERS:
/esat/spchtemp/scratch/jponcele/cgn_augmented_sequences --clean_cgn_dir /esat/spchdisk/scratch/jponcele/fhvae_jakob/datasets/cgn_np_fbank_afgklno_unsup --out_dir /esat/spchdisk/scratch/jponcele/fhvae_jakob/datasets/cgn_augmented --ftype fbank

python ./preprocessing_examples/prepare_augmented_cgn_numpy.py /esat/spchtemp/scratch/jponcele/cgn_allcomps_sequences_augmented --clean_cgn_dir /esat/spchtemp/scratch/jponcele/cgn_allcomps_sequences --out_dir /esat/spchtemp/scratch/jponcele/cgn_allcomps_sequences_augmented --ftype fbank
"""
import os
import wave
import librosa
import argparse
import subprocess
import shutil
from collections import defaultdict
from sphfile import SPHFile

# os.environ["CUDA_VISIBLE_DEVICES"]="0"

"""
5X augmented dataset created with augmentcgn.m based on cgn_np_fbank_afgklno, 
    for every file 4 copies with:
    - added noise 0-15 SNR
    - added noise 5-20 SNR
    - filtered 
    - filter + noise 5-20 SNR
    
factors copied from cgn_np_fbank_afgklno_unsup
"""

#componentslist = ["comp-a", "comp-f", "comp-g", "comp-k", "comp-l", "comp-n", "comp-o"]
#componentslist = ["comp-a", "comp-b", "comp-e", "comp-f", "comp-g", "comp-h", "comp-i", "comp-j", "comp-k", "comp-l", "comp-m", "comp-n", "comp-o"]
componentslist = ["comp-c", "comp-d"]
aug_types = ["noisy1", "noisy2", "filtered", "noisyandfiltered"]

def maybe_makedir(d):
    try:
        os.makedirs(d)
    except OSError:
        pass

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("aug_cgn_dir", type=str,
                    help="data directory containing the split wav files of CGN")
parser.add_argument("--clean_cgn_dir", type=str,
                    help="data directory containing clean already prepared CGN fhvae dataset")
parser.add_argument("--ftype", type=str, default="fbank", choices=["fbank", "spec"],
                    help="feature type")
parser.add_argument("--out_dir", type=str, default="./datasets/cgn_per_speaker",
                    help="output data directory")
parser.add_argument("--dev_spk", type=str, default="./misc/cgn/cgn_per_spk_afgklno_dev.list",
                    help="path to list of dev set speakers")
parser.add_argument("--test_spk", type=str, default="./misc/cgn/cgn_per_spk_afgklno_test.list",
                    help="path to list of test set speakers")
args = parser.parse_args()
print(args)

# retrieve partition
with open(args.dev_spk) as f:
    dt_spks = [line.rstrip().lower() for line in f]
with open(args.test_spk) as f:
    tt_spks = [line.rstrip().lower() for line in f]

dt_spks = []
tt_spks = []

print("making wav.scp")

clean_seqs = defaultdict(set)
aug_seqs = defaultdict(set)
comp_dict = dict()

for set in ['train']:  #, 'dev', 'test']:
    maybe_makedir(os.path.join(args.out_dir, set))
    out_scp = os.path.join(args.out_dir, set, "wav.scp")
    out_f = open(out_scp, "w+")

    # copy clean utts, format of line should be: "spk_nr_comp <location>"
    with open(os.path.join(args.clean_cgn_dir, set, "wav.scp")) as rd:
        line = rd.readline()
        while line:
            uttid = line.rstrip().split(' ')[0]
            clean_seqs[set].add(uttid)
            out_f.write(line)
            line = rd.readline()

    for root, _, fnames in sorted(os.walk(args.aug_cgn_dir)):
        regio = root.split("/")[-1].lower()  # nl/vl
        comp = root.split("/")[-2].lower()
        if comp not in componentslist:
            continue
        for fname in fnames:
            if fname.endswith(".wav") or fname.endswith(".WAV"):
                comp_dict[fname] = comp.split("-")[1]
                spk_and_nr = '_'.join(fname.split(".")[0].lower().split("_")[0:2])
                clean_uttid = "%s_%s" % (spk_and_nr, comp.split("-")[1])
                aug = fname.split(".")[0].lower().split("_")[2]

                if clean_uttid in clean_seqs[set]:
                    # path = "%s/%s/%s/%s" % (args.cgn_dir, comp, regio, fname)
                    path = "%s/%s" % (root, fname)
                    uttid = "%s_%s_%s_dB" % (spk_and_nr, comp.split("-")[1], aug)
                    aug_seqs[set].add(uttid)
                    out_f.write("%s %s\n" % (uttid, path))

    out_f.close()

    assert (len(aug_seqs[set]) % len(clean_seqs[set])) == 0, "# Augmented seqs is not a multiple of # clean seqs"

print("converted to wav and dumped scp files")


# compute feature
feat_dir = os.path.abspath("%s/%s" % (args.out_dir, args.ftype))
maybe_makedir(feat_dir)


def compute_feature(name):
    cmd = ["python", "./scripts/preprocess/prepare_numpy_data.py", "--ftype=%s" % args.ftype]
    cmd += ["%s/%s/wav.scp" % (args.out_dir, name), feat_dir]
    cmd += ["%s/%s/feats.scp" % (args.out_dir, name)]
    cmd += ["%s/%s/len.scp" % (args.out_dir, name)]

    p = subprocess.Popen(cmd)
    if p.wait() != 0:
        raise RuntimeError("Non-zero (%d) return code for `%s`" % (p.returncode, " ".join(cmd)))


for name in ["train"]:  #, "dev", "test"]:
    compute_feature(name)

print("computed features")

# copy factor files of clean set
maybe_makedir(os.path.join(args.out_dir, "fac"))
for file in os.listdir(os.path.join(args.clean_cgn_dir, "fac")):
    if file.startswith("all_facs_"):
        shutil.copy2(os.path.join(args.clean_cgn_dir, "fac", file),
                     os.path.join(args.out_dir, "fac", file))

print("copied clean factor files")

# # copy factor files of clean set and append augmented sequences by copying labels --> TOO BIG FILES!!!
# maybe_makedir(os.path.join(args.out_dir, "fac"))
#
# for file in os.listdir(os.path.join(args.clean_cgn_dir, "fac")):
#     if file.startswith("all_facs_"):
#         with open(os.path.join(args.clean_cgn_dir, "fac", file), 'r') as fp:
#             with open(os.path.join(args.out_dir, "fac", file), 'w+') as tp:
#                 line = fp.readline()
#                 # seq labs
#                 if len(line.split(' ')) == 2:
#                     while line:
#                         clean_name, lab = line.rstrip().split(' ')
#                         tp.write(line)  # copy clean seq
#                         for augm in aug_types:  # append augmented seqs
#                             aug_name = "%s_%s_dB" % (clean_name, augm)
#                             tp.write(aug_name + ' ' + lab + '\n')
#                         line = fp.readline()
#                 # talabs
#                 elif len(line.split(' ')) == 1:
#                     talab_lines = []
#                     clean_name = line.rstrip()
#                     line = fp.readline()
#                     while line:
#                         # name
#                         if len(line.split(' ')) == 1:
#                             tp.write(clean_name+'\n')
#                             for talab in talab_lines:
#                                 tp.write(talab)
#
#                             for augm in aug_types:
#                                 aug_name = "%s_%s_dB" % (clean_name, augm)
#                                 tp.write(aug_name+'\n')
#                                 for talab in talab_lines:
#                                     tp.write(talab)
#
#                             clean_name = line.rstrip()
#                             talab_lines = []
#                         else:
#                             talab_lines.append(line)
#                         line = fp.readline()
#
#                     # last seq of file needs to be handled separately
#                     tp.write(clean_name + '\n')
#                     for talab in talab_lines:
#                         tp.write(talab)
#
#                     for augm in aug_types:
#                         aug_name = "%s_%s_dB" % (clean_name, augm)
#                         tp.write(aug_name + '\n')
#                         for talab in talab_lines:
#                             tp.write(talab)
#
# print("generated factor files")

# get noise/RIR specs of augmented files
out_f = open(os.path.join(args.out_dir, "fac", "augment_specs.txt"), "w")

for augm in aug_types:
    sum_f = "%s/wav/summary_%s.txt" % (args.aug_cgn_dir, augm)
    with open(os.path.join(sum_f), 'r') as pd:
        line = pd.readline()
        if augm == "noisy1" or augm == "noisy2":
            case = 0
            assert len(line.split(' ')) == 4, "format = <wav> <SNR> <noisetype> <noisefile>"
        elif augm == "filtered":
            case = 1
            assert len(line.split(' ')) == 3, "format = <wav> <roomtype> <RIRlength>"
        elif augm == "noisyandfiltered":
            case = 2
            assert len(line.split(' ')) == 7, "format = <wav> <SNR> <noisetype> <noisefile> - <roomtype> <RIRlength>"

        while line:
            wav = line.rstrip().split(' ')[0]
            spk_and_nr = '_'.join(wav.split('.')[0].lower().split('_')[0:2])
            aug = wav.split('.')[0].lower().split('_')[2]
            comp = comp_dict[wav]
            uttid = "%s_%s_%s_dB" % (spk_and_nr, comp, aug)

            if case == 0:
                _, SNR, ntype, _ = line.rstrip().split(' ')
                write_spec = "n %s_%s" % (ntype, SNR)

            elif case == 1:
                _, room, rlength = line.rstrip().split(' ')
                write_spec = "f %s_%s" % (room, rlength)

            elif case == 2:
                _, SNR, ntype, _, _, room, rlength = line.rstrip().split(' ')
                write_spec = "n_f %s_%s_%s_%s" % (room, rlength, ntype, SNR)

            out_f.write("%s %s\n" % (uttid, write_spec))

            line = pd.readline()

out_f.close()

print("saved augmentation specs")
