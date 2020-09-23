from __future__ import division
import os
import shutil
import wave
import argparse
import subprocess
from sphfile import SPHFile

"""
add noise to CGN test data at several SNRs

This script expects you already have a dataset created with prepare_cgn_numpy.py.

STEPS:
1) extract Filtering And Noise Adding Tool files (and explanatory PDF) from:
   http://aurora.hsnr.de/download.html
2) compile: "make -f filter_add_noise.make"
3) run this script with appropriate parameters (pointing towards your installed program)
4) you might have to chmod some created files

NOTE:
first part (up until compute_feature) can be run on CPU, rest requires GPU (to load librosa)?

## COMMAND PARAMETERS:
/users/spraak/spchdata/aurora/CDdata_aurora2/noises /esat/spchdisk/scratch/jponcele/fhvae_jakob/datasets/cgn_np_fbank_afgklno_unsup /esat/spchdisk/scratch/jponcele/fhvae_jakob/datasets/cgn_np_fbank_afgklno_noisy
 /users/spraak/jponcele/NoiseFilter --ftype fbank
"""
SNR_values = ["0", "5", "10", "15", "20"]

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("noises_dir", type=str,
        help="directory containing raw noisefiles of aurora 2, e.g. aurora/CDdata_aurora2/noises")
parser.add_argument("cgn_exp_dir", type=str,
        help="CGN expdir to which to add noise")
parser.add_argument("out_dir", type=str,
        help="output data directory")
parser.add_argument("noisefilter", type=str,
        help="dir with compiled noise filt_add program")
parser.add_argument("--ftype", type=str, default="fbank", choices=["fbank", "spec"],
        help="feature type")
args = parser.parse_args()
print(args)

# init dirs
os.makedirs(os.path.join(args.out_dir, "tmp_raw_files"), exist_ok=True)
os.makedirs(os.path.join(args.out_dir, "tmp_raw_mixed_files"), exist_ok=True)
os.makedirs(os.path.join(args.out_dir, "wav"), exist_ok=True)
os.makedirs(os.path.join(args.out_dir, "test"), exist_ok=True)

# find the .raw noises
noises = []
os.makedirs(os.path.join(args.out_dir, "noises"), exist_ok=True)
for noise in os.listdir(args.noises_dir):
    shutil.copy2(os.path.join(args.noises_dir, noise), os.path.join(args.out_dir, "noises"))
    noises.append(os.path.join(args.out_dir, "noises", noise))

# find .wav files of test set that we want to add noise to
wavs = []
wavdict = {}
with open(os.path.join(args.cgn_exp_dir, "test", "wav.scp"), "r") as fd:
    line = fd.readline()
    while line:
        uttname = line.rstrip().split(" ")[0]
        wav = line.rstrip().split(" ")[1]
        wavs.append(wav)
        wavdict[wav] = uttname
        line = fd.readline()

print('Creating .raw files from .wav files...')

# convert .wav to .raw
raws = []
noisy_raws = []
for wav in wavs:
    #fname = (wav.split("/")[-1]).split('.')[0]
    fname = wavdict[wav]
    out = os.path.join(args.out_dir, "tmp_raw_files", fname+'.raw')
    raws.append(out)
    noisy_raws.append(os.path.join(args.out_dir, "tmp_raw_mixed_files", fname+'.raw'))
    cmdstring = "sox %s --bits 16 --encoding signed-integer --endian little %s" % (wav, out)
    subprocess.call(cmdstring, shell=True)

# prepare lists for c-program
in_list = os.path.join(args.out_dir, "in.list")
out_list = os.path.join(args.out_dir, "out.list")
with open(in_list, "w") as iid:
    with open(out_list, "w") as oid:
        for raw in raws:
            iid.write(raw+'\n')
        for noisy_raw in noisy_raws:
            oid.write(noisy_raw+'\n')

print('Adding noise...')

# run c-program
prog = os.path.join(args.noisefilter, "filter_add_noise")
log = os.path.join(args.out_dir, "filter.log")
scp_dict = {}
for noise in noises:
    for SNR in SNR_values:
        os.makedirs(os.path.join(args.out_dir, "tmp_raw_mixed_files"), exist_ok=True)
        # p341 is filter for 16kHz data, r is seed for randgen
        # -m is how SNR is calculated: after a_weight or on 0-8kHz signal with -m snr_8khz
        command = "%s -i %s -o %s -n %s -u -f p341 -s %s -r 1000 -m a_weight -e %s" \
                  % (prog, in_list, out_list, noise, SNR, log)
        subprocess.call(command, shell=True)

        # convert mixed .raw to .wav files
        noisespec = (noise.split('/')[-1]).split('.')[0]+"_"+SNR+"_dB"
        for noisy_raw in noisy_raws:
            fname = (noisy_raw.split("/")[-1]).split('.')[0]
            nname = fname + "_" + noisespec
            out = os.path.join(args.out_dir, "wav", nname + ".wav")
            cmdstring = \
                "sox -r 16k --bits 16 --encoding signed-integer -c 1 --endian little %s %s" \
                 % (noisy_raw, out)
            subprocess.call(cmdstring, shell=True)
            scp_dict[nname] = out

        # clean up
        shutil.rmtree(os.path.join(args.out_dir, "tmp_raw_mixed_files"))

# final clean up and create train/test/dev
shutil.rmtree(os.path.join(args.out_dir, "tmp_raw_files"))
os.symlink(os.path.join(args.cgn_exp_dir, "train"), os.path.join(args.out_dir, "train"))
os.symlink(os.path.join(args.cgn_exp_dir, "dev"), os.path.join(args.out_dir, "dev"))

clean_wav_dict = {}
with open(os.path.join(args.cgn_exp_dir, "test", "wav.scp"), 'r') as rd:
    line = rd.readline()
    while line:
        fname = line.split(" ")[0]
        loc = line.rstrip().split(" ")[1]
        clean_wav_dict[fname] = loc
        line = rd.readline()

with open(os.path.join(args.out_dir, "test", "wav.scp"), "w+") as td:
    for fname, loc in clean_wav_dict.items():
        td.write(fname + " " + loc + "\n")
    for fname, loc in scp_dict.items():
        td.write(fname + " " + loc + "\n")

# compute feature
feat_dir = os.path.abspath("%s/%s" % (args.out_dir, args.ftype))
os.makedirs(feat_dir, exist_ok=True)

def compute_feature(name):
    cmd = ["python", "./scripts/preprocess/prepare_numpy_data.py", "--ftype=%s" % args.ftype]
    cmd += ["%s/%s/wav.scp" % (args.out_dir, name), feat_dir]
    cmd += ["%s/%s/feats.scp" % (args.out_dir, name)]
    cmd += ["%s/%s/len.scp" % (args.out_dir, name)]

    p = subprocess.Popen(cmd)
    if p.wait() != 0:
        raise RuntimeError("Non-zero (%d) return code for `%s`" % (p.returncode, " ".join(cmd)))

compute_feature("test")

print("computed features")

# append noisy files to the fac files (by copying labels of clean ones)
os.makedirs(os.path.join(args.out_dir, "fac"), exist_ok=True)
for file in os.listdir(os.path.join(args.cgn_exp_dir, "fac")):
    if file.startswith("all_facs_"):
        with open(os.path.join(args.cgn_exp_dir, "fac", file), 'r') as fp:
            with open(os.path.join(args.out_dir, "fac", file), 'w+') as tp:
                line = fp.readline()
                # seq labs
                if len(line.split(' ')) == 2:
                    while line:
                        cleanseq = line.split(' ')[0]
                        lab = line.rstrip().split(' ')[1]
                        # see if in test set
                        if cleanseq in clean_wav_dict:
                            for noise in noises:
                                for snr in SNR_values:
                                    noisespec = (noise.split('/')[-1]).split('.')[0] + "_" + snr + "_dB"
                                    nname = cleanseq+'_'+noisespec
                                    tp.write(nname+ ' ' +lab + '\n')
                        tp.write(line)
                        line = fp.readline()
                # talabs
                elif len(line.split(' ')) == 1:
                    talab_lines = []
                    cleanseq = line.rstrip()
                    line = fp.readline()
                    while line:
                        # name
                        if len(line.split(' ')) == 1:
                            # see if in test set
                            if cleanseq in clean_wav_dict:
                                for noise in noises:
                                    for snr in SNR_values:
                                        noisespec = (noise.split('/')[-1]).split('.')[0] + "_" + snr + "_dB"
                                        nname = cleanseq + '_' + noisespec
                                        tp.write(nname + '\n')
                                        for talab in talab_lines:
                                            tp.write(talab)

                            tp.write(cleanseq+'\n')
                            for talab in talab_lines:
                                tp.write(talab)

                            cleanseq = line.rstrip()
                            talab_lines = []
                        else:
                            talab_lines.append(line)
                        line = fp.readline()


                    # last seq of file needs to be handled separately
                    tp.write(cleanseq+'\n')
                    for talab in talab_lines:
                        tp.write(talab)

                    if cleanseq in clean_wav_dict:
                        for noise in noises:
                            for snr in SNR_values:
                                noisespec = (noise.split('/')[-1]).split('.')[0] + "_" + snr + "_dB"
                                nname = cleanseq + '_' + noisespec
                                tp.write(nname + '\n')
                                for talab in talab_lines:
                                    tp.write(talab)

print("written factor files")

fix_scplists_cgn(args.out_dir)

print("fixed scp-lists of test set")

print("Done.")


def fix_scplists_cgn(expdir):

    # fix the error in noisy cgn test dataset where component was missing from feats/len but present in wav.Scp

    #expdir = '/esat/spchdisk/scratch/jponcele/fhvae_jakob/datasets/cgn_np_fbank_afgklno_noisy'

    compdict = {}
    lendict = {}
    featdict = {}

    with open(os.path.join(expdir, 'test', 'wav.scp'), 'r') as pd:
        line = pd.readline()
        while line:
            fullutt = line.rstrip().split(' ')[0]
            prts = fullutt.split('_')
            comp = prts.pop(2)
            utt_name = '_'.join(prts)
            compdict[utt_name] = comp
            line = pd.readline()

    with open(os.path.join(expdir, 'test', 'len.scp'), 'r') as fd:
        line = fd.readline()
        while line:
            utt = line.rstrip().split(' ')[0]
            leng = line.rstrip().split(' ')[1]
            if len(utt.split('_')) == 3:
                utt = '_'.join(utt.split('_')[0:2])
            lendict[utt] = leng
            line = fd.readline()

    with open(os.path.join(expdir, 'test', 'feats.scp'), 'r') as rd:
        line = rd.readline()
        while line:
            utt = line.rstrip().split(' ')[0]
            if len(utt.split('_')) == 3:
                utt = '_'.join(utt.split('_')[0:2])
            comp = compdict[utt]

            feat = line.rstrip().split(' ')[1]
            pth = feat.split('/')[-1]
            prts = pth.split('_')
            if len(prts) > 3:
                prts.insert(2, comp)
                newfeat = feat.split('/')[:-1]
                newfeat.append('_'.join(prts))
                feat = '/'.join(newfeat)

            featdict[utt] = feat
            line = rd.readline()

    with open(os.path.join(expdir, 'test', 'len.scp'), 'w+') as td:
        for utt in lendict:
            leng = lendict[utt]
            prts = utt.split('_')
            comp = compdict[utt]
            if len(prts) == 2:
                prts.append(comp)
            else:
                prts.insert(2, comp)
            fullutt = '_'.join(prts)
            td.write(fullutt+' '+leng+'\n')

    with open(os.path.join(expdir, 'test', 'feats.scp'), 'w+') as qd:
        for utt in featdict:
            feat = featdict[utt]
            prts = utt.split('_')
            comp = compdict[utt]
            if len(prts) == 2:
                prts.append(comp)
            else:
                prts.insert(2, comp)
            fullutt = '_'.join(prts)
            qd.write(fullutt+' '+feat+'\n')

    #final fix because fbank features .npy are named without comp in it, except the clean ones
    with open(os.path.join(expdir, 'test', 'tmp_feats.scp'), 'w+') as td:
        with open(os.path.join(expdir, 'test', 'feats.scp'), 'r') as pd:
            line = pd.readline()
            while line:
                utt = line.split(' ')[0].rstrip()
                if 'dB' not in utt:
                    td.write(line)
                else:
                    loc = line.split(' ')[1]
                    f = loc.split('/')[-1]
                    g = f.split('_')
                    g.pop(2)
                    new_f = '_'.join(g)
                    new_loc = '/'.join(loc.split('/')[:-1]) + '/'+new_f
                    td.write(utt+' '+new_loc)

                line = pd.readline()
    os.remove(os.path.join(expdir, 'test', 'feats.scp'))
    os.rename(os.path.join(expdir, 'test', 'tmp_feats.scp'), os.path.join(expdir, 'test', 'feats.scp'))
