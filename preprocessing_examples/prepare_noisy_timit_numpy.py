from __future__ import division
import os
import shutil
import wave
import argparse
import subprocess
from sphfile import SPHFile

"""
add noise to TIMIT test data at several SNRs

This script expects you already have a dataset created with prepare_timit_numpy.py.

STEPS:
1) extract Filtering And Noise Adding Tool files (and explanatory PDF) from:
   http://aurora.hsnr.de/download.html
2) compile: "make -f filter_add_noise.make"
3) run this script with appropriate parameters (pointing towards your installed program)
4) you might have to chmod some created files

## COMMAND PARAMETERS:
/users/spraak/spchdata/aurora/CDdata_aurora2/noises /esat/spchdisk/scratch/jponcele/fhvae_jakob/datasets/timit_np_fbank_4 /esat/spchdisk/scratch/jponcele/fhvae_jakob/datasets/timit_np_fbank_noisy
 /users/spraak/jponcele/NoiseFilter --ftype fbank
"""
SNR_values = ["0", "5", "10", "15", "20"]

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("noises_dir", type=str,
        help="directory containing raw noisefiles of aurora 2, e.g. aurora/CDdata_aurora2/noises")
parser.add_argument("timit_exp_dir", type=str,
        help="TIMIT expdir to which to add noise")
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
with open(os.path.join(args.timit_exp_dir, "test", "wav.scp"), "r") as fd:
    line = fd.readline()
    while line:
        a = line.rstrip().split(" ")[1]
        wavs.append(line.rstrip().split(" ")[1])
        line = fd.readline()

# convert .wav to .raw
raws = []
noisy_raws = []
for wav in wavs:
    fname = (wav.split("/")[-1]).split('.')[0]
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
os.symlink(os.path.join(args.timit_exp_dir, "train"), os.path.join(args.out_dir, "train"))
os.symlink(os.path.join(args.timit_exp_dir, "dev"), os.path.join(args.out_dir, "dev"))

clean_wav_dict = {}
with open(os.path.join(args.timit_exp_dir, "test", "wav.scp"), 'r') as rd:
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
for file in os.listdir(os.path.join(args.timit_exp_dir, "fac")):
    if file.startswith("all_facs_"):
        with open(os.path.join(args.timit_exp_dir, "fac", file), 'r') as fp:
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

print("Done.")