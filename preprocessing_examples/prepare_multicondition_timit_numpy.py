from __future__ import division
import os
import shutil
import wave
import argparse
import subprocess
import random
from sphfile import SPHFile
from collections import defaultdict

"""
add noise to TIMIT train+dev at several SNRs

This script expects you already have created
 - a clean dataset with prepare_timit_numpy.py
 - a noisy test set with prepare_noisy_timit_numpy.py
 
Training: 25% clean data, 75% with added noise of average SNR 15dB ([10dB --> 20dB])
Test: clean + all noises added to every test 

## COMMAND PARAMETERS:
/users/spraak/spchdata/aurora/CDdata_aurora2/noises
/esat/spchdisk/scratch/jponcele/fhvae_jakob/datasets/timit_np_fbank_noisy
/esat/spchdisk/scratch/jponcele/fhvae_jakob/datasets/timit_np_fbank_multicond
/users/spraak/jponcele/NoiseFilter --ftype fbank
"""
# as in AURORA4
train_SNR_max = 20
train_SNR_min = 10

# testing
# SNR_values = ["0", "5", "10", "15", "20"]


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("noises_dir", type=str,
        help="directory containing raw noisefiles of aurora 2, e.g. aurora/CDdata_aurora2/noises")
parser.add_argument("timit_exp_dir", type=str,
        help="TIMIT dir with noisy test set")
parser.add_argument("out_dir", type=str,
        help="output data directory")
parser.add_argument("noisefilter", type=str,
        help="dir with compiled noise filt_add program")
parser.add_argument("--ftype", type=str, default="fbank", choices=["fbank", "spec"],
        help="feature type")
args = parser.parse_args()
print(args)

# init dirs
os.makedirs(os.path.join(args.out_dir, "wav"), exist_ok=True)

# find the .raw noises
noises = []
os.makedirs(os.path.join(args.out_dir, "noises"), exist_ok=True)
for noise in os.listdir(args.noises_dir):
    shutil.copy2(os.path.join(args.noises_dir, noise), os.path.join(args.out_dir, "noises"))
    noises.append(os.path.join(args.out_dir, "noises", noise))

nnames = {}
nspec = {}

for set in ["train", "dev"]:

    os.makedirs(os.path.join(args.out_dir, set), exist_ok=True)

    # find .wav files of set that we want to add noise to
    wavs = []
    with open(os.path.join(args.timit_exp_dir, set, "wav.scp"), "r") as fd:
        line = fd.readline()
        while line:
            wavs.append(line.rstrip().split(" ")[1])
            line = fd.readline()

    # only add noise to 75% of data
    # wavs = [wav for idx, wav in enumerate(wavs) if idx % 4 != 0]

    random.shuffle(wavs)
    clean_wavs = wavs[0:int(len(wavs)/4)]
    noisy_wavs = wavs[int(len(wavs)/4):]

    nspec[set] = {}
    for noise in noises:
        nspec[set][noise] = defaultdict(list)

    for wav in noisy_wavs:
        noise = random.choice(noises)
        SNR = random.randint(train_SNR_min, train_SNR_max)
        nspec[set][noise][SNR].append(wav)


    os.makedirs(os.path.join(args.out_dir, "tmp_raw_files"), exist_ok=True)

    scp_dict = {}

    for noise in nspec[set]:
        for SNR in nspec[set][noise]:
            wavlist = nspec[set][noise][SNR]

            # convert .wav to .raw
            raws = []
            noisy_raws = []
            os.makedirs(os.path.join(args.out_dir, "tmp_raw_mixed_files"), exist_ok=True)

            for wav in wavlist:
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

            # p341 is filter for 16kHz data, r is seed for randgen
            # -m is how SNR is calculated: after a_weight or on 0-8kHz signal with -m snr_8khz
            command = "%s -i %s -o %s -n %s -u -f p341 -s %s -r 1000 -m a_weight -e %s" \
                      % (prog, in_list, out_list, noise, str(SNR), log)
            subprocess.call(command, shell=True)

            # convert mixed .raw to .wav files
            noisespec = (noise.split('/')[-1]).split('.')[0]+"_"+str(SNR)+"_dB"
            for noisy_raw in noisy_raws:
                fname = (noisy_raw.split("/")[-1]).split('.')[0]
                nname = fname + "_" + noisespec
                out = os.path.join(args.out_dir, "wav", nname + ".wav")
                cmdstring = \
                    "sox -r 16k --bits 16 --encoding signed-integer -c 1 --endian little %s %s" \
                     % (noisy_raw, out)
                subprocess.call(cmdstring, shell=True)
                scp_dict[nname] = out
                nnames[fname] = nname

            # clean up
            shutil.rmtree(os.path.join(args.out_dir, "tmp_raw_mixed_files"))

    shutil.rmtree(os.path.join(args.out_dir, "tmp_raw_files"))

    clean_wav_dict = {}
    with open(os.path.join(args.timit_exp_dir, set, "wav.scp"), 'r') as rd:
        line = rd.readline()
        while line:
            fname = line.split(" ")[0]
            loc = line.rstrip().split(" ")[1]
            clean_wav_dict[fname] = loc
            line = rd.readline()

    with open(os.path.join(args.out_dir, set, "wav.scp"), "w+") as td:
        for wav in clean_wavs:
            wavname = (wav.split('/')[-1]).split('.')[0]
            td.write(wavname+" "+clean_wav_dict[wavname]+"\n")
        for fname, loc in scp_dict.items():
            td.write(fname + " " + loc + "\n")

# link to noisy test set
os.symlink(os.path.join(args.timit_exp_dir, "test"), os.path.join(args.out_dir, "test"))

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

for set in ["train", "dev"]:
    compute_feature(set)

    print("computed features for %s set" % set)

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
                        wav = line.split(' ')[0]
                        lab = line.rstrip().split(' ')[1]
                        # see if noise added
                        if wav in nnames:
                            nname = nnames[wav]
                            tp.write(nname+ ' ' +lab + '\n')
                        else:
                            tp.write(line)
                        line = fp.readline()
                # talabs
                elif len(line.split(' ')) == 1:
                    talab_lines = []
                    wav = line.rstrip()
                    line = fp.readline()
                    while line:
                        # name
                        if len(line.split(' ')) == 1:
                            # see if noise added
                            if wav in nnames:
                                nname = nnames[wav]
                                tp.write(nname + '\n')
                                for talab in talab_lines:
                                    tp.write(talab)
                            else:
                                tp.write(wav+'\n')
                                for talab in talab_lines:
                                    tp.write(talab)

                            wav = line.rstrip()
                            talab_lines = []
                        else:
                            talab_lines.append(line)
                        line = fp.readline()


                    # last seq of file needs to be handled separately
                    tp.write(wav+'\n')
                    for talab in talab_lines:
                        tp.write(talab)

                    if wav in nnames:
                        nname = nnames[wav]
                        tp.write(nname + '\n')
                        for talab in talab_lines:
                            tp.write(talab)

print("written factor files")

fc = open(os.path.join(args.out_dir, 'fac', 'all_facs_noise.scp'), 'w+')
fs = open(os.path.join(args.out_dir, 'fac', 'all_facs_snr.scp'), 'w+')
fn = open(os.path.join(args.out_dir, 'fac', 'all_facs_noisetype.scp'), 'w+')

with open(os.path.join(args.out_dir, 'fac', 'all_facs_gender.scp'), 'r') as pd:
    line = pd.readline()
    while line:
        seq = line.rstrip().split(' ')[0]
        if seq.endswith('dB'):
            snr = seq.split('_')[-2]
            ntype = seq.split('_')[-3]
            fc.write(seq+' '+'noisy\n')
            fs.write(seq+' '+str(snr)+'\n')
            fn.write(seq+' '+str(ntype)+'\n')

        else:
            fc.write(seq+' '+'clean\n')
            fs.write(seq+' '+'clean\n')
            fn.write(seq+' '+'clean\n')

        line = pd.readline()

fc.close()
fs.close()
fn.close()

with open(os.path.join(args.out_dir, 'fac', 'training_noise_summary.txt'), 'w+') as f:
    for set in nspec:
        f.write('SET: %s \n' % set)
        for noise in nspec[set]:
            noisename = noise.split('/')[-1]
            f.write('\t Noise: %s \n' % noisename)
            for SNR in sorted(nspec[set][noise]):
                f.write('\t \t SNR: %s \n' % str(SNR))
                for wav in nspec[set][noise][SNR]:
                    wavname = (wav.split('/')[-1]).split('.')[0]
                    f.write('\t \t \t %s \n' % str(wavname))

print("Done.")