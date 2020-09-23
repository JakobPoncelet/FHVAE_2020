"""
prepare CGN data for FHVAE

## COMMAND PARAMETERS:
/esat/spchdisk/scratch/jponcele/fhvae_jakob/datasets/cgn_np_fbank_afgklno --ftype fbank --out_dir /esat/spchdisk/scratch/jponcele/fhvae_jakob/datasets/cgn_np_fbank_afgklno

/esat/spchtemp/scratch/jponcele/cgn_nl_sequences --ftype fbank --out_dir /esat/spchtemp/scratch/jponcele/cgn_nl_sequences
"""
import os
import wave
import argparse
import subprocess
from sphfile import SPHFile

# os.environ["CUDA_VISIBLE_DEVICES"]="0"

"""
# FIRST RUN ./misc/makewavs_CGN.m IN MATLAB TO GENERATE SMALLER AUDIO FILES FROM THE CGN DATABASE

use as matlab arguments, based on cgn_dir and --out_dir argument you will use in this script:

outdir = '<cgn_dir>/wav/'
talabfile = '<out_dir>/fac/all_facs_phones.scp'
components = 'afgklno'   (for example)

NOTE: the matlab file can give an alarm for some wav files and doesnt store the phones for those files 
[alarm = length(sam)/fs*100 < length(label)-1].
In this script those wav files are filtered out (line 139), but might be better to change when working with unsupervised data, i.e. add them to the files but empty factors.
"""

#componentslist = ["comp-a", "comp-f", "comp-g", "comp-k", "comp-l", "comp-n", "comp-o"]
# componentslist = ["comp-k", "comp-o"]
#componentslist = ["comp-a", "comp-b", "comp-e", "comp-f", "comp-g", "comp-h", "comp-i", "comp-j", "comp-k", "comp-l", "comp-m", "comp-n", "comp-o"]
#componentslist = ["comp-c", "comp-d"]

componentslist = ["comp-a", "comp-b", "comp-c", "comp-d", "comp-e", "comp-f", "comp-g", "comp-h", "comp-i", "comp-j", "comp-k", "comp-l", "comp-m", "comp-n", "comp-o"]

def maybe_makedir(d):
    try:
        os.makedirs(d)
    except OSError:
        pass


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("cgn_dir", type=str,
                    help="data directory containing the split wav files of CGN (made by makewavs003.m)")
parser.add_argument("--ftype", type=str, default="fbank", choices=["fbank", "spec"],
                    help="feature type")
parser.add_argument("--out_dir", type=str, default="./datasets/cgn_per_speaker",
                    help="output data directory")
parser.add_argument("--dev_spk", type=str, default="./misc/cgn/cgn_per_spk_afgklno_dev.list",
                    help="path to list of dev set speakers")
parser.add_argument("--test_spk", type=str, default="./misc/cgn/cgn_per_spk_afgklno_test.list",
                    help="path to list of test set speakers")
parser.add_argument("--factor_file", type=str, default="./misc/cgn/speakers_regions.csv",
                    help="file containing factors for every speaker (like region, sex, ...)")
args = parser.parse_args()
print(args)

# retrieve partition
with open(args.dev_spk) as f:
    dt_spks = [line.rstrip().lower() for line in f]
with open(args.test_spk) as f:
    tt_spks = [line.rstrip().lower() for line in f]

dt_spks=[]
tt_spks=[]


# compute regularizing factors
# reg1=BirthRegion, reg2=ResRegion, res3=EducationRegion
with open(args.factor_file) as f:
    speaker=[l.rstrip('\n').split(';')[0].lower() for l in f]
    f.seek(0)
    gender=[l.rstrip('\n').split(';')[1] for l in f]
    f.seek(0)
    reg1=[l.rstrip('\n').split(';')[2] for l in f]
    f.seek(0)
    reg2=[l.rstrip('\n').split(';')[3] for l in f]
    f.seek(0)
    reg3=[l.rstrip('\n').split(';')[4] for l in f]
    f.seek(0)
    size=[l.rstrip('\n').rstrip('\r').split(';')[5] for l in f]
    f.close()

f_spk=open(os.path.join(args.out_dir, 'fac',"all_facs_spk.scp"),"w")
f_comp=open(os.path.join(args.out_dir, 'fac',"all_facs_comp.scp"),"w")
f_reg1=open(os.path.join(args.out_dir, 'fac', "all_facs_reg1.scp"),"w")
f_reg2=open(os.path.join(args.out_dir, 'fac', "all_facs_reg2.scp"),"w")
f_reg3=open(os.path.join(args.out_dir, 'fac', "all_facs_reg3.scp"),"w")
f_gender=open(os.path.join(args.out_dir, 'fac', "all_facs_gender.scp"),"w")

fac_dict = {}

with open(os.path.join(args.out_dir, 'fac', "all_facs_phones.scp")) as f:
    for l in f:
        line=l.split()
        if len(line)==1:
            (spk,nr,comp)=line[0].split('_')
            spk_and_nr = str(spk)+'_'+str(nr)
            fac_dict[spk_and_nr] = comp
            k = speaker.index(spk.lower())
            f_spk.write('%s %s\n' % (line[0],spk))
            f_comp.write('%s %s\n' % (line[0], comp))
            f_reg1.write('%s %s\n' % (line[0], reg1[k]))
            f_reg2.write('%s %s\n' % (line[0], reg2[k]))
            f_reg3.write('%s %s\n' % (line[0], reg3[k]))
            f_gender.write('%s %s\n' % (line[0], gender[k]))

f_spk.close()
f_comp.close()
f_reg1.close()
f_reg2.close()
f_reg3.close()
f_gender.close()

print("stored regularizing factors")


print("making wav.scp")

# wav_dir = os.path.abspath("%s/wav" % args.out_dir)
tr_scp = "%s/train/wav.scp" % (args.out_dir)
dt_scp = "%s/dev/wav.scp" % (args.out_dir)
tt_scp = "%s/test/wav.scp" % (args.out_dir)

# maybe_makedir(wav_dir)
maybe_makedir(os.path.dirname(tr_scp))
maybe_makedir(os.path.dirname(dt_scp))
maybe_makedir(os.path.dirname(tt_scp))

tr_f = open(tr_scp, "w")
dt_f = open(dt_scp, "w")
tt_f = open(tt_scp, "w")

paths = []
bad_files = []
cnt = 0

for root, _, fnames in sorted(os.walk(args.cgn_dir)):
    regio = root.split("/")[-1].lower()  #nl/vl
    comp = root.split("/")[-2].lower()
    if comp not in componentslist:
        continue
    for fname in fnames:
        if fname.endswith(".wav") or fname.endswith(".WAV"):
            spk = fname.split("_")[0].lower()
            spk_and_nr = fname.split(".")[0].lower()

            # # a wav file was made but no phones stored
            # if spk_and_nr not in fac_dict:
            #     lookup = spk_and_nr+'_'+comp.split("-")[1]
            #     bad_files.append(lookup)
            #     continue

            if spk in dt_spks:
                f = dt_f
            elif spk in tt_spks:
                f = tt_f
            else:
                f = tr_f

            #path = "%s/%s/%s/%s" % (args.cgn_dir, comp, regio, fname)
            path = "%s/%s" % (root, fname)
            uttid = "%s_%s" % (spk_and_nr, comp.split("-")[1])
            f.write("%s %s\n" % (uttid, path))
            cnt += 1

tr_f.close()
dt_f.close()
tt_f.close()

# print("wavs without phone talabs that are left out:   (%i out of %i)" % (len(bad_files), cnt))
# print(bad_files)

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
