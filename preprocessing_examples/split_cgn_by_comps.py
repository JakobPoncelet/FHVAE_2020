import os
import shutil

'''
Create train/dev/test split for split cgn files to train a model,
based on training components of your choice.
Testing is done on complete components, instead of with a file like comps_devset.txt
(see make_cgn_kaldi.py).
This model can later be used to extract z1 features for kaldi

'''


###########
traincomps = 'okljmn'
testcomps = 'h'
devfrac = 0.05
outdir = '/esat/spchdisk/scratch/jponcele/fhvae_jakob/datasets/cgn_okljmn_h/'
indir = '/esat/spchtemp/scratch/jponcele/cgn_allcomps_sequences_augmented/train'
indir_fac = '/esat/spchtemp/scratch/jponcele/cgn_allcomps_sequences_augmented/fac'
num_noisy_versions = 4
###########

for nset in ['train', 'dev', 'test', 'fac']:
    os.makedirs(os.path.join(outdir, nset), exist_ok=True)

clean_utts = set()
for fname in ['wav.scp', 'feats.scp', 'len.scp']:
    tr_size, tt_size = 0, 0
    out_tr = open(os.path.join(outdir, 'train', fname), 'w')
    out_tt = open(os.path.join(outdir, 'test', fname), 'w')
    with open(os.path.join(indir, fname), 'r') as rd:
        line = rd.readline()
        while line:
            uttid = line.rstrip().split(' ')[0]
            comp = uttid.split('_')[2]
            if comp in traincomps:
                out_tr.write(line)
                tr_size += 1
                if 'dB' not in uttid:
                    clean_utts.add(uttid)
            elif comp in testcomps:
                out_tt.write(line)
                tt_size += 1
            line = rd.readline()
    out_tr.close()
    out_tt.close()

num_dev_utts = int(devfrac*len(clean_utts))
devset_utts = set(list(clean_utts)[0:num_dev_utts])

for fname in ['wav.scp', 'feats.scp', 'len.scp']:
    dev_size = 0
    out_f = open(os.path.join(outdir, 'dev', fname), 'w')

    with open(os.path.join(outdir, 'train', fname), 'r') as rd:
        line = rd.readline()
        while line:
            uttid = line.rstrip().split(' ')[0]
            if len(uttid.split('_')) > 3:
                uttid = '_'.join(uttid.split('_')[0:3])
            if uttid in devset_utts:
                out_f.write(line)
                dev_size += 1
            line = rd.readline()
    out_f.close()

print('Made dataset with %i trainseqs, %i devseqs, %i testseqs\n' % (tr_size, dev_size, tt_size))

for fac in os.listdir(indir_fac):
    out_f = open(os.path.join(outdir, 'fac', fac), 'w')
    with open(os.path.join(indir_fac, fac), 'r') as rd:
        line = rd.readline()
        if len(line.split(' ')) == 2:
            while line:
                uttid = line.rstrip().split(' ')[0]
                comp = uttid.split('_')[2]
                if comp in traincomps or comp in testcomps:
                    out_f.write(line)
                line = rd.readline()
        elif len(line.split(' ')) == 1:
            while line:
                if len(line.split(' ')) == 1:
                    uttid = line.rstrip()
                    comp = uttid.split('_')[2]
                    if comp in traincomps or comp in testcomps:
                        flag = True
                    else:
                        flag = False
                if flag:
                    out_f.write(line)
                line = rd.readline()
    out_f.close()

print('Copied necessary factor files')
