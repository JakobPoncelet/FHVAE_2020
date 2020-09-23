import os
import shutil

'''
Create train/dev/test split for split cgn files to train a model,
based on training components of your choice,
and using the comps_devset.txt file to choose the test set.
(to test on a complete component instead of predefined uttlist, see split_cgn_by_comps.py)
This model can later be used to extract z1 features for kaldi
'''

###########
traincomps = 'abcdefghijklmno'
testcomps = ''
devfrac = 0.05
#outdir = '/esat/spchdisk/scratch/jponcele/fhvae_jakob/datasets/cgn_allcomps_and_telephone'
#indir = '/esat/spchtemp/scratch/jponcele/cgn_allcomps_sequences_augmented/train'
#indir_fac = '/esat/spchtemp/scratch/jponcele/cgn_allcomps_sequences_augmented/fac'
#splitdetails = '/esat/spchtemp/scratch/jponcele/cgn_allcomps_sequences/wav/split_details.txt'
outdir = '/esat/spchdisk/scratch/jponcele/fhvae_jakob/datasets/cgn_allcomps_and_telephone_without_augmentation'
indir = '/esat/spchtemp/scratch/jponcele/cgn_allcomps_sequences/train'
indir_fac = '/esat/spchtemp/scratch/jponcele/cgn_allcomps_sequences/fac'
splitdetails = '/esat/spchtemp/scratch/jponcele/cgn_allcomps_sequences/wav/split_details.txt'
testset = './misc/cgn/comps_devset.txt'
num_noisy_versions = 4
###########

for nset in ['train', 'dev', 'test', 'fac']:
    os.makedirs(os.path.join(outdir, nset), exist_ok=True)

testfiles = set()
with open(testset, 'r') as rd:
    line = rd.readline()
    while line:
        testfiles.add(line.rstrip())
        line = rd.readline()

test_utts = set()
with open(splitdetails, 'r') as rd:
    line = rd.readline()
    line = rd.readline()
    while line:
        if line.rstrip().split(' ')[0] == 'COMPONENT':
            comp = line.rstrip().split(' ')[2]
        elif line.rstrip().split(' ')[0] == 'FILE:':
            fileid = line.rstrip().split(' ')[1]
        else:
            if fileid in testfiles:
                for uttid in line.rstrip().split(' '):
                    real_uttid = uttid+'_'+comp
                    test_utts.add(real_uttid)
        line = rd.readline()

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
            if 'dB' in uttid:
                clean_uttid = '_'.join(uttid.split('_')[0:3])
            else:
                clean_uttid = uttid
            if clean_uttid in test_utts:
                out_tt.write(line)
                tt_size += 1
            else:
                if comp in traincomps:
                    out_tr.write(line)
                    tr_size += 1
                    if 'dB' not in uttid:
                        clean_utts.add(uttid)
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
                if uttid in clean_utts or uttid in test_utts:
                    out_f.write(line)
                line = rd.readline()
        elif len(line.split(' ')) == 1:
            while line:
                if len(line.split(' ')) == 1:
                    uttid = line.rstrip()
                    if uttid in clean_utts or uttid in test_utts:
                        flag = True
                    else:
                        flag = False
                if flag:
                    out_f.write(line)
                line = rd.readline()
    out_f.close()

print('Copied necessary factor files')
