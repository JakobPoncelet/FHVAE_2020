import os
import sys
import shutil

'''
To split the testing for noisy cgn (too large to do in one script, memory/time crash)

Thus script will initialize <num_splits> directories in <expdir>, and copies the best/latest training checkpoint
to each subdirectory. The number of the split is added to the configuration file.

Afterwards, one can launch the test script for each subdirectory in parallel, by setting the --expdir parameter
in the test script to <expdir>/test_split/split_<i>.
Every subdirectory only uses 1/num_splits amount of the data (chosen by interleaving). 
For cgn, the amount of clean seqs is divisible by 6, so use this.

USAGE: "python split_test.py <expdir>"
'''

#########################
#cgn: 1734 clean seqs, deelbaar door 6
num_splits = 6
#########################

if len(sys.argv) != 2:
    print('Wrong arguments! Usage: python split_test.py expdir')
    exit(1)

expdir = str(sys.argv[1])

if not os.path.exists(expdir):
    print('Expdir doesnt exist!')
    exit(1)

if os.path.exists(os.path.join(expdir, 'test_split')):
    shutil.rmtree(os.path.join(expdir, 'test_split'))

os.makedirs(os.path.join(expdir, 'test_split'))

for split in range(1, num_splits+1):
    dest = '%s/test_split/split%i' % (expdir, split)
    shutil.copytree('%s/training_checkpoints' % expdir, '%s/training_checkpoints' % dest)
    shutil.copy2('%s/trainconf.pkl' % expdir, dest)
    shutil.copy2('%s/config.cfg' % expdir, dest)
    with open('%s/config.cfg' % dest, 'a+') as pt:
        pt.write('\ninterleave = %i\n' % split)



