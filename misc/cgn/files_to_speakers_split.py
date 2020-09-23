import os

####
meta = '/users/spraak/spchdata/cgn/data/meta/text/recordings.txt'
testlist = './misc/cgn/comps_devset.txt'
devfrac = 0.05
out_list = './misc/cgn/cgn_testset.list'
###

# get train-test split of files
with open(testlist, 'r') as rd:
    testfiles = rd.readlines()
testfiles = [f.strip().split('_')[0] for f in testfiles]  # remove suffix like _A0
testfiles = set(testfiles)

# get all speakers of the test set
testspks = set()
with open(meta,'r') as pd:
    line = pd.readline()
    idx = line.split('\t').index('speakerIDs')
    line = pd.readline()
    while line:
        file = line.split('\t')[0].strip()
        if file in testfiles:
            spk_list = line.split('\t')[idx]
            for spk in spk_list.split(','):
                testspks.add(spk.strip())
        line = pd.readline()

# save test speakers list
with open(out_list, 'w') as td:
    for spk in testspks:
        td.write(spk+'\n')
