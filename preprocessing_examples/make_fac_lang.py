import os

indir = '/esat/spchdisk/scratch/jponcele/fhvae_jakob/datasets/cgn_vlnl_all_without_augmentation'

compfac = os.path.join(indir, 'fac', 'all_facs_comp.scp')
langfac = os.path.join(indir, 'fac', 'all_facs_lang.scp')

with open(langfac, 'w') as tp:
    with open(compfac, 'r') as pd:
        line = pd.readline()
        while line:
            uttid = line.rstrip().split(' ')[0]
            lang = str(uttid[0])  #vXXXXXX for Flemish, nXXXXXX for Dutch
            tp.write(uttid+' '+lang+'\n')
            line = pd.readline()
