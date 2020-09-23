import os
import shutil

'''
Add NL data to a prepared FHVAE dataset.
Choose which NL components to include to the TRAIN set.
Only clean NL data.
'''

#####
comps = "abcdefghijklmno"
nl_dir = "/esat/spchtemp/scratch/jponcele/cgn_nl_sequences"
out_dir = "/esat/spchdisk/scratch/jponcele/fhvae_jakob/datasets/cgn_vlnl_all_without_augmentation"
#####

for f in ["wav.scp", "len.scp", "feats.scp"]:
    with open(os.path.join(out_dir, "train", f), "a") as td:
        with open(os.path.join(nl_dir, "train", f), "r") as rd:
            line = rd.readline()
            while line:
                uttid = line.rstrip().split(' ')[0]
                comp = uttid.split('_')[2]
                if comp in comps:
                    td.write(line)
                line = rd.readline()

for fac in os.listdir(os.path.join(out_dir, "fac")):
    if not fac.endswith('.scp'):
        continue
    if not fac.startswith('all_facs_'):
        continue
    if not os.path.exists(os.path.join(nl_dir, "fac", fac)):
        print("no factor file in %s for factor %s" % (nl_dir, fac))
        continue
    if fac == "all_facs_phones.scp":
        with open(os.path.join(out_dir, "fac", fac), "a") as td:
            with open(os.path.join(nl_dir, "fac", fac), "r") as rd:
                line = rd.readline()
                flag = False
                while line:
                    if len(line.split(' ')) == 1:
                        uttid = line.rstrip()
                        comp = uttid.split('_')[2]
                        if comp in comps:
                            flag = True
                        else:
                            flag = False
                    if flag:
                        td.write(line)
                    line = rd.readline()
    else:
        with open(os.path.join(out_dir, "fac", fac), "a") as td:
            with open(os.path.join(nl_dir, "fac", fac), "r") as rd:
                line = rd.readline()
                while line:
                    uttid = line.rstrip().split(' ')[0]
                    comp = uttid.split('_')[2]
                    if comp in comps:
                        td.write(line)
                    line = rd.readline()
