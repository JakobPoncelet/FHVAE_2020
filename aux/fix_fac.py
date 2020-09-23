import os

facdir ='/esat/spchdisk/scratch/jponcele/fhvae_jakob/datasets/timit_np_fbank_multicond_large/fac'

for fac in os.listdir(facdir):
    if fac.startswith('all_facs_'):
        with open(os.path.join(facdir, fac), 'r') as pd:
            updfac = os.path.join(facdir, fac+'_tmp')
            with open(updfac, 'w+') as td:
                line = pd.readline()
                if len(line.split(' ')) == 2:
                    while line:
                        seq = line.rstrip().split(' ')[0]
                        cleanseq = '_'.join(seq.split('_')[0:2])
                        lab = line.rstrip().split(' ')[1]
                        td.write(line)
                        td.write(cleanseq+' '+lab+'\n')
                        line = pd.readline()
                elif len(line.split(' ')) == 1:
                    talab_lines = []
                    seq = line.rstrip()
                    line = pd.readline()
                    while line:
                        if len(line.split(' ')) == 1:
                            td.write(seq+'\n')
                            for talab in talab_lines:
                                td.write(talab)
                            cleanseq = '_'.join(seq.split('_')[0:2])
                            td.write(cleanseq+'\n')
                            for talab in talab_lines:
                                td.write(talab)
                            
                            seq = line.rstrip()
                            talab_lines = []
                        else:
                            talab_lines.append(line)
                        line = pd.readline()
                    
                    td.write(seq+'\n')
                    for talab in talab_lines:
                        td.write(talab)
                    cleanseq = '_'.join(seq.split('_')[0:2])
                    td.write(cleanseq+'\n')
                    for talab in talab_lines:
                        td.write(talab)




