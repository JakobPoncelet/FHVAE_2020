import os
import sys
import glob
import shutil
import argparse
import pickle
import matplotlib
import matplotlib.pyplot as plt
import subprocess
from plot_SNR import comparison_plots

'''
Compare results of multiple experiments located in a higher-level directory <expdir>
and print/store the results in <outdir>
     - print accuracies of labels on clean/noisy test set, e.g. phones from z1, gender from z2
       (computed from softmax layer)
     - print adversarial accuracies of labels on clean/noisy test set, e.g. phones from z2, gender from z1
       (computed from softmax layer)
     - print phone classification accuracies (with external phone classifier) on clean/noisy test set 
       for different accuracy windows, for different latent variables as input and for different SNR levels

USAGE: python3 plot_summary.py <expdir> --outdir <outdir>
see Argument Parser below
'''


def main(expdir, outdir):

    summarize_label_accuracies(expdir, outdir, 'label_accuracy')
    summarize_label_accuracies(expdir, outdir, 'adversarial_accuracy')
    summarize_classifier(expdir, outdir, 'phones')
    summarize_classifier(expdir, outdir, 'gender')
    ## summary_15june (timit)
    # create_snr_plots(expdir, outdir, accuracy_window=0,
    #                  compare_exps=[[0,1], [0,1,2,3], [2,3,4,5], [6,7], [0,1,6,7]])

    ## summary_20june (cgn)
    #create_snr_plots(expdir, outdir, fac='phones', accuracy_window=0,
    #                 compare_exps=[[0, 1]])

    ## summary_22june (timit)
    # create_snr_plots(expdir, outdir, accuracy_window=0,
    #                 compare_exps=[[2, 3], [3, 4], [4, 5], [2, 3, 4, 5]])
    create_snr_plots(expdir, outdir, accuracy_window=0,
                     compare_exps=None)


    create_pdf(outdir)
    create_plots_per_exp(expdir)


def summarize_label_accuracies(expdir, outdir, name):

    print(("\nSummarizing the %s with the softmax classifier\n" % name).upper())

    fff = os.path.join(outdir, '%s.txt' % name)
    if os.path.exists(fff):
        os.remove(fff)

    accdict = {'clean': {}, 'full': {}}

    for exp in sorted(os.listdir(expdir)):

        if not os.path.isdir(os.path.join(expdir, exp)):
            continue

        # tested the experiment in splits with split_test.py (e.g. for large datasets like cgn)
        if os.path.exists(os.path.join(expdir, exp, 'test_split')):
            splitdir = os.path.join(expdir, exp, 'test_split')

            os.makedirs("%s/total/%s" % (splitdir, name), exist_ok=True)
            for split in os.listdir(splitdir):
                if not os.path.exists("%s/%s/test/txt/%s" % (splitdir, split, name)):
                    continue
                for ft in glob.glob("%s/%s/test/txt/%s/*_predictions.scp" % (splitdir, split, name)):

                    with open(ft, 'r') as fid:
                        lines = fid.readlines()

                    # sommige segs komen voor in meerdere splits en zijn dus dubbel/tripel aanwezig...
                    # nog niet gefixt
                    scp = os.path.basename(ft)
                    if not os.path.exists("%s/total/%s/%s" % (splitdir, name, scp)):
                        with open("%s/total/%s/%s" % (splitdir, name, scp), "w+") as pid:
                            pid.writelines(lines)
                    else:
                        with open("%s/total/%s/%s" % (splitdir, name, scp), 'a') as pid:
                            pid.writelines(lines[1:])

            scpdir = "%s/total" % splitdir
        else:
            scpdir = "%s/%s/test/txt" % (expdir, exp)

        for fl in glob.glob(os.path.join(scpdir, name, '*_predictions.scp')):

            fac = os.path.basename(fl).split('_')[0]

            if fac in accdict['clean']:
                accdict['clean'][fac][exp] = 0.0
                accdict['clean'][fac][exp] = 0.0
            else:
                accdict['clean'][fac] = {exp: 0.0}
                accdict['full'][fac] = {exp: 0.0}

            clean_total = 0
            clean_correct = 0
            total = 0
            correct = 0
            clean = True

            with open(fl, 'r') as pd:
                line = pd.readline()
                line = pd.readline()

                # seq label
                if len(line.split(' ')) == 3:

                    while line:
                        clean = True
                        if 'dB' in line.split(' ')[0]:
                            clean = False

                        total += 1
                        if clean:
                            clean_total += 1

                        if line.rstrip().split(' ')[1].rstrip() == line.rstrip().split(' ')[2].rstrip():
                            correct += 1
                            if clean:
                                clean_correct += 1

                        line = pd.readline()

                else:
                    while line:
                        while len(line.rstrip().split(' ')) == 5:
                            if 'dB' in line.split(' ')[1]:
                                clean = False
                            else:
                                clean = True
                            line = pd.readline()

                        total += 1
                        if clean:
                            clean_total += 1

                        if line.rstrip().split('\t')[2].rstrip() == line.rstrip().split('\t')[3].rstrip():
                            correct += 1
                            if clean:
                                clean_correct += 1

                        line = pd.readline()

            acc = correct/total
            if clean_total == 0:
                clean_acc = 0.0
            else:
                clean_acc = clean_correct / clean_total

            # _print("%s of %s for fac %s is %f on clean and %f on full test set" % (name, exp, fac, clean_acc, acc), fff)

            accdict['clean'][fac][exp] = clean_acc
            accdict['full'][fac][exp] = acc

    for typ in ['clean', 'full']:
        _print('\n', fff)
        _print('%s %s' % (typ.upper(), name.upper()), fff)
        _print_table(accdict[typ], fff)


def summarize_classifier(expdir, outdir, fac=None):
    print(("\nSummarizing the classifier accuracies of factor %s\n" % fac).upper())

    fff = os.path.join(outdir, 'classifier.txt')
    if os.path.exists(fff):
        os.remove(fff)

    window_res = {}

    for exp in sorted(os.listdir(expdir)):

        if not os.path.isdir(os.path.join(expdir, exp, 'classifier_exp')):
            continue

        for classexp in sorted(os.listdir(os.path.join(expdir, exp, 'classifier_exp'))):

            if fac is not None:
                if fac not in classexp:
                    continue

            if classexp.startswith('z1_z2'):
                lvar = 'z1_z2'
            elif classexp.startswith('z1'):
                lvar = 'z1'
            elif classexp.startswith('z2'):
                lvar = 'z2'
            else:
                continue

            res = os.path.join(expdir, exp, 'classifier_exp', classexp, 'results')

            with open("%s/acc_win_noisy_eval_dict.pkl" % res, "rb") as pid:
                accdict = pickle.load(pid)

            for accwin in accdict:
                if accwin not in window_res:
                    window_res[accwin] = {'clean': {}, 'noisy_z1': {}, 'noisy_z2': {}, 'noisy_z1_z2': {}}

                clean_acc = accdict[accwin]['clean']
                noisy_acc = accdict[accwin]['SNR']

                if lvar not in window_res[accwin]['clean']:
                    window_res[accwin]['clean'][lvar] = {}

                if exp not in window_res[accwin]['clean'][lvar]:
                    window_res[accwin]['clean'][lvar][exp] = clean_acc
                else:
                    if clean_acc > window_res[accwin]['clean'][lvar][exp]:
                        window_res[accwin]['clean'][lvar][exp] = clean_acc

                for SNR in sorted(list(noisy_acc.keys())):
                    SNRstr = str(SNR)+' dB'
                    nname = 'noisy_'+lvar
                    if SNRstr not in window_res[accwin][nname]:
                        window_res[accwin][nname][SNRstr] = {}
                    if exp not in window_res[accwin][nname][SNRstr]:
                        window_res[accwin][nname][SNRstr][exp] = noisy_acc[SNR]
                    else:
                        if noisy_acc[SNR] > window_res[accwin][nname][SNRstr][exp]:
                            window_res[accwin][nname][SNRstr][exp] = noisy_acc[SNR]

    for name in ['clean', 'noisy_z1', 'noisy_z2', 'noisy_z1_z2']:
        for accwin in window_res:
            if name == 'clean':
                _print(("classification accuracies for accuracy window of %i on clean test set" % accwin).upper(), fff)
            else:
                lvar = name.split('_')[1:]
                lvar = '_'.join(lvar)
                _print(("classification accuracies on %s for accuracy window of %i on noisy test set" % (lvar, accwin)).upper(), fff)
            _print_table(window_res[accwin][name], fff)


def create_snr_plots(expdir, outdir, fac='phones', accuracy_window=0, compare_exps=None):
    '''
    compare_exps contains indices of which experiments (sorted in alphabetical order) to compare per figure
    for example: if compare_exps = [[0,1], [2,3]]
                     figure 1 will compare exps 0 and 1,
                     figure 2 will compare exps 2 and 3,
                     figure 3 will compare all exps (default included)
     set compare_exps = None to only compare all in one fig
    '''


    print(("\nCreating plots of the phone classifier accuracy in function of SNR").upper())
    print(("       (for an accuracy window of %s only)\n" % str(accuracy_window)).upper())

    plotdir = os.path.join(outdir, 'SNR_plots')
    if os.path.exists(plotdir):
        shutil.rmtree(plotdir)
    os.makedirs(plotdir)

    var_res = {'z1': {}, 'z2': {}, 'z1_z2': {}}

    for exp in sorted(os.listdir(expdir)):

        if not os.path.isdir(os.path.join(expdir, exp, 'classifier_exp')):
            continue

        for classexp in sorted(os.listdir(os.path.join(expdir, exp, 'classifier_exp'))):

            if fac not in classexp:
                continue

            if classexp.startswith('z1_z2'):
                lvar = 'z1_z2'
            elif classexp.startswith('z1'):
                lvar = 'z1'
            elif classexp.startswith('z2'):
                lvar = 'z2'
            else:
                continue

            res = os.path.join(expdir, exp, 'classifier_exp', classexp, 'results')
            with open("%s/acc_win_noisy_eval_dict.pkl" % res, "rb") as pid:
                accdict = pickle.load(pid)

            clean_acc = accdict[accuracy_window]['clean']
            noisy_acc = accdict[accuracy_window]['SNR']

            if exp not in var_res[lvar]:
                var_res[lvar][exp] = {}

            for SNR in sorted(noisy_acc.keys()):
                if SNR not in var_res[lvar][exp]:
                    var_res[lvar][exp][SNR] = noisy_acc[SNR]
                else:
                    if noisy_acc[SNR] > var_res[lvar][exp][SNR]:
                        var_res[lvar][exp][SNR] = noisy_acc[SNR]

            ##
            if 'clean' not in var_res[lvar][exp]:
                var_res[lvar][exp]['clean'] = clean_acc
            elif clean_acc > var_res[lvar][exp]['clean']:
                var_res[lvar][exp]['clean'] = clean_acc

    print(var_res)
    for lvar in var_res:
        all_exps = list(var_res[lvar].keys())
        if compare_exps is not None:
            explist = []
            for comps in compare_exps:
                explist.append([all_exps[i] for i in comps])
            explist.append(all_exps)
        else:
            explist = [all_exps]

        for i, exps in enumerate(explist):
            plt.figure()
            for exp in exps:
                expname = '_'.join(exp.split('_')[1:])
                res = var_res[lvar][exp]
                lists = res.items()
                snr, acc = zip(*lists)
                plt.plot(snr, acc, label=expname)
            plt.ylabel('Phoneme accuracy')
            plt.xlabel('SNR (dB)')
            plt.title('Phoneme recognition in function of SNR for %s' % lvar)
            plt.legend()
            if exps == all_exps:
                plt.savefig(os.path.join(plotdir, '%s_plot_all.pdf' % lvar), format='pdf')
                plt.show(block=True)
            else:
                nname = '_'.join([str(k) for k in compare_exps[i]])
                plt.savefig(os.path.join(plotdir, '%s_plot_%s.pdf' % (lvar, nname)), format='pdf')
                plt.close()


def create_plots_per_exp(expdir):

    print(("\nCreating plots per experiment (stored in every expdir)\n").upper())
    for exp in sorted(os.listdir(expdir)):

        if not os.path.isdir(os.path.join(expdir, exp, 'classifier_exp')):
            continue

        comparison_plots(os.path.join(expdir, exp), show=False)


def create_pdf(outdir):

    if subprocess.run(["which", "libreoffice"]).returncode != 0:
        return

    print("\nCREATING PDF OF RESULTS\n")
    with open(os.path.join(outdir, 'tmp.txt'), 'w') as fid:
        for item in sorted(os.scandir(outdir), key=lambda d: d.stat().st_mtime):
            if item.name.endswith('.txt'):
                with open(os.path.join(outdir, item.name), 'r') as pd:
                    lines = pd.readlines()
                    fid.writelines(lines)

    subprocess.run(["libreoffice", "--headless", "--invisible", "--convert-to", "pdf", "--outdir", "%s" % outdir, "%s" % os.path.join(outdir, 'tmp.txt')])
    os.remove(os.path.join(outdir, 'tmp.txt'))
    os.rename(os.path.join(outdir, 'tmp.pdf'), os.path.join(outdir, 'summary.pdf'))


def _print(cmd, fileloc):
    print(cmd)
    with open(fileloc, 'a') as ppp:
        print(cmd, file=ppp)


def _print_table(dct, fileloc):
    data = ['Exp']
    for fac in dct:
        data.append(fac)
    n_cols = len(data)

    skip = []

    for fac in dct:
        for exp in dct[fac]:
            if exp not in skip:
                data.append('_'.join(exp.split('_')[1:]))
                for fac in dct:
                    if exp in dct[fac]:
                        data.append("{:.2f}".format(dct[fac][exp]))
                    else:
                        data.append(' ')
                skip.append(exp)
    _format_table(data, n_cols, 20, 8, fileloc)


def _format_table(data, cols, headerwide, wide, fileloc):
    '''Prints formatted data on columns of given width.
    data = list of all data in table from left to right
    cols = #columns
    headerwide = width of first column
    wide = width of other columns
    '''
    n, r = divmod(len(data), cols)
    header_pat = '{{:{}}}'.format(headerwide)
    pat = '{{:{}}}'.format(wide)

    line = ''
    for _ in range(n):
        line += header_pat
        line += pat * (cols-1)
        line += '\n'
    last_line = pat * r
    _print(line.format(*data), fileloc)
    _print(last_line.format(*data[n*cols:]), fileloc)


if __name__ == '__main__':

    # parse the arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("expdir", type=str,
                        help="directory containing multiple experiments that you want to compare")
    parser.add_argument("--outdir", type=str, default=None,
                        help="where to store the results")
    args = parser.parse_args()

    print("Expdir: %s" % args.expdir)
    if not os.path.isdir(args.expdir):
        print("Expdir does not exist.")
        exit(1)

    if args.outdir is None:
        args.outdir = os.path.join(args.expdir, 'result')

    print("Outdir: %s" % args.outdir)
    if os.path.exists(args.outdir):
        print("Outdir exists already: overwriting!")
        shutil.rmtree(args.outdir)
    os.makedirs(args.outdir)

    main(args.expdir, args.outdir)
