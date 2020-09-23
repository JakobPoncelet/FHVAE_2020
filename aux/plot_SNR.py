import os
import sys
import argparse
import matplotlib
import matplotlib.pyplot as plt
import pickle
import shutil

'''
Create plots of phone classification accuracy in function of SNR 
  - for different latent variables used, specified in <latent_vars>
  - for different accuracy windows
  - in a comparison plot

This script is also called from plot_summary.py.

USAGE: python plot_SNR.py <expdir>
'''


def main(expdir):

    #####################################
    latent_vars = ['z1', 'z2', 'z1_z2']
    show = True
    #####################################

    comparison_plots(expdir, show, latent_vars)


def comparison_plots(expdir, show=False, latent_vars=['z1', 'z2', 'z1_z2']):

    print('Creating SNR plots of noisy test in %s' % expdir)

    plotsdir = os.path.join(expdir, 'SNR_plots')
    if os.path.exists(plotsdir):
        shutil.rmtree(plotsdir)
    os.makedirs(plotsdir)

    ## TO DO: check if starts with z1 or z1_z2 and make comparison plot

    # find classifier with best accuracy on complete set (and accuracy_window = 0)
    # this is not only clean!
    clean_accs = {}
    for exp in os.listdir(os.path.join(expdir, "classifier_exp")):
        if exp.startswith('z1') and (not exp.startswith('z1_z2')):
            with open(os.path.join(expdir, "classifier_exp", exp, "results", "accuracy.txt"), 'r') as pd:
                res = pd.read()
                acc = float((res.split('\t')[1]).split(' ')[-1])
            clean_accs[exp] = acc
    best_exp = max(clean_accs, key=clean_accs.get)

    eval_dicts = {}
    # only z1 should always exist
    with open(os.path.join(expdir, "classifier_exp", best_exp, "results", "acc_win_noisy_eval_dict.pkl"), "rb") as fid:
        eval_dict_z1 = pickle.load(fid)
        eval_dicts['z1'] = eval_dict_z1

    if os.path.exists(os.path.join(expdir,"classifier_exp", "z2" + best_exp[2:], "results", "acc_win_noisy_eval_dict.pkl")):
        with open(os.path.join(expdir,"classifier_exp", "z2" + best_exp[2:], "results", "acc_win_noisy_eval_dict.pkl"), "rb") as fid:
            eval_dict_z2 = pickle.load(fid)
            eval_dicts['z2'] = eval_dict_z2

    if os.path.exists(os.path.join(expdir, "classifier_exp", "z1_z2" + best_exp[2:], "results", "acc_win_noisy_eval_dict.pkl")):
        with open(os.path.join(expdir, "classifier_exp", "z1_z2" + best_exp[2:], "results", "acc_win_noisy_eval_dict.pkl"), "rb") as fid:
            eval_dict_z1_z2 = pickle.load(fid)
            eval_dicts['z1_z2'] = eval_dict_z1_z2

    if True:
        for var in latent_vars:
            if var in eval_dicts:
                eval_dict = eval_dicts[var]
                for accwin in eval_dict:
                    snrperf = eval_dict[accwin]['SNR']
                    cleanperf = eval_dict[accwin]['clean']
                    lists = sorted(snrperf.items())
                    snr, acc = zip(*lists)
                    snr += ('clean',)
                    acc += (float(cleanperf),)

                    plt.figure(int(accwin))
                    plt.plot(snr, acc, label=var)
                    plt.ylabel('Phoneme accuracy')
                    plt.xlabel('SNR (dB)')
                    plt.title('Phoneme recognition in function of SNR for accuracywindow of %s' % str(accwin))
                    plt.legend()
                    plt.savefig(os.path.join(plotsdir, 'SNR_accwin_%i.pdf' % accwin), format='pdf')
                    if show:
                        plt.show()

    # comparison of accuracy windows
    if True:
        for i, var in enumerate(latent_vars):
            plt.figure(i+10)
            if var in eval_dicts:
                eval_dict = eval_dicts[var]
                for accwin in eval_dict:
                    snrperf = eval_dict[accwin]['SNR']
                    cleanperf = eval_dict[accwin]['clean']
                    lists = sorted(snrperf.items())
                    snr, acc = zip(*lists)
                    snr += ('clean',)
                    acc += (float(cleanperf),)
                    plt.plot(snr, acc, label='accuracy window = %i' % accwin)
                plt.ylabel('Phoneme accuracy')
                plt.xlabel('SNR (dB)')
                plt.title('Phoneme recognition in function of SNR for different accuracywindows for %s' % var)
                plt.legend()
                plt.savefig(os.path.join(plotsdir, 'SNR_accwin_comp.pdf'), format='pdf')
                if show:
                    plt.show()

    # comparison of z1/z2/z1_z2 for accwin = 0
    plt.figure(20)
    for var in latent_vars:
        if var in eval_dicts:
            eval_dict = eval_dicts[var]
            snrperf = eval_dict[0]['SNR']
            cleanperf = eval_dict[0]['clean']
            lists = sorted(snrperf.items())
            snr, acc = zip(*lists)
            snr += ('clean',)
            acc += (float(cleanperf),)
            plt.plot(snr, acc, label=var)
    plt.ylabel('Phoneme accuracy')
    plt.xlabel('SNR (dB)')
    plt.title('Phoneme recognition in function of SNR')
    plt.legend()
    plt.savefig(os.path.join(plotsdir, 'SNR_latentvar_comp.pdf'), format='pdf')
    if show:
        plt.show()

    plt.close('all')


if __name__ == '__main__':

    # parse the arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("expdir", type=str,
                        help="directory containing experiment of which you want to make plots")
    args = parser.parse_args()

    print("Expdir: %s" % args.expdir)
    if not os.path.isdir(args.expdir):
        print("Expdir does not exist.")
        exit(1)

    main(args.expdir)




