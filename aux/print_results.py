import os
import sys

'''
Find 'accuracy.txt' file for each classification experiment performed and print out the results to compare experiments.

USAGE: "python3 print_results.py <accuracy_window>"
	    accuracy_window: default=0
'''

######################################################################
## SETUP (use python3)
# location of all experiments
expdir = '/esat/spchdisk/scratch/jponcele/fhvae_jakob/summary_20june'
# list of keywords of which one has to be present in experiments
keywords = ['exp']  #['normflow', 'NF', 'may', 'noisy']
######################################################################

if len(sys.argv) == 2:
	accwin = int(sys.argv[1])
else:
	accwin = 0

print('PRINTING RESULTS OF ACCURACY WINDOW = %i' % accwin)
print('only results from z1!  (otherwise change line classifier.startswith)')
# print in order of least recently created(/edited)
for exp in sorted(os.scandir(expdir), key=lambda d: d.stat().st_mtime):
	exp = exp.name
	if any(x in exp for x in keywords):
		print('Experiment: %s' % exp)
		classdir = os.path.join(expdir, exp, 'classifier_exp')
		if os.path.exists(classdir):
			for classifier in os.listdir(classdir):
				if classifier.startswith('z1') and not classifier.startswith('z1_z2'):

					if 'phones' in classifier:

						res = os.path.join(classdir, classifier, 'results', 'accuracy.txt')
						with open(res, 'r') as pd:
							res = pd.read()
							# find result for accuracy window
							acc = (res.split('\t')[accwin+1]).split(' ')[-1]
							print('\t '+str(res.split('\t')[0])+'\t'+str(acc))
