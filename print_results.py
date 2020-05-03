import os

######################################################################
## SETUP (use python3)
# location of all experiments
expdir = '/esat/spchdisk/scratch/jponcele/fhvae_jakob'
# list of keywords of which one has to be present in experiments
keywords = ['29apr','27april','24april','23april','21april','19april']
######################################################################

print('PRINTING RESULTS OF ACCURACY WINDOW=0')
for exp in sorted(os.scandir(expdir), key=lambda d: d.stat().st_mtime):
	exp = exp.name
	if any(x in exp for x in keywords):
		print('Experiment: %s' % exp)
		classdir = os.path.join(expdir, exp, 'classifier_exp')
		if os.path.exists(classdir):
			for classifier in os.listdir(classdir):
				res = os.path.join(classdir, classifier, 'results', 'accuracy.txt')
				with open(res, 'r') as pd:
					res = pd.read()

					# find result for accuracy window 0
					acc = (res.split('\t')[1]).split(' ')[-1]
					print('\t '+str(res.split('\t')[0])+'\t'+str(acc))
