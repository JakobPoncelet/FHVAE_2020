import os
import sys
import matplotlib
import matplotlib.pyplot as plt

## USAGE: python plot_losses.py expdir

if len(sys.argv) != 2:
	print('Wrong arguments! Usage: python plot_losses.py expdir')
	exit(1)

expdir = str(sys.argv[1])

if not os.path.exists(expdir):
	print('Expdir doesnt exist!')
	exit(1)

print('Retrieving losses of %s' % expdir)

lossdir = os.path.join(expdir, 'losses') 
os.makedirs(os.path.join(lossdir, 'plots'), exist_ok=True)

# loss components
lb = []
log_qy_mu2 = []
log_qy_mu1 = []
log_b = []
log_c = []

# lower bound components
log_pmu2 = []
log_pmu1 = []
neg_kld_z2 = []
neg_kld_z1 = []
log_px_z = []

with open(os.path.join(lossdir, 'result_mean_loss.txt'), 'r') as f:
	for i, l in enumerate(f):
    		pass
	num_epochs = i+1

with open(os.path.join(lossdir, 'result_comp_loss.txt'), 'r') as pd:
	line = pd.readline()
	while line:
		if line.split(' ')[0] == 'EPOCH:':
			epoch = int(line.split(' ')[1].rstrip())

			line = pd.readline()
			lb.append(float((line.split('\t')[2].rstrip()).split('=')[1]))
			log_qy_mu2.append(float((line.split('\t')[3].rstrip()).split('=')[1]))
			log_qy_mu1.append(float((line.split('\t')[4].rstrip()).split('=')[1]))
			log_b.append(float((line.split('\t')[5].rstrip()).split('=')[1]))
			log_c.append(float((line.split('\t')[6].rstrip()).split('=')[1]))

			if (epoch == 1) or (epoch == num_epochs):
				print('epoch ', epoch,' \n', line)

			line = pd.readline()
			log_pmu2.append(float((line.split('\t')[2].rstrip()).split('=')[1]))
			log_pmu1.append(float((line.split('\t')[3].rstrip()).split('=')[1]))
			neg_kld_z2.append(float((line.split('\t')[4].rstrip()).split('=')[1]))
			neg_kld_z1.append(float((line.split('\t')[5].rstrip()).split('=')[1]))
			log_px_z.append(float((line.split('\t')[6].rstrip()).split('=')[1]))

			if (epoch == 1) or (epoch == num_epochs):
				print('epoch ', epoch,' \n', line)

		line = pd.readline()

plt.figure(1)
plt.plot(lb, label='lower bound (lb)')
plt.plot(log_qy_mu1, label='discr loss mu1 (log_qy_mu1)')
plt.plot(log_qy_mu2, label='discr loss mu2 (log_qy_mu2)')
plt.plot(log_b, label='reg loss z1 (log_b)')
plt.plot(log_c, label='reg loss z2 (log_c)')
plt.ylabel('Training Loss')
plt.xlabel('Epoch')
plt.title('Loss Components of 1 Batch')
plt.legend()
plt.savefig(os.path.join(lossdir, 'plots', 'Total_Loss.pdf'), format='pdf')
plt.show()


plt.figure(2)
plt.plot(log_qy_mu1, label='discr loss mu1 (log_qy_mu1)')
plt.plot(log_qy_mu2, label='discr loss mu2 (log_qy_mu2)')
plt.plot(log_b, label='reg loss z1 (log_b)')
plt.plot(log_c, label='reg loss z2 (log_c)')
plt.ylabel('Training Loss')
plt.xlabel('Epoch')
plt.title('Loss Components of 1 Batch Without Lower Bound')
plt.legend()
plt.savefig(os.path.join(lossdir, 'plots', 'Total_Loss_wo_lb.pdf'), format='pdf')
plt.show()


plt.figure(3)
plt.plot(log_pmu1, label='prior mu1 (log_pmu1)')
plt.plot(log_pmu2, label='prior mu2 (log_pmu2)')
plt.plot(neg_kld_z1, label='z1 KLD (neg_kld_z1)')
plt.plot(neg_kld_z2, label='z2 KLD (neg_kld_z2)')
plt.plot(log_px_z, label='reconstr loss (log_px_z)')
plt.ylabel('Training Loss')
plt.xlabel('Epoch')
plt.title('Lower Bound Components of 1 Batch')
plt.legend()
plt.savefig(os.path.join(lossdir, 'plots', 'Lower_Bound_Loss.pdf'), format='pdf')
plt.show()


plt.figure(4)
plt.plot(log_pmu1, label='prior mu1 (log_pmu1)')
plt.plot(log_pmu2, label='prior mu2 (log_pmu2)')
plt.plot(neg_kld_z1, label='z1 KLD (neg_kld_z1)')
plt.plot(neg_kld_z2, label='z2 KLD (neg_kld_z2)')
plt.ylabel('Training Loss')
plt.xlabel('Epoch')
plt.title('Lower Bound Components of 1 Batch Without Reconstruction Loss')
plt.legend()
plt.savefig(os.path.join(lossdir, 'plots', 'Lower_Bound_Loss_wo_rec.pdf'), format='pdf')
plt.show()

