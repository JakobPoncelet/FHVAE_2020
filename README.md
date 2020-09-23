# FHVAE_TF2
## Implementation
Implementation of Extended Regularised Scalable Factorised Hierarchical Variational Autoencoder in Tensorflow 2.1 and Python 3.6.8. The code is partly based on the original from https://github.com/wnhsu/ScalableFHVAE.


## Model
It is possible to use all LSTM models, or use a Transformer for the encoder (still quite experimental, results not convincing yet). For LSTM, one can choose between unidirectional LSTMs, bidirectional LSTMs and bidirectional LSTMs with pyramidal LSTM+attention for Z2 encoder.
 
Regularizations have been added to Z1 and to Z2, with a cross-entropy loss based on classification into labels. For now, sequence factors like gender, region and speaker recognition are added on Z2, and segmentspecific time-aligned labels ('talabs') like phones and phoneclass recognition are added on Z1. Wav-files without labels are also allowed (for training with unsupervised data).

The latent variable Z2 is related to a sequence-specific prior, the latent variable Z1 is related to a (segment)phone-specific prior. 

After training, during testing the latent variables are written out. A NN classifier is then trained on the Z1 of train set to classify the phones from the Z1 of every segment.


## Supported datasets
The TIMIT database can be fully replicated using the example script. 
CGN also has a preparation script (of which some part has to run in Matlab).


## How to run
It is advised to start with TIMIT and go through the code step by step.

1) Preprocessing examples --> prepare_timit  (this will set up your database and create all necessary files)

2) Change the template config.cfg file to your needs or use one in the configs-directory.

Now to run locally:

3) Scripts/train --> python run_hs_train.py --expdir=... --config=...

4) Scripts/test --> python run_eval.py --expdir=... --save=...

5) Scripts/classifier --> python run_classifier.py --expdir=... --config=...


## Running training on Condor
The expdir has to exist already! Look at the default config-files.

Options: 

    condor_submit condor/jobfile_train.job expdir=... config=...
    
    condor_submit condor/jobfile_test.job expdir=... save=True/False
    
    condor_submit condor/jobfile_classifier.job expdir=... config=..

Or do training/testing/classification sequentially with one jobfile. This will call run_all.sh, make sure it is executable (permissions!):

     condor_submit condor/jobfile_all.job expdir=...  trainconfig=... classconfig=...


## Contact
jakob.poncelet[at]esat.kuleuven.be


## References
Hsu, W. N., Zhang, Y., and Glass, J. Unsupervised learning of disentangled and interpretable representations from sequential data. In NIPS, 2017.

Hsu, W. N. and  Glass, J. Scalable  factorized  hierarchical  variational autoencoder training. In Interspeech, 2018.

Van hamme, H. Extension of Factorized Hierarchical Variational
Auto-encoders with Phoneme Classes. 2020
