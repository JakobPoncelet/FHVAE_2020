#!/bin/bash

if [ $# -ne 3 ]; then
    echo "not all arguments are supplied, should be: expdir=... trainconfig=... classconfig=..."

# train
/esat/spchdisk/scratch/jponcele/anaconda3/envs/tf21/bin/python scripts/train/run_hs_train.py --expdir $1 --config $2

# test
/esat/spchdisk/scratch/jponcele/anaconda3/envs/tf21/bin/python scripts/test/run_eval.py --expdir $1 --save True

# classify
/esat/spchdisk/scratch/jponcele/anaconda3/envs/tf21/bin/python scripts/classifier/run_classifier.py --expdir $1 --config $3
