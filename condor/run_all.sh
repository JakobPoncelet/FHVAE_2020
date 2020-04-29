#!/bin/bash
export PYTHONPATH=""
if [ $# -ne 3 ]; then
    echo "not all arguments are supplied, should be: expdir=... trainconfig=... classconfig=..."
fi

echo 'expdir=' $1
echo 'trainconfig=' $2
echo 'classconfig=' $3

echo "Training..."
# train
/esat/spchdisk/scratch/jponcele/anaconda3/envs/tf21/bin/python scripts/train/run_hs_train.py --expdir $1 --config $2

echo "Testing..."
# test
/esat/spchdisk/scratch/jponcele/anaconda3/envs/tf21/bin/python scripts/test/run_eval.py --expdir $1 --save True

echo "Building classifier..."
# classify
/esat/spchdisk/scratch/jponcele/anaconda3/envs/tf21/bin/python scripts/classifier/run_classifier.py --expdir $1 --config $3


