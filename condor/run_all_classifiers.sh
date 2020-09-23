#!/bin/bash
export PYTHONPATH=""
if [ $# -ne 2 ]; then
    echo "not all arguments are supplied, should be: expdir=... config=..."
fi

echo 'expdir=' $1
echo 'config=' $2

echo "Building classifier..."
# classify
/esat/spchdisk/scratch/jponcele/anaconda3/envs/tf21/bin/python scripts/classifier/run_classifier.py --expdir $1 --config $2 --var z1

/esat/spchdisk/scratch/jponcele/anaconda3/envs/tf21/bin/python scripts/classifier/run_classifier.py --expdir $1 --config $2 --var z2

/esat/spchdisk/scratch/jponcele/anaconda3/envs/tf21/bin/python scripts/classifier/run_classifier.py --expdir $1 --config $2 --var z1_z2


