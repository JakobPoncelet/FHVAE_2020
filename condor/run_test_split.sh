#!/bin/bash
export PYTHONPATH=""
if [ $# -ne 1 ]; then
    echo "not all arguments are supplied, should be: expdir=..."
fi

echo 'expdir=' $1

echo "Creating test split"
/esat/spchdisk/scratch/jponcele/anaconda3/envs/tf21/bin/python split_test.py $1

echo "Testing..."

/esat/spchdisk/scratch/jponcele/anaconda3/envs/tf21/bin/python scripts/test/run_eval.py --expdir "$1"/test_split/split1 --save False

/esat/spchdisk/scratch/jponcele/anaconda3/envs/tf21/bin/python scripts/test/run_eval.py --expdir "$1"/test_split/split2 --save False

/esat/spchdisk/scratch/jponcele/anaconda3/envs/tf21/bin/python scripts/test/run_eval.py --expdir "$1"/test_split/split3 --save False

/esat/spchdisk/scratch/jponcele/anaconda3/envs/tf21/bin/python scripts/test/run_eval.py --expdir $1/test_split/split4 --save False

/esat/spchdisk/scratch/jponcele/anaconda3/envs/tf21/bin/python scripts/test/run_eval.py --expdir "$1"/test_split/split5 --save False

/esat/spchdisk/scratch/jponcele/anaconda3/envs/tf21/bin/python scripts/test/run_eval.py --expdir "$1"/test_split/split6 --save False

