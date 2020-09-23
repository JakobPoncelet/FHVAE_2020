#!/bin/bash
export PYTHONPATH=""
source /esat/spchdisk/scratch/jponcele/anaconda3/bin/activate tf21
python --version

if [ $# -ne 1 ]; then
    echo "not all arguments are supplied, should be: expdir=..."
fi

echo 'expdir=' $1

python scripts/save_to_kaldi/run_extract_lvar.py --expdir $1 --datadir cgn_kaldi_feats --segments ./misc/cgn/vl_without_a.segments --suffix train

python scripts/save_to_kaldi/run_extract_lvar.py --expdir $1 --datadir cgn_kaldi_feats --segments ./misc/cgn/vl_without_a_dev.segments --suffix dev

python scripts/save_to_kaldi/run_extract_lvar.py --expdir $1 --datadir nbest_kaldi_feats --segments ./misc/cgn/nbest.segments --suffix test

python scripts/save_to_kaldi/run_extract_lvar.py --expdir $1 --datadir nbest_augmented_kaldi_feats --segments ./misc/cgn/nbest_augmented.segments --suffix test-augmented
