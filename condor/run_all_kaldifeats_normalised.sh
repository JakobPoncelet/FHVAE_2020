#!/bin/bash
export PYTHONPATH=""
if [ $# -ne 1 ]; then
    echo "not all arguments are supplied, should be: expdir=..."
fi

echo 'expdir=' $1

/esat/spchdisk/scratch/jponcele/anaconda3/envs/tf21/bin/python scripts/save_to_kaldi/run_extract_lvar.py --expdir $1 --datadir cgn_kaldi_feats_normalised --segments ./misc/cgn/vl_without_a.segments --suffix train

/esat/spchdisk/scratch/jponcele/anaconda3/envs/tf21/bin/python scripts/save_to_kaldi/run_extract_lvar.py --expdir $1 --datadir cgn_kaldi_feats_normalised --segments ./misc/cgn/vl_without_a_dev.segments --suffix dev

/esat/spchdisk/scratch/jponcele/anaconda3/envs/tf21/bin/python scripts/save_to_kaldi/run_extract_lvar.py --expdir $1 --datadir nbest_kaldi_feats_normalised --segments ./misc/cgn/nbest.segments --suffix test

/esat/spchdisk/scratch/jponcele/anaconda3/envs/tf21/bin/python scripts/save_to_kaldi/run_extract_lvar.py --expdir $1 --datadir nbest_augmented_kaldi_feats_normalised --segments ./misc/cgn/nbest_augmented.segments --suffix test-augmented
