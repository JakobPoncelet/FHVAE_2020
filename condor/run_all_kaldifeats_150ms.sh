#!/bin/bash
export PYTHONPATH=""
if [ $# -ne 1 ]; then
    echo "not all arguments are supplied, should be: expdir=..."
fi

echo 'expdir=' $1

/esat/spchdisk/scratch/jponcele/anaconda3/envs/tf21/bin/python scripts/save_to_kaldi/run_extract_lvar.py --expdir $1 --datadir cgn_kaldi_feats --segments ./misc/cgn/vl_without_a.segments --suffix train --seg_len 15

/esat/spchdisk/scratch/jponcele/anaconda3/envs/tf21/bin/python scripts/save_to_kaldi/run_extract_lvar.py --expdir $1 --datadir cgn_kaldi_feats --segments ./misc/cgn/vl_without_a_dev.segments --suffix dev --seg_len 15

/esat/spchdisk/scratch/jponcele/anaconda3/envs/tf21/bin/python scripts/save_to_kaldi/run_extract_lvar.py --expdir $1 --datadir nbest_kaldi_feats --segments ./misc/cgn/nbest.segments --suffix test --seg_len 15


