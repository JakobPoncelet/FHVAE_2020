#!/bin/bash
export PYTHONPATH=""
if [ $# -ne 1 ]; then
    echo "not all arguments are supplied, should be: expdir=..."
fi

echo 'expdir=' $1

/esat/spchdisk/scratch/jponcele/anaconda3/envs/tf21/bin/python scripts/save_to_kaldi/run_extract_lvar.py --expdir $1 --datadir cgn_kaldi_feats --segments ./misc/cgn/train_s_WB.segments --suffix train_s

/esat/spchdisk/scratch/jponcele/anaconda3/envs/tf21/bin/python scripts/save_to_kaldi/run_extract_lvar.py --expdir $1 --datadir cgn_tel_kaldi_feats --segments ./misc/cgn/train_t.segments --suffix train_t

/esat/spchdisk/scratch/jponcele/anaconda3/envs/tf21/bin/python scripts/save_to_kaldi/run_extract_lvar.py --expdir $1 --datadir cgn_kaldi_feats --segments ./misc/cgn/dev_s.segments --suffix dev_s

/esat/spchdisk/scratch/jponcele/anaconda3/envs/tf21/bin/python scripts/save_to_kaldi/run_extract_lvar.py --expdir $1 --datadir cgn_tel_kaldi_feats --segments ./misc/cgn/dev_t.segments --suffix dev_t

/esat/spchdisk/scratch/jponcele/anaconda3/envs/tf21/bin/python scripts/save_to_kaldi/run_extract_lvar.py --expdir $1 --datadir nbest_kaldi_feats --segments ./misc/cgn/nbest.segments --suffix test

/esat/spchdisk/scratch/jponcele/anaconda3/envs/tf21/bin/python scripts/save_to_kaldi/run_extract_lvar.py --expdir $1 --datadir nbest_augmented_kaldi_feats --segments ./misc/cgn/nbest_augmented.segments --suffix test-augmented

/esat/spchdisk/scratch/jponcele/anaconda3/envs/tf21/bin/python scripts/save_to_kaldi/run_extract_lvar.py --expdir $1 --datadir nbest_cts_kaldi_feats --segments ./misc/cgn/nbest_cts.segments --suffix test-cts
