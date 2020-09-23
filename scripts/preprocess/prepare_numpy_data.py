from __future__ import absolute_import
import os
import sys
print(sys.path)
print(os.getcwd())
sys.path.insert(0, os.getcwd())
import time
import argparse
import numpy as np
import librosa
import sox
from fhvae.datasets.audio_utils import *


# python scripts/preprocess/prepare_numpy_data.py datasets/timit_np_fbank/train/wav.scp datasets/timit_np_fbank/fbank datasets/timit_np_fbank/train/feat.scp datasets/timit_np_fbank/train/len.scp --ftype fbank

def _maybe_makedir(d):
    try:
        os.makedirs(d)
    except OSError:
        pass


def main(wav_scp, np_dir, feat_scp, len_scp, reader, mapper, arg_sr, speed_perturb):
    np_dir = os.path.abspath(np_dir)
    _maybe_makedir(np_dir)
    _maybe_makedir(os.path.dirname(feat_scp))
    _maybe_makedir(os.path.dirname(len_scp))

    if speed_perturb:
        tfm_slow = sox.Transformer()
        tfm_slow.speed(factor=0.9)

        tfm_fast = sox.Transformer()
        tfm_fast.speed(factor=1.1)

    stime = time.time()
    with open(wav_scp) as f, open(feat_scp, "w") as ff, open(len_scp, "w") as fl:
        for i, line in enumerate(f):
            seq, path = line.rstrip().split()
            y, _sr = reader(path, arg_sr)

            if speed_perturb:  # seq is e.g.: sp1.1-fv800161 or sp0.9-fv800161
                if seq.startswith('sp'):
                    factor = seq.split('-')[0][2:5]

                    if factor == '0.9':
                        y = tfm_slow.build_array(input_array=y, sample_rate_in=_sr)
                    elif factor == '1.1':
                        y = tfm_fast.build_array(input_array=y, sample_rate_in=_sr)
                    else:
                        msg = "Speed perturbation factor should be 0.9 or 1.1, but is %s" % factor
                        raise ValueError(msg)

            feat = mapper(y, _sr)
            # print path, seq, np.min(feat), np.max(feat)
            np_path = "%s/%s.npy" % (np_dir, seq)
            with open(np_path, "wb") as fnp:
                np.save(fnp, feat)
            ff.write("%s %s\n" % (seq, np_path))
            fl.write("%s %s\n" % (seq, len(feat)))

            if (i + 1) % 1000 == 0:
                print(("%s files, %.fs" % (i + 1, time.time() - stime)))

    print("processed total %s audio files; time elapsed = %.fs" % (i + 1, time.time() - stime))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("wav_scp", type=str, help="input wav scp file")
    parser.add_argument("np_dir", type=str, help="output directory for numpy matrices")
    parser.add_argument("feat_scp", type=str, help="output feats.scp file")
    parser.add_argument("len_scp", type=str, help="output len.scp file")
    parser.add_argument("--ftype", type=str, default="fbank", choices=["fbank", "spec"],
                        help="feature type to compute")
    parser.add_argument("--sr", type=int, default=None,
                        help="resample raw audio to specified value if not None (None=use original fs)")
    parser.add_argument("--win_t", type=float, default=0.025,
                        help="window size in second")
    parser.add_argument("--hop_t", type=float, default=0.010,
                        help="frame spacing in second")
    parser.add_argument("--n_mels", type=int, default=80,
                        help="number of filter banks if choosing fbank")
    parser.add_argument("--speed_perturb", type=str, default='false',
                        help="whether to use 0.9 and 1.1 speed perturbation")
    args = parser.parse_args()
    print(args)

    reader = lambda path, sr: librosa.load(path, sr, mono=True)
    if args.ftype == "fbank":
        mapper = lambda y, sr: np.transpose(to_melspec(
            y, sr, int(sr * args.win_t), args.hop_t, args.win_t, n_mels=args.n_mels))
    elif args.ftype == "spec":
        mapper = lambda y, sr: np.transpose(rstft(
            y, sr, int(sr * args.win_t), args.hop_t, args.win_t))

    if args.speed_perturb.lower() in ['yes', 'true']:
        speed_perturb = True
    else:
        speed_perturb = False

    main(args.wav_scp, args.np_dir, args.feat_scp, args.len_scp, reader, mapper, args.sr, speed_perturb)
