import sys
import tensorflow as tf

def seq_to_seg_mapper(key, clean_key, feats, lens, labs, talabs, seg_len, seg_shift, rand_seg, num_talabs, num_noisy_vers, compute_labs=True):

    # map to index of clean segment for seed generation
    if num_noisy_vers > 0:
        key = key // num_noisy_vers

    nsegs = tf.math.floordiv((lens - seg_len), seg_shift) + 1
    if rand_seg:
        # seg_starts = tf.random.uniform([nsegs], minval=0, maxval=lens-seg_len+1, dtype=tf.int64)
        seg_starts = tf.random.stateless_uniform([nsegs], seed=[key, clean_key], minval=0, maxval=lens-seg_len+1, dtype=tf.int64)

    else:
        seg_starts = tf.range(nsegs)*seg_shift

    ends = seg_starts + seg_len
    centers = tf.math.floordiv(seg_starts + ends, 2)

    def get_centered_talab(center, talabs, i):
        # talabstarts = talabs[i, 0, :]
        # idx = tf.math.argmin(tf.math.abs(talabstarts-center))-1
        diffs = center - talabs[i, 0, :]
        idx = tf.math.argmin(tf.where(tf.math.greater_equal(diffs, 0), diffs, diffs+999))
        center_talab = talabs[i, 2, idx]
        return center_talab

    if compute_labs:
        seg_talabs = []
        for i in range(0, num_talabs):
            true_talabs = tf.map_fn(lambda x : get_centered_talab(x, talabs, i), centers, \
                                parallel_iterations=10, back_prop=False)
            seg_talabs.append(true_talabs)
        seg_talabs = tf.stack(seg_talabs, axis=1)

    def get_seg_feats(feats, start, seg_len):
        seg_feats = tf.slice(feats, [start, 0], [seg_len, -1])
        return seg_feats

    seg_feats = tf.map_fn(lambda x : get_seg_feats(feats, x, seg_len), seg_starts, \
                          dtype=tf.float32, parallel_iterations=10, back_prop=False, infer_shape=False)

    nsegs = tf.expand_dims(nsegs, 0)
    seg_keys = tf.tile(tf.expand_dims(key, 0), nsegs)
    seg_nsegs = tf.tile(nsegs, nsegs)

    if compute_labs:
        labs = tf.expand_dims(labs, axis=0)
        seg_labs = tf.tile(labs, tf.concat([nsegs, tf.ones_like(nsegs)], axis=0))

    if not compute_labs:
        seg_labs = tf.zeros_like(seg_starts)
        seg_talabs = tf.zeros_like(seg_starts)

    return seg_keys, seg_feats, seg_nsegs, seg_labs, seg_talabs, seg_starts
