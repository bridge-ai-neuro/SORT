"""
Functions to create a dataset for the Sequence Order Recall Task (SORT).
For a given input sequence, the user defines the excerpt length, segment length, and the bins that define the distance
between segments. They will also define a number of excerpts N and a set of segment distance bins of length
n_bins. Each excerpt will have 1 segment pair in each distance bin.
Therefore create_sort_samples() will output data for (N * n_bins) SORT evaluation trials.

**Sampling checks**
The create_sort_samples() function will verify that the excerpts are uniformly distributed across the input sequence,
and that the segment pairs are uniformly distributed within the excerpts.

In case the excerpt or segment sampling has additional conditions or restrictions (e.g. only use start indices that
coincide with a sentence boundary), the user can modify the sampling function to check the necessary conditions
and throw an error if the sampling does not meet them. The function will oversample (e.g. generate N + m trials,
where m is the number of extra samples), remove bad samples, then remove extra samples to return the correct number
of trials. See `create_sort_text_dataset.py` for an example of how to implement an additional sampling restriction.

**Definition of excerpt, segment, and distance bins**
Assume a discrete sequence X.
We will randomly sample N excerpts of length L_E from X, where each excerpt E is a subsequence within X.
For each excerpt E, we then sample a pair of non-overlapping segments S = {S1, S2}, each of length L_S. Each segment
is a subsequence of E, and therefore also X. We define start indices i and j such that:
    E[i] = S1[0]
    E[j] = S2[0]
    i < j
    i + L < j
The distance between segments D_s is measured between these start indices, D_s = j - i.
For each excerpt E, we will sample segment pairs with varying start indices and distances D_s. The segment distances
are sampled from a set of n_bins. e.g. 2 bins could be defined by bin edges [L_S, 100, 200].
"""
import numpy as np
import warnings

from sort.utils import sample_random_subsequence, check_sample_uniformity, sample_segments_offset_bounds


def sample_segments(n_per_bucket, permuted_answers, sample_fn, segment_len, excerpts, sample_kwargs):
    # Helper function to sample N segment pairs corresponding to N excerpts. This is set apart as a subfunction to
    # allow the user to modify the `sample_fn` input to impose any sampling constraints on the sequence. If the
    # sampling fails the constraint, it should return an Exception to be caught here.
    bad_samples = []
    these_samples = [None] * n_per_bucket
    these_segments = [[None, None]] * n_per_bucket
    these_seg_pos = np.zeros((n_per_bucket, 2), dtype=np.int32)
    for sample_ind in range(n_per_bucket):
        excerpt = excerpts[sample_ind]
        try:
            s0, s1 = sample_fn(excerpt, segment_len, **sample_kwargs)
        except Exception as err:
            print(f"{err}: Sample {sample_ind} couldn't get the right segment offset")
            bad_samples.append(sample_ind)
            continue
        # If the answer doesn't match the pre-allocated one, swap the segment orders
        if ((permuted_answers[sample_ind] == 0) and (s1 < s0)) or ((permuted_answers[sample_ind] == 1) and (s0 < s1)):
            s0, s1 = s1, s0
        these_seg_pos[sample_ind, :] = np.array([s0, s1])
        seg0 = excerpt[s0:s0 + segment_len]
        seg1 = excerpt[s1:s1 + segment_len]
        these_samples[sample_ind] = excerpt
        these_segments[sample_ind] = (seg0, seg1)

    return these_seg_pos, these_samples, these_segments, bad_samples


def create_sort_samples(input_sequence, n_per_bucket, excerpt_len, segment_len, segment_distance_bins,
                        extra_samples=10, seed=100, verbose=False):
    """
    :param input_sequence: numpy ndarray. Input sequence to be sampled. First dimension should be the sequence
                           timesteps, i.e. [T, dim1, dim2] for a 3-D sequence input where dim1 and dim2 are
                           some sequence features.
    :param n_per_bucket: int. Number of excerpts, which is also the number of samples per distance bucket
    :param excerpt_len: int. Length of the text excerpt that contains the two segments
    :param segment_len: int. Length of the two segments to put in order
    :param segment_distance_bins: array-like of ints. These are the bin edges for segment distances. Distance is
                                  computed between the first element of each segment, i.e. abs(seg0_inds[0] -
                                  seg1_inds[0]). Item in bin is lower_bound < item <= upper_bound.
    :param extra_samples: int, default 10. Number of extra samples to generate in case some samples do not pass the
                          desired criteria (here this is only falling within the appropriate distance bin).
    :param seed: int, default 100. Random numpy seed for data generation
    :param verbose: bool, default False.
    :return: Returns 6 items, most of which are dicts where the keys are the right edges of the segment distance
    bins. All data with N items has corresponding indices (e.g. excerpt index 0 corresponds to segment pair at index 0,
    answers at index 0, and segment pair positions at index 0).
        bucketed_samples: dict. Keys are the distance bins. Values are numpy ndarrays of the sequence excerpts,
                          with dimensionality [N, excerpt_len, dim1, ..., dimn]. The excerpts are the same across
                          bins -- it is simply repeated for ease of use.
        bucketed_segments: dict. Keys are the distance bins. Each bin contains N 2-element tuples representing the
                           segment pair corresponding to each excerpt. The first element in the tuple is the segment
                           to be presented first in the SORT evaluation, and second element is the segment to be
                           presented second. Each segment is a numpy ndarray of shape [segment_len, dim1, ..., dimn].
        answers: dict. Keys are the distance bins. Values are numpy ndarrays containing the index of the answer,
                 where 0 indicates segment1 comes first, and 1 indicates segment2 comes first.
        segment_positions: dict. Keys are the distance bins. Each bin contains N 2-element tuples. First element is
                           the excerpt index corresponding to the start of the segment to be presented first in the
                           SORT evaluation, i.e. segment position 25 means excerpt[25] == segment[0]
        excerpt_pos: array-like containing N values between 0 and 1. These are the normalized start indices for each
                     excerpt in the sequence, normalized by the total number of sequence timesteps T.
        args: dict. Contains dataset creation parameters (excerpt_len, segment_len, segment_distance_bins).
    """

    if (n_per_bucket + extra_samples) % 2 != 0:
        raise ValueError("Please make n_per_bucket + extra_samples an even number so the trials are counterbalanced.")
    np.random.seed(seed)

    # Sample excerpts from the sequence
    excerpts = []
    excerpt_pos = []  # will normalize to length of the book
    i = 0
    while True:
        for sample_ind in range(n_per_bucket + extra_samples):
            tmp, excerpt_ind = sample_random_subsequence(input_sequence, excerpt_len)
            excerpts.append(tmp)
            excerpt_pos.append(excerpt_ind / len(input_sequence))  # normalized to length of book
        is_uniform = check_sample_uniformity(excerpt_pos, distribution_scale=(len(input_sequence) - excerpt_len) /
                                             len(input_sequence), verbose=verbose)
        if is_uniform:
            break
        else:
            i += 1
            if i > 100:
                raise ValueError("Couldn't distribute excerpts uniformly")

    # Create array of segment offsets
    offsets = np.array(segment_distance_bins)
    left_offset, right_offset = offsets[:-1], offsets[1:]
    if left_offset[0] < segment_len:
        warnings.warn("WARNING: your segments will overlap! To prevent this, ensure that the smallest distance bin " +
                      "is at least the segment length.")

    # Pre-allocate answer array for permuting, ensuring exactly 50% is answer A
    ans_array = np.tile([0, 1], (n_per_bucket + extra_samples) // 2)

    sample_fn = sample_segments_offset_bounds
    sample_kwargs = {}

    skip_samples = set()
    bucketed_samples, bucketed_segments = {}, {}
    answers = {}
    segment_positions = {}
    seg_pos_all = []
    for oi, (l, r) in enumerate(zip(left_offset, right_offset)):
        print(f"Creating {n_per_bucket + extra_samples} samples with distance between {l}, {r}")
        sample_kwargs.update({'offset_bounds': (l, r)})
        # Permute the list of answers
        permuted_answers = np.random.permutation(ans_array)
        these_seg_pos, these_samples, these_segments, bad_samples = \
            sample_segments(n_per_bucket + extra_samples, permuted_answers, sample_fn, segment_len,
                            excerpts, sample_kwargs)
        if verbose:
            tmp = these_seg_pos.copy()
            tmp = np.delete(tmp, bad_samples, axis=0)
            diffs = abs(np.diff(tmp))
            print(np.mean(diffs), np.min(diffs), np.max(diffs))
        skip_samples.update(set(bad_samples))
        seg_pos_all.append(these_seg_pos[:, 0])
        # Save into dictionaries
        dist_bucket = r
        bucketed_samples[dist_bucket] = these_samples
        bucketed_segments[dist_bucket] = these_segments
        answers[dist_bucket] = permuted_answers
        segment_positions[dist_bucket] = these_seg_pos
        if len(bad_samples) > 0:
            print(f"{len(bad_samples)} bad samples for the {dist_bucket} bucket")
        del permuted_answers

    if len(skip_samples) > 0:
        drop_more = extra_samples - len(skip_samples)
        more_inds = np.random.choice(list(skip_samples ^ set(list(range(n_per_bucket + extra_samples)))), drop_more, replace=False)
        drop_inds = list(skip_samples) + more_inds.tolist()
    else:
        drop_inds = np.random.choice(list(range(n_per_bucket + extra_samples)), extra_samples, replace=False)
    seg_pos_all = []
    excerpt_pos = np.delete(excerpt_pos, drop_inds, axis=0)
    for b in segment_distance_bins[1:]:
        bucketed_samples[b] = np.delete(bucketed_samples[b], drop_inds, axis=0)
        bucketed_segments[b] = np.delete(bucketed_segments[b], drop_inds, axis=0)
        answers[b] = np.delete(answers[b], drop_inds, axis=0)
        segment_positions[b] = np.delete(segment_positions[b], drop_inds, axis=0)
        seg_pos_all.append(segment_positions[b][:, 0])
        assert len(answers[b]) == n_per_bucket, \
            f"Sample winnowing gave an unexpected length {len(answers[b])} instead of {n_per_bucket}"
    seg_pos = np.concatenate(seg_pos_all)
    is_uniform = check_sample_uniformity(seg_pos, distribution_scale=excerpt_len - segment_len, verbose=verbose)
    if not is_uniform:
        raise ValueError("Segment positions are not uniform!")

    args = {'excerpt_len': excerpt_len, 'segment_len': segment_len, 'distance_bins': segment_distance_bins}

    return bucketed_samples, bucketed_segments, answers, segment_positions, excerpt_pos, args

