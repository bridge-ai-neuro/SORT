"""
Functions to create a text dataset for the Sequence Order Recall Task (SORT). This can be used to recreate the
BookSORT dataset.

Assumes that the input is an array-like sequence of words. The words can be joined with a space separator to form a
continuous string of text. The excerpts and segments are sampled such that they always begin at a new sentence
boundary. For example if the excerpt is ["She", "rises.", "They" "rise."] then segments can only begin at "She" or
"They", and not "rises." or "rise.".

Read the top of sort/create_sort_dataset.py for more details on the data sampling, term definitions, etc.
"""
import numpy as np

from sort.dataset_creation.utils import check_sample_uniformity
from sort.dataset_creation.text_utils import sample_random_text, sample_segments_sentence_start_bins


def sample_segments(n_per_bucket, permuted_answers, sample_fn, segment_len, excerpts, sample_kwargs):
    # Helper function to sample N segment pairs corresponding to N excerpts. The `sample_fn` only returns segments
    # that begin at sentence boundaries, and will raise an error if sampling fails. The `sample_fn` also returns the
    # indices corresponding to the sentence boundaries so we can maintain a distribution of all possible sentence
    # boundaries to sample from. This will allow us to check whether this distribution is sampled uniformly.
    # This helper function also joins the string arrays to form a continuous string for each excerpt and segment.
    bad_samples = []
    these_samples = [None] * n_per_bucket
    these_segments = [[None, None]] * n_per_bucket
    these_seg_pos = np.zeros((n_per_bucket, 2), dtype=np.int32)
    all_sent_starts = []
    for sample_ind in range(n_per_bucket):
        excerpt = excerpts[sample_ind]
        output_text = " ".join(excerpt)
        try:
            s0, s1, sent_starts = sample_fn(excerpt, segment_len, **sample_kwargs)
            all_sent_starts.append(sent_starts)
        except Exception as err:
            print(f"{err}: Sample {sample_ind} couldn't get the right segment offset")
            bad_samples.append(sample_ind)
            continue
        # If the answer doesn't match the pre-allocated one, swap the segment orders
        if ((permuted_answers[sample_ind] == 0) and (s1 < s0)) or ((permuted_answers[sample_ind] == 1) and (s0 < s1)):
            s0, s1 = s1, s0
        these_seg_pos[sample_ind, :] = np.array([s0, s1])
        seg0 = " ".join(excerpt[s0:s0 + segment_len])
        seg1 = " ".join(excerpt[s1:s1 + segment_len])
        these_samples[sample_ind] = output_text
        these_segments[sample_ind] = (seg0, seg1)
    all_sent_starts = np.concatenate(all_sent_starts)

    return these_seg_pos, these_samples, these_segments, bad_samples, all_sent_starts


def create_sort_samples(input_sequence, n_per_bucket, excerpt_len, segment_len, segment_distance_bins,
                        extra_samples=10, seed=100, verbose=False, enforce_sentence_bounds=(True, False)):
    """
    Modification of sort.create_sort_dataset.create_sort_samples().

    :param input_sequence: array-like. Sequence of words. which can be joined by a whitespace " " separator to form a
                           continuous string
    :param n_per_bucket: int. Number of excerpts, which is also the number of samples per distance bucket
    :param excerpt_len: int. Length of the text excerpt that contains the two segments
    :param segment_len: int. Length of the two segments to put in order
    :param segment_distance_bins: array-like of ints. These are the bin edges for segment distances. Distance is
                                  computed between the first element of each segment, i.e. abs(seg0_inds[0] -
                                  seg1_inds[0]). Item in bin is lower_bound < item <= upper_bound.
    :param extra_samples: int, default 10. Number of extra samples to generate in case some samples do not pass the
                          desired criteria (here this is only falling within the appropriate distance bin).
    :param seed: int, default 100. Random numpy seed for data generation
    :param verbose: bool, default False
    :param enforce_sentence_bounds: tuple of (start_at_sentence, end_at_sentence), default (True, False). If True,
                                        forces the sample to fall at sentence boundaries. Default only forces beginning
                                        of sample to start at a sentence boundary, but it can end before a sentence
                                        boundary.
    :return: Returns 6 items, most of which are dicts where the keys are the right edges of the segment distance
    bins. All data with N items has corresponding indices (e.g. excerpt index 0 corresponds to segment pair at index 0,
    answers at index 0, and segment pair positions at index 0).
        bucketed_samples: dict. Keys are the distance bins. Values are the excerpts, given as continuous strings.
                          The excerpts are the same across bins -- it is simply repeated for ease of use.
        bucketed_segments: dict. Keys are the distance bins. Each bin contains N 2-element tuples representing the
                           segment pair corresponding to each excerpt. The first element in the tuple is the segment
                           to be presented first in the SORT evaluation, and second element is the segment to be
                           presented second. Each segment is a string.
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
    i = 0
    while True:
        excerpts = []
        excerpt_pos = []  # will normalize to length of the book

        for sample_ind in range(n_per_bucket + extra_samples):
            tmp, excerpt_ind = sample_random_text(input_sequence, excerpt_len, enforce_sentence_bounds=enforce_sentence_bounds)
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

    # Pre-allocate answer array for permuting, ensuring exactly 50% is answer A
    ans_array = np.tile([0, 1], (n_per_bucket + extra_samples) // 2)

    sample_fn = sample_segments_sentence_start_bins
    sample_kwargs = {}

    skip_samples = set()
    bucketed_samples, bucketed_segments = {}, {}
    answers = {}
    segment_positions = {}
    seg_pos_all = []
    sent_starts_all = []
    for oi, (l, r) in enumerate(zip(left_offset, right_offset)):
        print(f"Creating {n_per_bucket + extra_samples} samples with distance between {l}, {r}")
        sample_kwargs.update({'offset_bounds': (l, r)})
        # Permute the list of answers
        permuted_answers = np.random.permutation(ans_array)
        these_seg_pos, these_samples, these_segments, bad_samples, sent_starts = \
            sample_segments(n_per_bucket + extra_samples, permuted_answers, sample_fn, segment_len,
                            excerpts, sample_kwargs)
        if verbose:
            tmp = these_seg_pos.copy()
            tmp = np.delete(tmp, bad_samples, axis=0)
            diffs = abs(np.diff(tmp))
            print(np.mean(diffs), np.min(diffs), np.max(diffs))
        skip_samples.update(set(bad_samples))
        seg_pos_all.append(these_seg_pos[:, 0])
        sent_starts_all.append(sent_starts)
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
        if drop_more < 0:
            raise ValueError(f"Increase extra_samples or try a different random seed. You had {len(skip_samples)} bad "
                             "samples so you need at least that number.")
        more_inds = np.random.choice(list(skip_samples ^ set(list(range(n_per_bucket + extra_samples)))), drop_more,
                                     replace=False)
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
    sent_starts = np.concatenate(sent_starts_all)
    is_uniform = check_sample_uniformity(seg_pos, distribution=sent_starts, verbose=verbose)
    if not is_uniform:
        raise ValueError("Segment positions are not uniform!")

    args = {'excerpt_len': excerpt_len, 'segment_len': segment_len, 'distance_bins': segment_distance_bins}

    return bucketed_samples, bucketed_segments, answers, segment_positions, excerpt_pos, args

