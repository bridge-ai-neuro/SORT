import numpy as np
import pytest
from sort.create_sort_dataset import create_sort_samples as create_dataset


# FOR NUMERIC INPUT TESTS
seq = np.random.randint(0, 255, (1000,))


def input_output_assertions(sequence, N, sl, el, bins):
    d = create_dataset(input_sequence=sequence, n_per_bucket=N, excerpt_len=el,
                       segment_len=sl, segment_distance_bins=bins)
    # TEST THAT OUTPUTS ARE THE EXPECTED DIMENSIONS
    assert len(d) == 6, "Number of outputs from dataset creation is wrong"
    # unpack responses
    excerpts, segments, answers, segment_positions, excerpt_pos, args = d
    bin_keys = list(excerpts.keys())
    assert len(bin_keys) == len(bins) - 1, "Unexpected bins"
    assert all([x == y for x, y in zip(bin_keys, list(segments.keys()))]), "Unexpected data keys to segs / excerpts"
    ex = excerpts[bins[1]]
    assert len(ex) == N, "Number of excerpts do not equal n_per_bucket as expected"
    if len(bins) > 2:
        for b in bins[2:]:
            # Excerpts should be the same across bins
            assert np.all(ex == excerpts[b]), "Excerpts not identical across distance bins"
    b = bins[1]
    s0, spos0 = segments[b][0], segment_positions[b][0]
    assert len(s0) == 2 and len(spos0) == 2, "Should have a pair of segments!"
    # TEST THAT SEGMENT POSITIONS ARE CORRECT
    assert np.all(ex[0][spos0[0]:spos0[0] + sl] == s0[0]), "Segment 1 is unexpected"
    assert np.all(ex[0][spos0[1]:spos0[1] + sl] == s0[1]), "Segment 2 is unexpected"


def test_numeric_input_output():
    input_output_assertions(seq, N=10, sl=5, el=100, bins=[5, 10, 25, 50])


def test_2d_numeric_input_output():
    dim2 = 25
    seq2d = np.random.randint(0, 255, (1000, dim2))
    N = 10
    sl = 5
    el = 100
    bins = [5, 10, 25, 50]

    d = create_dataset(input_sequence=seq2d, n_per_bucket=N, excerpt_len=el,
                       segment_len=sl, segment_distance_bins=bins)
    # TEST THAT OUTPUTS ARE THE EXPECTED DIMENSIONS
    assert len(d) == 6, "Number of outputs from dataset creation is wrong"
    # unpack responses
    excerpts, segments, answers, segment_positions, excerpt_pos, args = d
    bin_keys = list(excerpts.keys())
    assert len(bin_keys) == len(bins) - 1, "Unexpected bins"
    assert all([x == y for x, y in zip(bin_keys, list(segments.keys()))]), "Unexpected data keys to segs / excerpts"
    ex = excerpts[bins[1]]
    assert len(ex) == N, "Number of excerpts do not equal n_per_bucket as expected"
    assert ex.shape[-1] == dim2, "2-d sequence excerpt has unexpected 2nd dimension"
    if len(bins) > 2:
        for b in bins[2:]:
            # Excerpts should be the same across bins
            assert np.all(ex == excerpts[b]), "Excerpts not identical across distance bins"
    b = bins[1]
    s0, spos0 = segments[b][0], segment_positions[b][0]
    assert s0.shape[-1] == dim2, "2-d sequence segment has unexpected 2nd dimension"
    assert len(s0) == 2 and len(spos0) == 2, "Should have a pair of segments!"
    # TEST THAT SEGMENT POSITIONS ARE CORRECT
    assert np.all(ex[0][spos0[0]:spos0[0] + sl] == s0[0]), "Segment 1 is unexpected"
    assert np.all(ex[0][spos0[1]:spos0[1] + sl] == s0[1]), "Segment 2 is unexpected"


def test_str_input_output():
    str_seq = "".join([chr(x) for x in seq])  # convert to ascii
    input_output_assertions(str_seq, N=10, sl=5, el=100, bins=[5, 10, 25, 50])


def test_number_trials():
    # Tests that an exception is raised when the number of trials is odd (conditions will not be counterbalanced)
    with pytest.raises(ValueError):
        _ = create_dataset(input_sequence=seq, n_per_bucket=10, excerpt_len=100,
                           segment_len=5, segment_distance_bins=[5, 10, 25], extra_samples=5)
    with pytest.raises(ValueError):
        _ = create_dataset(input_sequence=seq, n_per_bucket=15, excerpt_len=100,
                           segment_len=5, segment_distance_bins=[5, 10, 25])
    _ = create_dataset(input_sequence=seq, n_per_bucket=15, excerpt_len=100,
                       segment_len=5, segment_distance_bins=[5, 10, 25], extra_samples=5)


def test_impossible_bounds():
    with pytest.raises(Exception):
        _ = create_dataset(input_sequence=seq, n_per_bucket=8, excerpt_len=10,
                           segment_len=5, segment_distance_bins=[10, 25])


def test_excerpt_uniformity():
    # Test ability of code to detect when excerpts are not uniformly distributed
    seq = np.random.randint(0, 255, (10,))
    with pytest.raises(ValueError) as e:
        d = create_dataset(input_sequence=seq, n_per_bucket=4, excerpt_len=8,
                           segment_len=2, segment_distance_bins=[2, 5])
    assert str(e.value) == "Couldn't distribute excerpts uniformly"


def test_sample_uniformity():
    # Test ability of code to detect when segments are not uniformly distributed in bin
    with pytest.raises(ValueError) as e:
        d = create_dataset(input_sequence=seq, n_per_bucket=50, excerpt_len=15,
                           segment_len=5, segment_distance_bins=[5, 15])
    assert str(e.value) == "Segment positions are not uniform!"


def test_difficult_bounds_resampling():
    # Test ability of code to meet difficult constraints
    # We expect at least one printed message about resampling when running this.
    input_output_assertions(seq, N=4, sl=5, el=15, bins=[5, 7])


def test_segment_overlap():
    sl = 5
    with pytest.warns(Warning):
        # Should raise a warning when the segment distance bin is inappropriate
        _ = create_dataset(input_sequence=seq, n_per_bucket=10, excerpt_len=100,
                           segment_len=sl, segment_distance_bins=[0, 5, 10])
    # Test that generated segments will not actually overlap
    d = create_dataset(input_sequence=seq, n_per_bucket=4, excerpt_len=100,
                       segment_len=sl, segment_distance_bins=[5, 6])
    excerpts, segments, answers, segment_positions, excerpt_pos, args = d
    seg_pairs = segment_positions[6]
    for sp in seg_pairs:
        sp1, sp2 = sp.min(), sp.max()
        sd = sp2 - sp1
        assert sp1 + sl < sp2, "Segment positions don't pass i + L < j"
        assert (sd >= 5) and (sd <= 6), "Segment distance doesn't fall in expected bin"
