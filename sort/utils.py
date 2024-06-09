import numpy as np
from scipy import stats


def get_models_plot_info():
    it_str = '-inst'  # instruction tuned suffix for model names.

    models_plot_info = {
        "mistral-instruct-7b": {
            "name": "mistral-7b-v1",
            "label": "Mistral-v1-7b" + it_str,
            "color": "cornflowerblue"
        },
        "mistral-instruct-7b-v2": {
            "name": "mistral-7b-instruct-v2",
            "label": "Mistral-v2-7b" + it_str,
            "color": "teal"
        },
        "llama3-8b-instruct":  {
            "name": "llama3-8b",
            "label":  "Llama3-8b"+ it_str,
            "color": "darkorange"
        },
        "gemma7b_1.1_inst": {
            "name": "gemma-7b-1.1",
            "label": "Gemma-1.1-7b" + it_str,
            "color": "red"
        },
        "Nous-Hermes-2-Mixtral-8x7B-DPO": {
            "name": "mistral_8x7b_dpo",
            "label": "Mixtral-8x7b-DPO" + it_str,
            "color": "darkgoldenrod"
        },
        "Mixtral-8x22b": {
            "name": "mixtral-8x22",
            "label": "Mixtral-8x22b" + it_str,
            "color": "deepskyblue"},  # fixed
        "llama2_7b-instruct": {
            "name": "llama2-7b",
            "label": "Llama2-7b" + it_str,
            "color": "darkviolet"
        },
        "llama2_70b-instruct": {
            "name": "llama2-70b",
            "label": "Llama2-70b" + it_str,
            "color": "magenta"
        },
        "llama3_70b-instruct": {
            "name": "llama3-70b",
            "label": "Llama3-70b" + it_str,
            "color": "hotpink"
        },
        "gpt3-5": {
            "name": "gpt-3.5",
            "label": "GPT-3.5-turbo",
            "color": "brown"
        },
        "gpt4": {
            "name": "gpt-4",
            "label": "GPT-4",
            "color": "olive"
        },
    }
    return models_plot_info


def sample_random_subsequence(sequence, len_subseq):
    """
    Pulls out an excerpt of length N.

    :param sequence:    an array-like to sample from
    :param len_subseq:  the desired length of the subsequence
    :return: (subsequence, first_index_of_sequence) where `sequence[first_index_of_sequence] == subsequence[0]`
    """
    start_ind = np.random.choice(range(0, len(sequence) - len_subseq), replace=False)
    end_ind = len_subseq + start_ind
    return sequence[start_ind:end_ind], start_ind


def check_sample_uniformity(position_indices, distribution=None, distribution_scale=None, center=0.0,
        thresh=0.05, verbose=False):
    # Returns True if the position_indices are sampled uniformly from the distribution
    if distribution:
        kst = stats.kstest(position_indices, distribution)
    else:
        kst = stats.kstest(position_indices, stats.uniform(loc=center, scale=distribution_scale).cdf)
    if verbose:
        print(f"KS test for uniform distribution p={kst.pvalue}")
    return kst.pvalue > thresh


def sample_segments_offset_bounds(sequence, segment_length, offset_bounds):
    lower_bound, upper_bound = offset_bounds
    index_list = np.arange(0, len(sequence) - segment_length)

    j = 0
    while True:
        if j > 10000:
            raise ValueError("Tried 10000 times to sample in this excerpt! Did not meet constraints.")
        start_ind1 = np.random.choice(index_list, replace=False)
        offsets = index_list - start_ind1
        offsets = offsets[offsets != 0]
        meets_conditions = offsets[(abs(offsets) > lower_bound) & (abs(offsets) <= upper_bound)]
        if len(meets_conditions) > 0:
            offset = np.random.choice(meets_conditions)
            start_ind2 = start_ind1 + offset
            break
        j += 1
    assert abs(start_ind1 - start_ind2) <= upper_bound, "Sample violated distance bound!"

    return start_ind1, start_ind2
