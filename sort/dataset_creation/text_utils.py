import numpy as np
import re
import string

SENTENCE_MARKERS = {'.', '?', '!', ';'}
PUNCTUATION = string.punctuation + '“”,’'
nopunc = str.maketrans('', '', PUNCTUATION)
roman_numerals = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV", "XV"]


def is_capitalized(text_str):
    # Returns True if the text string is capitalized, after stripping out punctuation.
    try:
        return text_str.translate(nopunc)[0].isupper()
    except Exception as e:
        return False


def has_sentence_marker(text_str):
    # Returns True if the string contains a sentence marker.
    if text_str.lower() in ['mr.', 'mrs.', 'dr.']:
        return False  # assume these are followed by names and never are the end of sentences
    else:
        return len(SENTENCE_MARKERS & set(text_str)) > 0


def sample_random_text(words, num_words, max_len=None, enforce_sentence_bounds=(True, False)):
    """
    Pulls out an excerpt of length num_words. If desired, you can enforce the segment to begin or end at sentence
    boundaries.  Similar to utils.sample_random_subsequence, with more features.

    :param words: an array-like of words to sample from.
    :param num_words: the desired length of the segment. the segment may not be exactly num_words if you set any part of
                      enforce_sentence_bounds to be True.
    :param max_len: the maximum length of the segment.
    :param enforce_sentence_bounds: tuple of (start_at_sentence, end_at_sentence). If True, forces the beginning and/or
                                    end of the sampled segment to fall at sentence boundaries.
    :return: (word_segment, first_index_of_segment) where `words[first_index_of_segment] == word_segment[0]`
    """
    start_ind = np.random.choice(range(0, len(words)-num_words), replace=False)
    if enforce_sentence_bounds[0]:
        # Find the start of this sentence
        while (start_ind > 0) and (not has_sentence_marker(words[start_ind - 1])):
            start_ind = start_ind - 1
    end_ind = num_words+start_ind
    if enforce_sentence_bounds[1]:
        # Find the end of the sentence
        while not has_sentence_marker(words[end_ind-1]):
            end_ind += 1
        if (end_ind - start_ind) > max_len:
            end_ind = num_words + start_ind
            while '.' not in words[end_ind-1]:
                end_ind = end_ind - 1
    return words[start_ind:end_ind], start_ind


def sample_segments_sentence_start_bins(sample_text, seg_length, offset_bounds):
    # Modification of utils.sample_segments_offset_bounds to force segments to begin at a sentence boundary
    lower_bound, upper_bound = offset_bounds
    sent_starts = np.array([i + 1 for i, w in enumerate(sample_text) if has_sentence_marker(w)])
    sent_starts = np.insert(sent_starts, 0, [0])
    sent_starts = sent_starts[sent_starts < (len(sample_text) - seg_length)]

    j = 0
    while True:
        if j > 10000:
            raise ValueError("Tried 10000 times to sample in this excerpt! Did not meet constraints.")
        start_ind1 = np.random.choice(sent_starts, replace=False)
        offsets = sent_starts - start_ind1
        offsets = offsets[offsets != 0]
        meets_conditions = offsets[(abs(offsets) > lower_bound) & (abs(offsets) <= upper_bound)]
        if len(meets_conditions) > 0:
            offset = np.random.choice(meets_conditions)
            start_ind2 = start_ind1 + offset
            break
        j += 1
    if upper_bound > lower_bound:
        assert abs(start_ind1 - start_ind2) < upper_bound, "Sample violated distance bound!"

    return start_ind1, start_ind2, sent_starts


def concatenate_text_without_titles(words_book, chapters):
    chapter_inds = np.concatenate((np.array([0]), np.cumsum(chapters['chapter_inds'])))
    all_text_without_titles = []
    for ci in np.arange(1, len(chapter_inds)):
        ch_text = words_book[chapter_inds[ci - 1]:chapter_inds[ci]]
        if chapters['chapter_titles']:
            # Remove the full title
            ch_title = chapters['chapter_titles'][ci - 1].split()
            last_i = [i for i in range(len(ch_title) + 5) if ch_title[-1].lower() == ch_text[i].lower()]
            chapter_text_without_title = ch_text[last_i[-1] + 1:]
        else:
            # Some books don't have chapter titles and only have numeric section markers
            last_i = [i for i in range(5) if ch_text[i].isnumeric()]
            if len(last_i) > 0:
                # Handle standard numeric section markers
                chapter_text_without_title = ch_text[last_i[-1] + 1:]
            else:
                # Handle roman numeral section markers
                chapter_text_without_title = ch_text[2:]
                ri = 0
                txt = ' '.join(chapter_text_without_title)
                tmp = re.search(roman_numerals[ri], txt)
                while tmp is not None:
                    txt = txt[:tmp.span()[0]] + txt[tmp.span()[1]:]
                    ri += 1
                    tmp = re.search(roman_numerals[ri], txt)
                chapter_text_without_title = txt.split()
        all_text_without_titles.append(chapter_text_without_title)
    return np.hstack(all_text_without_titles)
