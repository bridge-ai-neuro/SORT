"""
Given clean input .txt files, this example will:
(1) Perform minimal preprocessing on the input .txt files.
    - Extract title from first line of the text file.
    - Split remaining string of text into words based on whitespace.
    - Assign a numerical document ID based on order of processing.
(2) Generate SORT samples from the text for a given excerpt and segment length.
"""

import argparse
import numpy as np
import os
import pandas as pd

from sort.dataset_creation.create_sort_text_dataset import create_sort_samples


def main(text_file_path, excerpt_lengths, segment_lengths, samples_per_condition, output_dir,
         save_word_arrays, extra_sample_percent):
    # Very minimal preprocessing on text.
    #   - Extract title from first line of the text file.
    #   - Split remaining string of text into words based on whitespace.
    #   - Assign a numerical document ID based on order of processing.
    assert os.path.exists(text_file_path), f"No path {text_file_path} found!"
    files = os.listdir(text_file_path)
    doc_ids = []
    all_doc_text = {}
    all_doc_metadata = {}
    for i, fn in enumerate(files):
        if fn.endswith(".txt"):
            with open(os.path.join(text_file_path, fn), "r", encoding='utf-8') as f:
                raw_text = f.readlines()
            title = raw_text[0].strip()
            remaining_text = " ".join(raw_text[1:])
            text_array = np.array(remaining_text.split())
            print(f"Turned {fn} into an array of words. Sample: {text_array[25:50]}")
            if save_word_arrays:
                np.save(os.path.join(text_file_path, f'{i}_words.npy'), text_array, allow_pickle=True)
            doc_ids.append(i)
            all_doc_text[i] = text_array
            all_doc_metadata[i] = {}
            all_doc_metadata[i]['doc_title'] = title
            all_doc_metadata[i]['num_words'] = len(text_array)
            del fn, title, remaining_text, text_array
    # Make the output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Other variables we will use in dataset creation
    n_samples = None  # This is a placeholder variable that will be defined after the first segments are generated
    # The following are the column labels for the data frames / CSV files
    doc_df_cols = ["doc_idx", "doc_title", "num_words"]
    excerpt_df_cols = ["doc_idx", "excerpt_idx", "excerpt_pos", "excerpt_text"]
    segment_df_cols = ["doc_idx", "excerpt_idx", "segment_idx",
                       "segment_1", "segment_2", "seg1_pos", "seg2_pos",
                       "distance_bin", "present_seg1_first"]
    # Number of
    extra_n = int(samples_per_condition * extra_sample_percent)
    extra_n = extra_n - (extra_n % 2)

    for el in excerpt_lengths:
        for sl in segment_lengths:
            data_to_save = {b: dict() for b in doc_ids}
            if el < 5000:
                segment_distance_bins = [sl, el // 4, el // 3, el // 2, int(el // 1.25)]
            else:
                segment_distance_bins = [sl, 1000, el // 4, el // 2, int(el // 1.25)]
            n_bins = len(segment_distance_bins)
            # Initialize output data frames (which will be written to CSV)
            doc_df = pd.DataFrame(columns=doc_df_cols)
            excerpt_df = pd.DataFrame(columns=excerpt_df_cols)
            segment_df = pd.DataFrame(columns=segment_df_cols)
            for i, doc_id in enumerate(doc_ids):
                doc_text = all_doc_text[doc_id]
                metadata = all_doc_metadata[doc_id]
                # add to docs dataframe
                if metadata["num_words"] < el:
                    print(
                        f"Skipping doc {doc_id}: {metadata['doc_title']} {metadata['num_words']} words -- too short!")
                    continue
                vals = [doc_id, metadata['doc_title'].title(), metadata["num_words"]]
                doc_df = pd.concat([doc_df, pd.DataFrame(dict(zip(doc_df_cols, vals)), index=[0])],
                                    ignore_index=True)
                print(f"PROCESSING doc {i} {doc_id} e{el},s{sl}")
                output = create_sort_samples(doc_text, samples_per_condition, excerpt_len=el, segment_len=sl,
                                             segment_distance_bins=segment_distance_bins, seed=doc_id + el + sl,
                                             extra_samples=extra_n)
                samples, segments, answers, segment_positions, excerpt_pos, args = output  # unpack the output
                dist_keys = list(samples.keys())
                if n_samples is None:
                    n_samples = len(samples[segment_distance_bins[1]])

                for j, dk in enumerate(dist_keys):
                    if j == 0:
                        doc_idx = np.repeat(doc_id, n_samples)
                        excerpt_idx = np.arange(0, n_samples)
                        ex_df = pd.DataFrame(
                            dict(zip(excerpt_df_cols, [doc_idx, excerpt_idx, excerpt_pos, samples[dk]])),
                            index=excerpt_idx)
                    elif j == 1:
                        assert np.all(
                            samples[dk] == ex_df['excerpt_text'].to_numpy()), "Excerpts do not match across bins!"
                    seg, seg_pos, ans = segments[dk], segment_positions[dk], answers[dk]
                    # Re-sort the data so segment_1 == the segment that occurs first
                    segment_1 = np.take_along_axis(seg, ans[:, None], axis=1).squeeze()  # the correct answer
                    segment_2 = np.take_along_axis(seg, 1 - ans[:, None], axis=1).squeeze()
                    seg1_pos = np.take_along_axis(seg_pos, ans[:, None], axis=1).squeeze()  # position of correct answer
                    seg2_pos = np.take_along_axis(seg_pos, 1 - ans[:, None], axis=1).squeeze()
                    assert np.all(seg1_pos < seg2_pos), "Segment position 1 should always occur before 2, but does not"
                    vals = [doc_idx, excerpt_idx, np.zeros((n_samples,), dtype=int),
                            segment_1, segment_2, seg1_pos, seg2_pos, np.repeat([dk], n_samples), 1 - ans]
                    seg_df = pd.DataFrame(dict(zip(segment_df_cols, vals)), index=excerpt_idx)
                    segment_df = pd.concat([segment_df, seg_df], ignore_index=True)
                    del seg_df
                excerpt_df = pd.concat([excerpt_df, ex_df], ignore_index=True)
                del output, samples, segments, answers, segment_positions
                del seg, seg_pos, segment_1, segment_2, seg1_pos, seg2_pos
                del ex_df

            segment_df.to_csv(f"{output_dir}/segments_{el}-s{sl}-n{n_samples}.csv")
            excerpt_df.to_csv(f"{output_dir}/excerpts_{el}-s{sl}-n{n_samples}.csv")
            # add metadata to docs (title)
            doc_df.to_csv(f"{output_dir}/docs_{el}-s{sl}-n{n_samples}.csv")
            print(f'Wrote excerpt, segment info to {output_dir}/*_{el}-s{sl}-n{n_samples}.csv')


if __name__ == "__main__":
    # navigate to home directory
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))

    parser = argparse.ArgumentParser()
    parser.add_argument('-ta', '--text_path', type=str,
                        default=f"./data/text/",
                        help="Path to a directory with .txt files")
    parser.add_argument('-el', '--excerpt_len', type=int, nargs='+',
                        default=[250, 500],
                        help="Excerpt length (in words). Excerpts are taken from the doc text, and contain the " +
                             "entirety of both segments that need to be ordered.")
    parser.add_argument('-sl', '--segment_len', type=int, nargs='+',
                        default=[10],
                        help="Segment length (in words). Segments are taken from the excerpts.")
    parser.add_argument('-ns', '--nsamples_per_cond', type=int,
                        default=50)
    parser.add_argument('-o', '--output_path', type=str,
                        default='./data/docsort/',
                        help='Path to store the SORT CSV files')
    parser.add_argument('-s', '--store_arrays', type=bool,
                        default=True,
                        help='Whether or not to store the arrays of words from each document. Default is True.')
    parser.add_argument('-p', '--generate_percent_extra', type=int,
                        default=0.5,
                        help="Extra samples to generate (as a percentage of the total number per condition). This is " +
                             "needed so that the text samples will be roughly normally distributed across the " +
                             "the excerpt, and so that they begin at a sentence boundary.")
    args = parser.parse_args()

    main(args.text_path, args.excerpt_len, args.segment_len, args.nsamples_per_cond,
         args.output_path, args.store_arrays, args.generate_percent_extra)
