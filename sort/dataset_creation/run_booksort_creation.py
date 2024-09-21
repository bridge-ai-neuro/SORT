"""
Generates 3 separate CSV files for different excerpt & segment lengths. These csv files form BookSORT, and are the
inputs to SORT evaluation.

USAGE
The default parameters will generate the full BookSORT dataset:
    python run_booksort_creation.py
You may input arguments to control which books are used, the excerpt and segment lengths, the number of samples per
condition, and the path for the output CSV files.
    python run_booksort_creation.py --doc_ids 69087 --excerpt_len 250 --segment_len 50 --nsamples_per_cond 10 \
        --output_path ./custom_booksort_path/

SCRIPT OUTPUTS
This script outputs 3 CSV files for each excerpt length (el) and segment length (sl) combination. We can consider the
excerpt and segment length as a single condition.
(1) books_{el}-s{sl}-n{n_samples}.csv: Information about the books included in this condition.
(2) excerpts_{el}-s{sl}-n{n_samples}.csv: The text excerpts for each book and associated information.
(3) segments_{el}-s{sl}-n{n_samples}.csv: The text segments corresponding to each excerpt and associated information.
                                          The `segment_1` column is the text segment that appears first in the text,
                                          and `segment_2` appears last. There is also the column `present_seg1_first`
                                          which is a binary variable key to indicate whether segment 1 is presented
                                          as the first option for SORT evaluation. This value is counterbalanced
                                          within each condition to ensure chance performance is exactly 50%.

BOOK DATA
This script requires preprocessed book text files. Please run `sort/dataset_creation/preprocess_pg_books.py` to
generate two files for each book:
(1) {book_id}_words.npy: a cleaned numpy array of the words in the full text, including chapter titles but excluding
                         front and back matter in the book such as the table of contents, author's note, etc.
(2) {book_id}_chapter_info.npy: a metadata dictionary containing information about the chapters in the book. this
                                allows us to generate the BookSORT samples excluding the chapter titles.


"""
import argparse
from sort.dataset_creation.create_sort_text_dataset import create_sort_samples
from sort.dataset_creation.text_utils import concatenate_text_without_titles
import numpy as np
import os
import pandas as pd


def main(book_info_path, book_list, excerpt_lengths, segment_lengths, samples_per_condition, output_dir):
    # Checking the directories given as inputs
    assert any(['words.npy' in fname for fname in os.listdir(book_info_path)]), \
        f"No *words.npy files found in {book_info_path}!"  # Check that we have some word arrays
    assert any(['chapter_info.npy' in fname for fname in os.listdir(book_info_path)]), \
        f"No *chapter_info.npy files found in {book_info_path}!"  # Check that we have chapter info
    os.makedirs(output_dir, exist_ok=True)  # Make the output directory if needed

    # Other variables we will use in dataset creation
    n_samples = None
    book_df_cols = ["book_idx", "book_title", "num_words"]
    excerpt_df_cols = ["book_idx", "excerpt_idx", "excerpt_pos", "excerpt_text"]
    segment_df_cols = ["book_idx", "excerpt_idx", "segment_idx",
                       "segment_1", "segment_2", "seg1_pos", "seg2_pos",
                       "distance_bin", "present_seg1_first"]

    for el in excerpt_lengths:
        for sl in segment_lengths:
            data_to_save = {b: dict() for b in book_list}
            if el < 5000:
                segment_distance_bins = [sl, el // 4, el // 3, el // 2, int(el // 1.25)]
            else:
                segment_distance_bins = [sl, 1000, el // 4, el // 2, int(el // 1.25)]
            n_bins = len(segment_distance_bins)
            # Initialize output data frames (which will be written to CSV)
            book_df = pd.DataFrame(columns=book_df_cols)
            excerpt_df = pd.DataFrame(columns=excerpt_df_cols)
            segment_df = pd.DataFrame(columns=segment_df_cols)
            for i, book_id in enumerate(book_list):
                words_book = np.load(f'{book_info_path}/{book_id}_words.npy', allow_pickle=True)
                metadata = np.load(f"{book_info_path}/{book_id}_chapter_info.npy", allow_pickle=True).item()
                if book_id == 73017:
                    # Only one chapter (it's an essay), so we just need to strip the title from the beginning
                    last_i = len(metadata['book_title'].split())
                    chapter_text_without_title = words_book[last_i:]
                else:
                    book_text = concatenate_text_without_titles(words_book, metadata)
                # add to books dataframe
                if metadata["num_words"] < el:
                    print(f"Skipping book {book_id}: {metadata['book_title']} {metadata['num_words']} words -- too short!")
                    continue
                vals = [book_id, metadata['book_title'].title(), metadata["num_words"]]
                book_df = pd.concat([book_df, pd.DataFrame(dict(zip(book_df_cols, vals)), index=[0])], ignore_index=True)
                print(f"PROCESSING book {i} {book_id} e{el},s{sl}")
                output = create_sort_samples(book_text, samples_per_condition, excerpt_len=el, segment_len=sl,
                                             segment_distance_bins=segment_distance_bins, seed=book_id + el + sl,
                                             extra_samples=26)
                samples, segments, answers, segment_positions, excerpt_pos, args = output  # unpack the output
                dist_keys = list(samples.keys())
                if n_samples is None:
                    n_samples = len(samples[segment_distance_bins[1]])

                for j, dk in enumerate(dist_keys):
                    if j == 0:
                        book_idx = np.repeat(book_id, n_samples)
                        excerpt_idx = np.arange(0, n_samples)
                        ex_df = pd.DataFrame(dict(zip(excerpt_df_cols, [book_idx, excerpt_idx, excerpt_pos, samples[dk]])),
                                             index=excerpt_idx)
                    elif j == 1:
                        assert np.all(samples[dk] == ex_df['excerpt_text'].to_numpy()), "Excerpts do not match across bins!"
                    seg, seg_pos, ans = segments[dk], segment_positions[dk], answers[dk]
                    # Re-sort the data so segment_1 == the segment that occurs first
                    segment_1 = np.take_along_axis(seg, ans[:, None], axis=1).squeeze()  # the correct answer
                    segment_2 = np.take_along_axis(seg, 1 - ans[:, None], axis=1).squeeze()
                    seg1_pos = np.take_along_axis(seg_pos, ans[:, None], axis=1).squeeze()  # position of correct answer
                    seg2_pos = np.take_along_axis(seg_pos, 1 - ans[:, None], axis=1).squeeze()
                    assert np.all(seg1_pos < seg2_pos), "Segment position 1 should always occur before 2, but does not"
                    vals = [book_idx, excerpt_idx, np.zeros((n_samples,), dtype=int),
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
            # add metadata to books (title)
            book_df.to_csv(f"{output_dir}/books_{el}-s{sl}-n{n_samples}.csv")
            print(f'Wrote excerpt, segment info to {output_dir}/*_{el}-s{sl}-n{n_samples}.csv')


if __name__ == "__main__":
    os.chdir(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('-ta', '--text_array_path', type=str,
                        default=f"./data/pg/text_arrays",
                        help="Path to a directory with numpy arrays of text data")
    parser.add_argument('-d', '--doc_ids', type=int, nargs='+',
                        default=[69087, 72578, 72600, 72869, 72958, 72963, 72972, 73017, 73042],
                        help="List of document (here, book) identifiers")
    parser.add_argument('-el', '--excerpt_len', type=int, nargs='+',
                        default=[250, 1000, 2500, 10000, 20000],
                        help="Excerpt length (in words). Excerpts are taken from the book text, and contain the " +
                             "entirety of both segments that need to be ordered.")
    parser.add_argument('-sl', '--segment_len', type=int, nargs='+',
                        default=[20, 50],
                        help="Segment length (in words). Segments are taken from the excerpts.")
    parser.add_argument('-ns', '--nsamples_per_cond', type=int,
                        default=110)
    parser.add_argument('-o', '--output_path', type=str,
                        default='./data/booksort/',
                        help='Path to store the SORT CSV files')
    args = parser.parse_args()

    main(args.text_array_path, args.doc_ids, args.excerpt_len, args.segment_len, args.nsamples_per_cond,
         args.output_path)
