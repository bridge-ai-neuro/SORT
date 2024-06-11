"""
Generates 3 separate CSV files for different excerpt & segment lengths. These csv files form BookSORT, and are the
inputs to SORT evaluation.

SCRIPT INPUTS
This script requires preprocessed book text files. Please run `sort/dataset_creation/preprocess_pg_books.py` to
generate two files for each book:
(1) {book_id}_words.npy: a cleaned numpy array of the words in the full text, including chapter titles but excluding
                         front and back matter in the book such as the table of contents, author's note, etc.
(2) {book_id}_chapter_info.npy: a metadata dictionary containing information about the chapters in the book. this
                                allows us to generate the BookSORT samples excluding the chapter titles.

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
"""
from sort.dataset_creation.create_sort_text_dataset import create_sort_samples
from sort.dataset_creation.text_utils import concatenate_text_without_titles
import numpy as np
import pandas as pd

# Parameters you may wish to modify
book_info_path = "../../data/pg/text_arrays"
book_list = [69087, 72578, 72600, 72869, 72958, 72963, 72972, 73017, 73042]
excerpt_lengths = [250, 1000, 2500, 10000, 20000]
segment_lengths = [20, 50]
samples_per_condition = 100
output_dir = '../../data/booksort/'

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
            output = create_sort_samples(book_id, samples_per_condition, excerpt_len=el, segment_len=sl,
                                         segment_distance_bins=segment_distance_bins, seed=book_id + el + sl)
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
