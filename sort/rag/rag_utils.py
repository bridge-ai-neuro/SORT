import numpy as np
import pandas as pd
import pylcs
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from sort.dataset_creation.text_utils import concatenate_text_without_titles


query_options = [
    "Please determine the order in which the following text segments appeared in <booktitle>: <segments>",
    "We need to put text segments from <booktitle> in order. These are the segments: <segments>",
    "Please find these text segments from <booktitle>: <segments>",
    "Please find these text segments from <booktitle> to provide context for the next task: <segments>",
    "Which text chunks from <booktitle> contain the following segments? <segments>",
    "Which text excerpts from <booktitle> contain the following segments? <segments>",
    "Which text chunks from <booktitle> overlap with these text segments: <segments>",
    "<booktitle> contains this text: <segments>",
    "<segments>",
    "<booktitle> <segments>"]


def load_csv_data(data_path, file_suffix):
    # Load dataset samples
    books_df = pd.read_csv(f'{data_path}/books_{file_suffix}.csv')
    excerpts_df = pd.read_csv(f'{data_path}/excerpts_{file_suffix}.csv')
    segments_df = pd.read_csv(f'{data_path}/segments_{file_suffix}.csv')
    return books_df, excerpts_df, segments_df


def measure_retrieval_performance(search_query, db,
        books_df, excerpts_df, segments_df,
        k=10, min_excerpt_index=100, max_excerpt_index=120):
    recall = []
    # Search and record retrieved neighbors
    for idx, row in segments_df.iterrows():
        if row["excerpt_idx"] < max_excerpt_index and row["excerpt_idx"] >= min_excerpt_index:
            book_idx = row["book_idx"]
            excerpt_idx = row["excerpt_idx"]
            segment_1 = row["segment_1"]
            segment_2 = row["segment_2"]
            present_seg1_first = row["present_seg1_first"]
            excerpt = \
            excerpts_df[(excerpts_df["book_idx"] == book_idx) & (excerpts_df["excerpt_idx"] == excerpt_idx)].iloc[0][
                "excerpt_text"]
            book_title = books_df[books_df["book_idx"] == book_idx]["book_title"].iloc[0]

            # Construct the query, maintaining the randomized presentation order of the segments
            if present_seg1_first:
                segments_prompt = f"Segment A: {segment_1} Segment B: {segment_2}\n"
                segments = [segment_1, segment_2]
            else:
                segments_prompt = f"Segment A: {segment_2} Segment B: {segment_1}\n"
                segments = [segment_2, segment_1]
            search_query_string = search_query.replace("<booktitle>", book_title).replace("<segments>", segments_prompt)

            d = db.similarity_search(search_query_string, k=k)
            context = [dd.page_content for dd in d]
            # Test if any of the top 10 retrieved neighbors contains the segments in question
            avg_len = (len(segments[0]) + len(segments[1])) / 2
            contains_seg = np.zeros((k, 2), dtype=np.int32)
            for i, c in enumerate(context):
                longest_substrs = np.array(pylcs.lcs_string_of_list(c, segments))
                contains_seg[i, :] = longest_substrs > (avg_len * 0.5)
            recall.append(contains_seg)

            # results_row = row
            # results_row["context"] = context
            # results_row["contains_seg"] = contains_seg

    return np.array(recall)


class CustomDocLoader(BaseLoader):
    def __init__(self, books_df, book_path='../../data/pg/text_arrays'):
        """Initialize document loader using the metadata we have about the books"""
        self.ids = books_df['book_idx']
        self.titles = books_df['book_title']
        self.data_path = book_path

    def load(self):
        doc_list = []
        # Add each book as a document
        for book_id, title in zip(self.ids, self.titles):
            word_array = np.load(f'{self.data_path}/{book_id}_words.npy', allow_pickle=True)
            ch_info = np.load(f'{self.data_path}/{book_id}_chapter_info.npy', allow_pickle=True).item()
            text = concatenate_text_without_titles(word_array, ch_info)
            text = " ".join(text)
            doc = Document(page_content=text, metadata={'id': book_id, 'title': title})
            doc_list.append(doc)
        return doc_list

