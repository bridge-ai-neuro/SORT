import logging
import numpy as np
import os
import sys

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import time
import rag_utils


def get_database(index_str, embed_model, books_df, chunk_size=1024, rebuild=False):
    if os.path.exists(index_str) or rebuild:
        print(f"Loading search index from {index_str}")
        db = FAISS.load_local(index_str, embed_model, allow_dangerous_deserialization=True)
    else:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        print(f"Building search index...")
        # Load the books
        loader = rag_utils.CustomDocLoader(books_df)
        docs = loader.load()
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
        docs = text_splitter.split_documents(docs)
        doc_lens = np.array([len(d.page_content.split()) for d in docs])
        # Build the FAISS index from documents
        t0 = time.time()
        db = FAISS.from_documents(docs, embed_model)
        print(time.time() - t0)
        db.save_local(index_str)
    return db


def select_best_search_query(db, books_df, excerpts_df, segments_df):
    all_recall = []
    for si, search_query in enumerate(rag_utils.query_options):
        print(f"Getting performance for option {si}: {search_query}")
        rx = rag_utils.measure_retrieval_performance(search_query, db, books_df, excerpts_df, segments_df)
        all_recall.append(rx)
        print(f"10 recall@10: {rx.sum() / (rx.shape[0] * 2)}")
    all_recall = np.array(all_recall)
    recall_2 = all_recall[:, :, :2, :].sum(-1).sum(-1).mean(-1) / 2
    recall_10 = all_recall.sum(-1).sum(-1).mean(-1) / 2
    best_2 = recall_2.argmax()
    best_10 = recall_10.argmax()
    print(f"best recall@2: option {best_2} {recall_2[best_2]}\n{rag_utils.query_options[best_2]}")
    print(f"best recall@10: option {best_10} {recall_10[best_10]}\n{rag_utils.query_options[best_10]}")


if __name__ == "__main__":
    # PARAMETERS TO SET
    model_kwargs = {"device": "cuda"}
    encode_kwargs = {"normalize_embeddings": True, 'batch_size': 128}
    # Parameters used in paper
    chunk_size = 1024  # characters
    model_name = "BAAI/bge-large-en-v1.5"
    books, excerpts, segments = rag_utils.load_csv_data(data_path='../../data/booksort', file_suffix='250-s50-n110')

    # Load embedding model
    embed_model = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )  # Function name may differ if you use another embedding model
    model_str = model_name.split("/")[-1]
    index_str = f"faiss-index_{model_str}_fulltext_recursive-c{chunk_size}"

    # Build or load vector database
    db = get_database(index_str, embed_model, books, chunk_size, rebuild=False)

    # Select the best search prompt based on validation set
    select_best_search_query(db, books, excerpts, segments)
