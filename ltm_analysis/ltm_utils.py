import pandas as pd
import matplotlib.pyplot as plt
import scipy
import os
import json
import numpy as np
from scipy.stats import bootstrap
from tqdm.notebook import tqdm
import pickle
import glob

def add_correctness(df, label_list=["A","B"]):
    df["ground_truth"] = df[f"{label_list[0]}_is_first"].apply(lambda x: label_list[::-1][x])
    df["correct"] = df["answer"] == df["ground_truth"]
    return df

def add_excerpt_length(df):
    df["excerpt_length"] = df["data"].apply(lambda x: int(x.split("-")[0]))
    df["segment_length"] = df["data"].apply(lambda x: int(x.split("-")[1].replace("s","")))

    return df


def add_distance_bin(df):
    equal_bins,bin_edges = pd.qcut(df['segment_dist'], 18, labels=False,retbins = True)
    df['bin_idx'] = equal_bins
    bin_centers = []
    for i,bin_val in enumerate(bin_edges[:-1]):
        bin_center = np.mean([bin_val,bin_edges[i+1]])
        bin_centers.append(bin_center)
    bin_center_arr = [bin_centers[bin_idx] for bin_idx in equal_bins.values]
    df['bin_center'] = bin_center_arr
    return df

# Step 3: Define the helper function
def find_largest_bin(value, bins):
    # Find the largest bin that is greater than the value
    for bin_value in sorted(bins):
        if value < bin_value:
            return bin_value
    return np.max(bins)  # or some default value if needed

def recover_distance_bin(df,human):
    '''recover the original distance bins (same # samples across bins)'''
    original_distance_bins = [1000,2000,10000,20000,15000]
    if 'distance_bin_x' not in df.columns:
        # Step 4: Apply the helper function to the dataframe
        df['distance_bin'] = df['distance_in_whole_book_x'].apply(find_largest_bin, bins=original_distance_bins)
    else:
        df['distance_bin'] = df['distance_bin_x']
    return df

def add_human_distance_bin(df,human):
    '''bins used to present human data'''
    if human:
        bins = [0,400,1000,1500,4000,10000,16500,25000,40000]
        bin_strings = ['0-400','400-1k','1k-1.5k','1.5k-4k','4k-10k','10k-16.5k','16.5k-25k','25k-40k']
        bin_indices = np.digitize(df['segment_dist'], bins)
    else:
        bins = [0,400,1000,1500,4000,10000,16500]#,25000,40000]
        bin_strings = ['0-400','400-1k','1k-1.5k','1.5k-4k','4k-10k','10k-16.5k']
        bin_indices = np.digitize(df['segment_dist'], bins)
    # Adjust bin indices to be 0-based
    bin_indices -= 1
    #bin_indices_strings = [bin_strings[i] for i in bin_indices]
    df['bin_human'] = bin_indices
    return df

def add_segment_distance(df,segment_len = None):
    if segment_len is None:
        assert np.all(df['seg1_pos_x'] == df['seg1_pos_y'])
        assert np.all(df['seg2_pos_x'] == df['seg2_pos_y'])
        assert 'segment_length' in df.columns
        segment_len = df['segment_length']
        dist = np.abs(df['seg1_pos_x']-df['seg2_pos_x'])-segment_len
        
    else:
        if 'seg1_pos_x' in df.columns:
            dist = np.abs(df['seg1_pos_x']-df['seg2_pos_x'])-segment_len
        else:
            dist = np.abs(df['seg1_pos']-df['seg2_pos'])-segment_len
    df['segment_dist'] = dist
    return df

def apply_all(df, download_dir, label_list=["A","B"], LTM=True,human = False,segment_lengths = None):
    df = add_correctness(df,label_list)
    if human==False:
        df = add_excerpt_length(df)
        df = add_segment_distance(df)
        df = add_distance_bin(df)
    else:
        df = recover_distance_bin(df,human)
        if 'distance_in_whole_book_x' in df.columns:
            df['segment_dist'] = df['distance_in_whole_book_x']
        else:
            df = add_segment_distance(df,int(segment_lengths[0][1:]))
    df = add_human_distance_bin(df,human = True)
    return df