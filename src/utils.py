import numpy as np 
from numpy.linalg import norm
import pandas as pd
import random

def combine_modality_features(meta: pd.DataFrame, feats: np.ndarray): 
    '''
    Combine the features of different modalities together in the numpy array
    and return a single meta dataframe. 
    '''
    grouped = meta.groupby(['compound_concentration', 'site', 'plate_name', 'MoA', 'well'])

    df_combined = grouped.apply(lambda x: x.index.tolist(), include_groups=False)
    # Convert the Series to a DataFrame
    df_combined = df_combined.reset_index()

    # Rename the column with indices list
    df_combined = df_combined.rename(columns={0: 'indices_list'})

    feats_combined = [np.concatenate(feats[i]) for i in df_combined.indices_list]

    feats_combined = np.array(feats_combined)

    return df_combined, feats_combined

def load_data(meta_path: str, feats_path: str) -> tuple[pd.DataFrame, np.ndarray]:
    '''
    Load meta and feats file. 

    Args:
    - model: str, model name
    - type: str, 'train' or 'test'
    '''
    meta = pd.read_csv(meta_path, index_col=0)
    feats = np.load(feats_path)

    meta = meta.reset_index(drop=True)

    return meta, feats

def select_meta(meta:pd.DataFrame, feats:np.array, value, target='compound_name') -> tuple[pd.DataFrame, np.array]: 

    if isinstance(value, (str, int, float)): # if value is single 
        meta = meta[meta[target] == value] 
        
    else: 
        meta = meta[meta[target].isin(value)] # if there is a list of values

    idx = meta.index
    feats = feats[idx]
    return meta.reset_index(drop=True), feats

def reduce_number_moa(meta, feats, moa, frac):
    # Filter to get only the "Lipids" rows
    moa_df = meta[meta['MoA'] == moa]

    # Randomly sample half of the "Lipids" rows
    half_lipids_sample = moa_df.sample(frac=frac, random_state=42)  # random_state is set for reproducibility

    # Drop the sampled rows from the original DataFrame
    df_reduced = meta.drop(half_lipids_sample.index)

    feats_reduced = feats[df_reduced.index]

    return df_reduced.reset_index(drop=True), feats_reduced

def majority_vote_prediction(meta: pd.DataFrame, preds_class, preds_proba) -> pd.DataFrame: 
    '''
    Return the prediction based on the majority vote for each concerned well. 
    '''
    meta['preds_class'] = preds_class
    meta['preds_proba'] = [max(x) for x in preds_proba]

    # Group by 'compound_name' and 'preds_class', then get the value counts
    grouped = meta.groupby(['well', 'plate_name'])['preds_class'].value_counts().reset_index(name='counts')

    # Sort by 'compound_name' and 'counts' (in descending order)
    sorted_grouped = grouped.sort_values(['well', 'plate_name', 'counts'], ascending=[True, True, False])

    # Drop duplicates, keeping the first occurrence (which is the largest due to sorting)
    largest_counts = sorted_grouped.drop_duplicates(subset=['well', 'plate_name'], keep='first')

    largest_counts.rename(columns={'preds_class' : 'majority_class'}, inplace=True)

    merged = pd.merge(largest_counts, meta, on=['well', 'plate_name'])

    return merged