import pandas as pd
from mixer_config import (DATASET_CAPTIONS_PATH_TEST,
                    DATASET_CAPTIONS_PATH_TRAIN,
                    DATASET_CAPTIONS_PATH_VAL)


def load_imageclef_data() -> dict:
    """Load ImageCLEF dataset from directory"""
    print("\n[DATA] Loading ImageCLEF datasets...")
    
    # Get dataset paths from arguments
    dataset_captions_path_train = DATASET_CAPTIONS_PATH_TRAIN
    dataset_captions_path_valid = DATASET_CAPTIONS_PATH_VAL
    dataset_captions_path_test = DATASET_CAPTIONS_PATH_TEST

    print(f"[DATA] Loading training data from: {dataset_captions_path_train}")
    print(f"[DATA] Loading validation data from: {dataset_captions_path_valid}")
    print(f"[DATA] Loading test data from: {dataset_captions_path_test}")

    # Load datasets into pandas dataframes
    clef_captions_df_train = pd.read_csv(dataset_captions_path_train, names=["ID", "Caption"], header=0)
    clef_captions_df_valid = pd.read_csv(dataset_captions_path_valid, names=["ID", "Caption"], header=0)
    clef_captions_df_test = pd.read_csv(dataset_captions_path_test, names=["ID", "Caption"], header=0)

    # Debugging - using smaller subsets
    # print("[DEBUG] Using reduced dataset for debugging...")
    # clef_captions_df_train = clef_captions_df_train.head(100)
    # clef_captions_df_valid = clef_captions_df_valid.head(1)
    # clef_captions_df_test = clef_captions_df_test.iloc[2000:2001]

    # Convert to dictionary format
    captions_train = dict(zip(clef_captions_df_train.ID.to_list(), clef_captions_df_train.Caption.to_list()))
    captions_valid = dict(zip(clef_captions_df_valid.ID.to_list(), clef_captions_df_valid.Caption.to_list()))
    captions_test = dict(zip(clef_captions_df_test.ID.to_list(), clef_captions_df_test.Caption.to_list()))
    
    print(f"[DATA] Loaded {len(captions_train)} training, {len(captions_valid)} validation, {len(captions_test)} test samples")
    return captions_train, captions_valid, captions_test

def split_data(captions_train: dict, captions_valid: dict, captions_test: dict):
    """Split data into ID lists for each set"""
    print("\n[DATA] Splitting dataset into ID lists...")
    train_ids = list(captions_train.keys())
    dev_ids = list(captions_valid.keys())
    test_ids = list(captions_test.keys())
    return train_ids, dev_ids, test_ids

def split_(dict_to_split_train: dict, dict_to_split_val: dict, dict_to_split_test: dict):
    """Split caption dictionaries into lists of captions"""
    print("\n[DATA] Splitting caption dictionaries into lists...")
    train, dev, test = list(), list(), list()
    
    for k in dict_to_split_train.keys():
        train.append((dict_to_split_train[k].split(';')))
        
    for k in dict_to_split_val.keys():
        dev.append((dict_to_split_val[k].split(';')))
        
    for k in dict_to_split_test.keys():
        test.append((dict_to_split_test[k].split(';')))
            
    return train, dev, test