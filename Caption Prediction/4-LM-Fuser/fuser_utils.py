# fuser_utils.py
import os
import pandas as pd
import nltk
import evaluate
import numpy as np
from PIL import Image
from transformers import T5Tokenizer

# Ensure the necessary resources are downloaded
nltk.download("punkt", quiet=True)

# Constants for paths (Assuming config.py contains constants for file paths)
from fuser_config import *

# Initialize tokenizer and metric
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
metric = evaluate.load("rouge")

# Function to get image path
def get_image_path(image_id, dir=IMAGE_DIR):
    """Get full image path."""
    return Image.open(os.path.join(dir, image_id))

# Function to load and preprocess data
def load_and_preprocess_data():
    """Load and preprocess data."""
    # Load data
    train_inputs_df = pd.read_csv(TRAIN_INPUTS_PATH, sep='|')
    valid_inputs_df = pd.read_csv(VALID_INPUTS_PATH, sep='|')
    train_targets_df = pd.read_csv(TRAIN_TARGETS_PATH)
    valid_targets_df = pd.read_csv(VALID_TARGETS_PATH)

    # # Rename columns
    train_inputs_df.columns = ['ID', 'Caption_Model_1', 'Caption_Model_2', 'Caption_Model_3']
    valid_inputs_df.columns = ['ID', 'Caption_Model_1', 'Caption_Model_2', 'Caption_Model_3']

    # Ensure all IDs end with .jpg
    train_targets_df['ID'] = train_targets_df['ID'].apply(lambda x: x if x.endswith('.jpg') else x + '.jpg')
    valid_targets_df['ID'] = valid_targets_df['ID'].apply(lambda x: x if x.endswith('.jpg') else x + '.jpg')

    # Merge inputs and targets
    train_df = train_inputs_df.merge(train_targets_df, on='ID')
    valid_df = valid_inputs_df.merge(valid_targets_df, on='ID')

    return train_df, valid_df

def preprocess_function(examples, tokenizer):
    """Preprocess input examples for T5 model."""
    input_text = (
        "You are a medical expert. Given the following three image captions from different models, generate one unified caption that is concise, accurate, and free of repetition. "
        "Keep important clinical details, merge overlapping information, and use a professional tone:\n"
        f"1. {examples['Caption_Model_1']}\n"
        f"2. {examples['Caption_Model_2']}\n"
        f"3. {examples['Caption_Model_3']}\n"
        "Ensure that the final caption seamlessly integrates key elements from each description and maintains a clear and unified narrative.\n"
        "Unified caption:"
    )

    model_inputs = tokenizer(input_text, max_length=1000, truncation=True)

    labels = tokenizer(
        text_target=examples['Caption'],
        max_length=1000,
        truncation=True
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# Function to compute ROUGE metrics
def compute_metrics(eval_preds):
    """Compute ROUGE metrics."""
    preds, labels = eval_preds

    # Replace label padding (-100) with pad_token_id for decoding
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Convert to numpy arrays
    preds = np.array(preds)
    
    # Sanity check: clip preds to vocab size
    preds = np.clip(preds, 0, tokenizer.vocab_size - 1)

    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Split into sentences for ROUGE
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return result
