import os
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd
from tqdm import tqdm
from fuser_config import *

# Load the tokenizer and model
print("Loading tokenizer and model...")
tokenizer = T5Tokenizer.from_pretrained(FINAL_MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(FINAL_MODEL_PATH)
model.to(DEVICE)

# Load dataset
df = pd.read_csv(TEST_INPUTS_PATH, sep='|')
df.columns = ['ID', 'Caption_Model_1', 'Caption_Model_2', 'Caption_Model_3']
print('Length of the dataset:', len(df))

predictions = []

with open(TEST_OUTPUT_TXT_PATH, "a") as f:

    # Iterate over the dataset in batches
    for i in tqdm(range(0, len(df), TEST_BATCH_SIZE)):
        # Get the current batch
        batch_df = df.iloc[i:i + TEST_BATCH_SIZE]
        
        # Prepare the input texts for the current batch
        batch_input_texts = [
            f"You are a medical expert. Given the following three image captions from different models, generate one unified caption that is concise, accurate, and free of repetition. "
            f"Keep important clinical details, merge overlapping information, and use a professional tone:\n"
            f"1. {row['Caption_Model_1']}\n"
            f"2. {row['Caption_Model_2']}\n"
            f"3. {row['Caption_Model_3']}\n"
            "Ensure that the final caption seamlessly integrates key elements from each description and maintains a clear and unified narrative.\n"
            "Unified caption:"
            for _, row in batch_df.iterrows()
        ]
        
        # Tokenize the input texts
        batch_input_ids = tokenizer(batch_input_texts, return_tensors="pt", padding=True, truncation=True).input_ids.to(DEVICE)

        # Generate predictions for the batch
        batch_outputs = model.generate(
            batch_input_ids,
            max_length=120,
            do_sample=False,
            num_beams=4
        )
        
        # Decode the outputs and collect predictions
        for j, output in enumerate(batch_outputs):
            output_text = tokenizer.decode(output, skip_special_tokens=True)
            row = batch_df.iloc[j]

            print('Model 1:', row['Caption_Model_1'])
            print('Model 2:', row['Caption_Model_2'])
            print('Model 3:', row['Caption_Model_3'])
            print('Fuser caption:', output_text)
            print(50 * '-')

            # Append to the predictions list
            predictions.append((row['ID'], output_text))

            # Write predictions to the file
            f.write(f"{row['ID']}|{output_text}\n")

# Save the predictions to a CSV file
predictions_df = pd.DataFrame(predictions, columns=['ID', 'Caption'])
predictions_df.to_csv(TEST_OUTPUT_CSV_PATH, sep=',', index=False)
print(f"Predictions saved to {TEST_OUTPUT_CSV_PATH} and {TEST_OUTPUT_TXT_PATH}")
