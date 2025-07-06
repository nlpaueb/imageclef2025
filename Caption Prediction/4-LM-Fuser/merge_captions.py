import os
import pandas as pd

# Define paths
model_paths = {
    "model_1": "/path/to/your/model/1/inference/results/", # Replace with actual path
    "model_2": "/path/to/your/model/2/inference/results/", # Replace with actual path
    "model_3": "/path/to/your/model/3/inference/results/" # Replace with actual path
}

# Define output path
output_path = "/path/to/your/output/directory/" # Replace with actual path

# Expected data splits
splits = ["train", "valid", "dev"]

# Process each split
for split in splits:
    # Build full paths to CSV files
    file_1 = os.path.join(model_paths["model_1"], f"name_model_1_{split}.csv") # Replace with actual file name
    file_2 = os.path.join(model_paths["model_2"], f"name_model_2_{split}.csv") # Replace with actual file name
    file_3 = os.path.join(model_paths["model_3"], f"name_model_3_{split}.csv") # Replace with actual file name

    # Load CSVs
    df1 = pd.read_csv(file_1, sep="|")
    df2 = pd.read_csv(file_2, sep="|")
    df3 = pd.read_csv(file_3, sep="|")

    # Remove .jpg from the ID column if it exists
    df1["ID"] = df1["ID"].str.replace(".jpg", "", regex=False)
    df2["ID"] = df2["ID"].str.replace(".jpg", "", regex=False)
    df3["ID"] = df3["ID"].str.replace(".jpg", "", regex=False)

    # Print previews
    print(f"\n=== Split: {split} ===")
    print(df1.head())
    print(df2.head())
    print(df3.head())

    # Merge DataFrames on 'ID'
    merged = df1.merge(df2, on="ID", suffixes=("_Model_1", "_Model_2"))
    merged = merged.merge(df3, on="ID")
    merged = merged.rename(columns={"Caption": "Caption_Model_3"})

    # Keep only final caption columns
    merged = merged[["ID", "Caption_Model_1", "Caption_Model_2", "Caption_Model_3"]]

    # Save merged DataFrame to CSV
    merged.to_csv(os.path.join(output_path, f"merged_{split}.csv"), index=False, sep="|")
