from PIL import Image
import os
import pandas as pd
from tqdm import tqdm

def run_synthesizer(refiner, images_dir, captions_csv, neighbors_csv, result_csv):
    captions_df = pd.read_csv(captions_csv, names=["ID", "Caption"])
    neighbors_df = pd.read_csv(neighbors_csv, header=0, names=["Test Image", "Neighbor Image", "Neighbor Caption", "Similarities"])
    captions_df['ID'] = captions_df['ID'].apply(lambda x: x if x.endswith('.jpg') else x + '.jpg')

    results = []

    for _, row in tqdm(neighbors_df.iterrows(), total=len(neighbors_df)):
        image_id = row['Test Image']
        neighbor_id = row['Neighbor Image']

        try:
            image = Image.open(os.path.join(images_dir, image_id)).convert("RGB")
            neighbor_image = (
                Image.open(os.path.join(images_dir, neighbor_id)).convert("RGB")
                if pd.notna(neighbor_id) else None
            )
        except Exception as e:
            print(f"[Error] Skipping {image_id} due to: {e}")
            continue

        draft_row = captions_df[captions_df['ID'] == image_id]
        if draft_row.empty:
            print(f"[Warning] No draft caption for {image_id}")
            continue

        draft_caption = draft_row['Caption'].values[0]
        neighbor_caption = row['Neighbor Caption'] if pd.notna(row['Neighbor Caption']) else None

        refined_caption = refiner.refine_caption(
            image=image,
            draft_caption=draft_caption,
            neighbor_image=neighbor_image,
            neighbor_caption=neighbor_caption
        )

        print(f"\nDraft: {draft_caption}\nGenerated: {refined_caption}\n")
        results.append({"ID": image_id, "Caption": refined_caption})

    results_df = pd.DataFrame(results)
    results_df.to_csv(result_csv, sep=',', index=False)
    print(f"Saved refined captions to {result_csv}")
