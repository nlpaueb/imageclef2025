import torch
from torch.utils.data import DataLoader
from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor
from instructBLIP_dataset import CustomVisionDataset
from PIL import Image
from tqdm import tqdm
import pandas as pd
import os

from medclip_reranker import MedCLIPReranker
from medclip_config import (
    BEST_MODEL_PATH,
    TEST_CAPTIONS_PATH,
    IMAGES_PATH,            
    OUTPUT_CSV_PATH,
    BATCH_SIZE,
    NUM_RETURN_SEQUENCES,
    DEVICE,
    MEDCLIP_CHECKPOINT,
    INSTRUCTION
)

# Load test IDs
test_df = pd.read_csv(TEST_CAPTIONS_PATH)
test_df['ID'] = test_df['ID'].apply(lambda x: x.replace('.jpg', '') if x.endswith('.jpg') else x).tolist()
test_ids = test_df['ID']
dummy_captions = dict(zip(test_ids, [''] * len(test_ids)))

# Load InstructBLIP
print("Loading InstructBLIP model...")
instructblip_model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl")
instructblip_model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location='cpu'))
instructblip_model.to(DEVICE).eval()
instructblip_processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")

# Load MedCLIP reranker
reranker = MedCLIPReranker(checkpoint_path=MEDCLIP_CHECKPOINT, device=DEVICE)

# Prepare dataset and dataloader
print("Preparing dataset...")
dataset = CustomVisionDataset(dummy_captions, test_ids, instructblip_processor, mode='test')
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# Inference loop
print("Running inference...")
results = []

generation_args = {
    "max_length": 80,
    "num_return_sequences": NUM_RETURN_SEQUENCES,
    "do_sample": True,
    "top_p": 0.9,
    "top_k": 50,
    "temperature": 0.8,
    "repetition_penalty": 1.0,
    "length_penalty": 1.0,
    "return_dict_in_generate": True
}

with torch.no_grad():
    for batch_idx, (images, _, ids) in enumerate(tqdm(dataloader)):
        try:
            instructions = [INSTRUCTION] * len(images)
            inputs = instructblip_processor(images=images, text=instructions, return_tensors="pt").to(DEVICE)

            outputs = instructblip_model.generate(**inputs, **generation_args)
            all_captions = instructblip_processor.batch_decode(outputs.sequences, skip_special_tokens=True)

            for i in range(len(images)):
                img_id = ids[i]
                img_captions = all_captions[i*NUM_RETURN_SEQUENCES : (i+1)*NUM_RETURN_SEQUENCES]
                image_path = os.path.join(IMAGES_PATH, img_id + '.jpg')
                image = Image.open(image_path).convert("RGB")
                best_caption, _ = reranker.rank_captions(image, img_captions)
                results.append((img_id, best_caption))

        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            for i in range(len(images)):
                fallback_caption = all_captions[i*NUM_RETURN_SEQUENCES]
                results.append((ids[i].replace('.jpg', ''), fallback_caption))

# Save results
print(f"Saving predictions to {OUTPUT_CSV_PATH}...")
df_out = pd.DataFrame(results, columns=['ID', 'Caption'])
df_out.to_csv(OUTPUT_CSV_PATH, index=False)
print("Done.")
