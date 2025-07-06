import os
import pandas as pd
from PIL import Image
import torch
from tqdm import tqdm

def load_data(paths):
    captions = pd.read_csv(paths["captions_csv"], delimiter=',', names=['ID', 'Caption'])
    captions['ID'] = captions['ID'].apply(lambda x: x + '.jpg' if not x.endswith('.jpg') else x)

    neighbors = pd.read_csv(paths["neighbors_csv"], delimiter=',', header=0, names=['Test Image', 'Neighbor Image', 'Neighbor Caption', 'Similarities'])
    tags = pd.read_csv(paths["tags_csv"], delimiter=',', names=['ID', 'Cuis'])
    concept_map = pd.read_csv(paths["concept_map_csv"], delimiter=',', names=['CUI', 'Canonical Name'])

    return captions, neighbors, tags, concept_map

def get_tags(image_id, tags_df, concept_df):
    try:
        row = tags_df.loc[tags_df['ID'] == image_id]
        if not row.empty:
            cuis = row['Cuis'].values[0]
            cuis_list = cuis.split(";")
            tag_names = [concept_df.loc[concept_df['CUI'] == cui, 'Canonical Name'].values[0] for cui in cuis_list if not concept_df.loc[concept_df['CUI'] == cui].empty]
            return ", ".join(tag_names)
        return ""
    except Exception as e:
        print(f"[Error] Tag loading failed for {image_id}: {e}")
        return ""

def generate_caption(processor, model, device, image, tags, draft, neighbor_caption, instruction):
    text_blocks = [
        f"Tags for this image: {tags}",
        f"Here is the draft caption for this medical image:\n\nDraft Caption: {draft}"
    ]

    if neighbor_caption:
        text_blocks.append(
            f"A similar image has the following caption:\n\nNeighbor Caption: {neighbor_caption}\n\n"
            + instruction.replace("image content, tags, and the draft caption", "image content, tags, draft caption, and neighbor caption")
        )
    else:
        text_blocks.append(instruction)

    messages = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": "\n\n".join(text_blocks)}]
    }]

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=112,
            num_beams=4,
            do_sample=False
        )

    return processor.batch_decode(output_ids, skip_special_tokens=True)[0].split("Assistant:")[-1].strip()

def run_multisynthesizer(model, processor, paths, instruction, device):
    captions_df, neighbors_df, tags_df, concept_df = load_data(paths)
    results = []

    for _, row in tqdm(neighbors_df.iterrows(), total=len(neighbors_df)):
        image_id = row['Test Image']
        neighbor_caption = row['Neighbor Caption'] if pd.notna(row['Neighbor Caption']) else None

        try:
            image = Image.open(os.path.join(paths['images_dir'], image_id)).convert("RGB")
        except Exception as e:
            print(f"[Error] Could not load image {image_id}: {e}")
            continue

        draft_row = captions_df[captions_df['ID'] == image_id]
        if draft_row.empty:
            print(f"[Warning] No draft caption for {image_id}. Skipping.")
            continue

        draft_caption = draft_row['Caption'].values[0]
        tags = get_tags(image_id, tags_df, concept_df) or "No tags found"

        refined_caption = generate_caption(processor, model, device, image, tags, draft_caption, neighbor_caption, instruction)

        print(f"Draft: {draft_caption}\nNeighbor: {neighbor_caption}\nRefined: {refined_caption}\n{'-'*50}")
        results.append({"ID": image_id, "Caption": refined_caption})

    pd.DataFrame(results).to_csv(paths['result_csv'], sep=',', index=False)
    print(f"Inference complete. Results saved to '{paths['result_csv']}'.")
