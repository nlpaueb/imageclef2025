CONFIG = {
    "instruction": (
        "Using the image content, tags, and the draft caption, refine the caption.\n"
        "The refined caption must include:\n"
        "- Imaging modality (e.g., X-ray, CT, MRI)\n"
        "- Body part or organ\n"
        "- Significant findings or abnormalities\n\n"
        "If tags are available, ensure that the information from the tags is incorporated into the refined caption.\n"
        "Use formal radiology terminology. Do not use casual phrases.\n"
        "Do not add new information or speculate beyond the provided inputs.\n"
        "The refined caption can be more than one sentence if needed."
    ),

    "paths": {
        "images_dir": "/your/path/to/all/images",
        "captions_csv": "/your/path/to/draft/captions.csv",
        "neighbors_csv": "/your/path/to/neighbor/captions.csv",
        "tags_csv": "/your/path/to/predicted/tags.csv",
        "concept_map_csv": "/your/path/to/cui_mapping.csv",
        "result_csv": "/your/path/to/output/predictions.csv"
    },

    "models": {
        "idefics": {
            "model_id": "HuggingFaceM4/idefics2-8b"
        }
    }
}
