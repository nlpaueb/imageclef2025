CONFIG = {
    "refiner_type": "llama",  # Options: "llama", "idefics"

    "instruction": (
        "Using the content of the image (if available) and the draft caption, refine the caption into a single, concise sentence (maximum 50 words). "
        "Ensure the refined caption includes:\n"
        "- The imaging modality (e.g., X-ray, CT, MRI)\n"
        "- The body part or organ visualized\n"
        "- Significant findings or abnormalities\n\n"
        "Use formal radiology terminology. Avoid casual language such as 'looks like' or 'seems to'. "
        "The refined caption should clearly answer: What is the modality? What anatomical region is shown? What are the key findings?\n\n"
        "Do not add information that is not evident in the image. Do not speculate."
    ),

    "paths": {
        "image_dir": "/path/to/dev/images",
        "all_images_dir": "/path/to/all/images",
        "captions_csv": "/path/to/draft_captions.csv",
        "neighbors_csv": "/path/to/neighbors.csv",
        "result_csv": "/path/to/output/refined_captions.csv"
    },

    "models": {
        "llama": {
            "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "token": "hf_YourTokenHere"
        },
        "idefics": {
            "model_id": "HuggingFaceM4/idefics2-8b"
        }
    }
}





