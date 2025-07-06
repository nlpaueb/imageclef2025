import os
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from multisynth_config import CONFIG
from multisynthesizer import run_multisynthesizer

def main():
    # Environment and device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load configuration
    paths = CONFIG["paths"]
    instruction = CONFIG["instruction"]
    model_config = CONFIG["models"]["idefics"]

    # Load processor and model with quantization
    processor = AutoProcessor.from_pretrained(model_config["model_id"])
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    model = AutoModelForVision2Seq.from_pretrained(
        model_config["model_id"],
        torch_dtype=torch.float16,
        quantization_config=quant_config,
        device_map="auto"
    )

    # Run the synthesizer
    run_multisynthesizer(model, processor, paths, instruction, device)

if __name__ == "__main__":
    main()
