from .llama_refiner import LLaMACaptionRefiner
from .idefics_refiner import IdeficsCaptionRefiner

def get_refiner(refiner_type, config):
    if refiner_type == "llama":
        return LLaMACaptionRefiner(config["models"]["llama"])
    elif refiner_type == "idefics":
        return IdeficsCaptionRefiner(config["models"]["idefics"])
    else:
        raise ValueError(f"Unsupported refiner type: {refiner_type}")
