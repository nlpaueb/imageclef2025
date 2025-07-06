from synth_config import CONFIG
from synthesizer import run_synthesizer
from refiners import get_refiner

if __name__ == "__main__":
    refiner = get_refiner(CONFIG["refiner_type"], CONFIG)
    paths = CONFIG["paths"]
    run_synthesizer(
        refiner=refiner,
        images_dir=paths["images_dir"],
        captions_csv=paths["captions_csv"],
        neighbors_csv=paths["neighbors_csv"],
        result_csv=paths["result_csv"]
    )
