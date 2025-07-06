import torch

# Data paths for the ImageCLEF 2025 dataset
DATASET_CAPTIONS_PATH_TRAIN = "/media/SSD_2TB/imageclef2025/splits/processed/captions/train_captions.csv"
DATASET_CAPTIONS_PATH_VAL = "/media/SSD_2TB/imageclef2025/splits/processed/captions/valid_captions.csv"
DATASET_CAPTIONS_PATH_TEST = "/media/SSD_2TB/imageclef2025/splits/processed/captions/dev_captions.csv"
DATASET_IMAGES_PATH = "/media/SSD_2TB/imageclef2025/images/224x224/all/"

# Paths for model, results, and checkpoints
BEST_MODEL_PATH = "/media/SSD_2TB/imageclef2025/repo-code/Mixer/models/best-model-stratified.pt"
RESULTS_PATH = "/media/SSD_2TB/imageclef2025/repo-code/Mixer/results/results-stratified.txt"
GENERATIONS_PATH = "/media/SSD_2TB/imageclef2025/repo-code/Mixer/results/generations-stratified.csv"
CHECKPOINT_PATH = "/media/SSD_2TB/imageclef2025/repo-code/Mixer/checkpoints/"

# Model and processor names
MODEL_NAME = "Salesforce/instructblip-flan-t5-xl"

# Hyperparameters
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SEED = 42
INSTRUCTION = "You are an experienced radiologist. You are being given radiology images along with a short medical diagnosis. Generate a descriptive caption that highlights the location, nature and severity of the abnormality of the radiology image."
LEARNING_RATE = 2e-6
TRAIN_PARAMS = {
    'epochs': 20,
    'log_interval': 10,
    'dataloader': {
        'batch_size': 1,
        'shuffle': True,
        'num_workers': 4,
    },
}
VALID_PARAMS = {
    'epochs': 1,
    'log_interval': 10,
    'dataloader': {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 4,
    },
}
TEST_PARAMS = {
    'epochs': 1,
    'dataloader': {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 4,
    },
}