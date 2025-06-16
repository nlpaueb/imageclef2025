import torch

# Data paths for the ImageCLEF 2025 dataset
DATASET_CAPTIONS_PATH_TRAIN = "/path/to/your/dataset/splits/random/train_random.csv"
DATASET_CAPTIONS_PATH_VAL = "/path/to/your/dataset/splits/random/val_random.csv"
DATASET_CAPTIONS_PATH_TEST = "/path/to/your/dataset/splits/random/dev_random.csv"
DATASET_IMAGES_PATH = "/path/to/your/dataset/images/all/"
BEST_MODEL_PATH = "/path/to/your/models/best-model-random.pt"
RESULTS_PATH = "/path/to/your/results/results-random.txt"
GENERATIONS_PATH = "/path/to/your/results/generations-random.csv"

# Model and processor names
MODEL_NAME = "Salesforce/instructblip-flan-t5-xl"

# Hyperparameters
DEVICE = "cuda:5" if torch.cuda.is_available() else "cpu"
SEED = 42
INSTRUCTION = "You are an experienced radiologist. You are being given radiology images along with a short medical diagnosis. Generate a descriptive caption that highlights the location, nature and severity of the abnormality of the radiology image."
LEARNING_RATE = 2e-6
TRAIN_PARAMS = {
    'epochs': 50,
    'log_interval': 10,
    'dataloader': {
        'batch_size': 4,
        'shuffle': True,
        'num_workers': 4,
    },
}
VALID_PARAMS = {
    'epochs': 1,
    'log_interval': 10,
    'dataloader': {
        'batch_size': 4,
        'shuffle': False,
        'num_workers': 4,
    },
}
TEST_PARAMS = {
    'epochs': 1,
    'dataloader': {
        'batch_size': 4,
        'shuffle': False,
        'num_workers': 4,
    },
}