import torch

# Set constants
MAX_LENGTH = 300
L_RATE = 1e-4
TRAIN_BATCH_SIZE = 2
TEST_BATCH_SIZE = 4
PER_DEVICE_EVAL_BATCH = 1
NUM_EPOCHS = 2
MODEL_NAME = "google/flan-t5-large"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_WANDB = False  # Set to True if you want to use Weights & Biases

# Set paths
IMAGE_DIR = '/path/to/your/images'                            # Directory containing all input images
TRAIN_INPUTS_PATH = '/path/to/your/train/merged/captions.csv' # Merged model captions for training (from merge_captions.py)
VALID_INPUTS_PATH = '/path/to/your/valid/merged/captions.csv' # Merged model captions for validation (from merge_captions.py)
TEST_INPUTS_PATH = '/path/to/your/test/merged/captions.csv'   # Merged model captions for testing (from merge_captions.py)

TRAIN_TARGETS_PATH = '/path/to/your/train/captions.csv'       # Ground-truth target captions for training
VALID_TARGETS_PATH = '/path/to/your/valid/captions.csv'       # Ground-truth target captions for validation

LOGS_DIR = "/path/to/your/logs"                               # Directory to store training logs
OUTPUT_DIR = "/path/to/your/output"                           # Directory for model checkpoints and outputs
FINAL_MODEL_PATH = "/path/to/your/final/model"                # Final saved model after training

TEST_OUTPUT_TXT_PATH = "/path/to/your/test/output.txt"        # File to save test predictions (plain text)
TEST_OUTPUT_CSV_PATH = "/path/to/your/test/output.csv"        # File to save test predictions (CSV format)

