import os
import torch

# BEST_MODEL_PATH = '/path/to/your/best/model.pt'
# TEST_CAPTIONS_PATH = 'path/to/your/test/captions.csv'
# IMAGES_PATH = '/path/to/your/images/'
# OUTPUT_CSV_PATH = 'path/to/your/output/predictions.csv'

BEST_MODEL_PATH = '/media/SSD_2TB/imageclef2025/ippokratis-captioning/instruct-blip-ft/best-model.pt'
TEST_CAPTIONS_PATH = '/media/SSD_2TB/imageclef2025/test_images/test_ids.csv'
IMAGES_PATH = '/media/SSD_2TB/imageclef2025/test_images/original/'
OUTPUT_CSV_PATH = '/media/SSD_2TB/imageclef2025/repo-code/MedCLIP Reranker/temp.csv'
BATCH_SIZE = 4
NUM_RETURN_SEQUENCES = 4
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
MEDCLIP_CHECKPOINT = 'pretrained/medclip-vit'

INSTRUCTION = (
    "You are an experienced radiologist. You are being provided with radiology images along with a brief medical diagnosis. "
    "Generate a detailed and descriptive caption that highlights the location, nature, and severity of any abnormalities visible in the image."
)