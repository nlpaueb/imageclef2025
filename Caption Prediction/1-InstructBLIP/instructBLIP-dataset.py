"""
Custom Vision Dataset for ImageCLEF medical image captioning
Handles loading and preprocessing of images and captions for InstructBLIP model
"""

import os
from typing import Dict, List, Tuple

import torch
from instructBLIP_config import DATASET_IMAGES_PATH
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor


class CustomVisionDataset(torch.utils.data.Dataset):
    """Custom PyTorch Dataset for medical image captioning"""
    
    def __init__(self, 
                 captions_dict: Dict[str, str], 
                 image_ids: List[str], 
                 image_processor: AutoImageProcessor, 
                 mode: str = 'train') -> None:
        """
        Initialize the dataset
        
        Args:
            captions_dict: Dictionary mapping image IDs to captions
            image_ids: List of image IDs to include in this dataset
            image_processor: HuggingFace image processor for normalization
            mode: Dataset mode ('train', 'validation', or 'test')
        """
        print(f"\n[Dataset] Initializing {mode} dataset...")
        print(f"[Dataset] Found {len(image_ids)} samples")
        
        self.mode = mode
        self.captions_dict = captions_dict
        self.image_ids = image_ids
        self.image_processor = image_processor
        
        # Image directory configuration
        self.images_base_path = DATASET_IMAGES_PATH
        print(f"[Dataset] Loading images from: {self.images_base_path}")
        
        # Setup transforms based on dataset mode
        self._setup_transforms()
        
    def _setup_transforms(self) -> None:
        """Configure appropriate image transformations based on dataset mode"""
        print(f"[Dataset] Configuring {self.mode} transforms...")
        
        # Training transforms - includes augmentation
        if self.mode == 'train':
            self._transforms = transforms.Compose([
                transforms.RandomRotation(30),  # Random rotation for augmentation
                transforms.Resize((224, 224)),  # Standard size for ViT
                transforms.ToTensor(),          # Convert to tensor
            ])
            print("[Dataset] Using training transforms with augmentation")
            
        # Validation transforms - same as training but deterministic
        elif self.mode == 'validation':
            self._transforms = transforms.Compose([
                transforms.RandomRotation(30),  # Kept rotation for consistency
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            print("[Dataset] Using validation transforms")
            
        # Test transforms - no augmentation
        else:
            self._transforms = transforms.Compose([
                transforms.Resize((224, 224)),  # Fixed size
                transforms.ToTensor(),
            ])
            print("[Dataset] Using test transforms (no augmentation)")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset"""
        return len(self.image_ids)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str, str]:
        """
        Get a single sample from the dataset
        
        Args:
            index: Integer index of the sample to retrieve
            
        Returns:
            tuple: (image_tensor, caption, image_id)
        """
        try:
            # Get image ID and corresponding caption
            image_id = self.image_ids[index]
            caption = self.captions_dict[image_id]
            
            # Load and preprocess image
            image_path = os.path.join(self.images_base_path, f"{image_id}.jpg")
            image = Image.open(image_path).convert('RGB')  # Ensure RGB format
            
            # Apply transforms
            image_tensor = self._transforms(image)
            
            return image_tensor, caption, image_id
            
        except Exception as e:
            print(f"\n[ERROR] Failed to load sample {index} (ID: {image_id})")
            print(f"Error details: {str(e)}")
            raise

    def __repr__(self) -> str:
        """String representation of the dataset"""
        return (f"CustomVisionDataset(mode={self.mode}, "
                f"samples={len(self)}, "
                f"transforms={self._transforms})")