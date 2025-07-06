# # ===============================
# #     InstructBLIP FineTune
# # ===============================
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import pandas as pd
import torch
from instructBLIP_config import (BEST_MODEL_PATH, DEVICE, GENERATIONS_PATH,
                                 INSTRUCTION, LEARNING_RATE, MODEL_NAME,
                                 RESULTS_PATH, SEED, TEST_PARAMS, TRAIN_PARAMS,
                                 VALID_PARAMS)
from instructBLIP_dataset import CustomVisionDataset
from instructBLIP_utils import load_imageclef_data, split_data
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoImageProcessor, AutoTokenizer,
                          InstructBlipForConditionalGeneration,
                          InstructBlipProcessor)


class InstructBLIP:
    def __init__(self):
        """Initialize the InstructBLIP model."""
        print("\n[INIT] Initializing InstructBLIP model...")
        self.device = DEVICE
        print(f"[INIT] Using device: {self.device}")

    def _prepare_labels(self, captions, processor):
        """Prepare tokenized captions for model training/evaluation."""
        labels = processor.tokenizer(captions, padding="max_length", max_length=40, truncation=True, return_tensors="pt")
        labels["input_ids"] = torch.tensor([[-100 if x == processor.tokenizer.pad_token_id else x for x in ids.tolist()] 
                                           for ids in labels["input_ids"]])
        return labels.input_ids.clone().detach().to(self.device)

    def _process_batch(self, images, captions, instruction_, processor):
        """Process image-caption pairs and return model inputs."""
        instruction = [instruction_] * len(captions)
        inputs = processor(images=images, text=instruction, return_tensors="pt", do_rescale=False).to(self.device)
        labels = self._prepare_labels(captions, processor)
        return inputs, labels

    def train(self, loader, model, processor, epoch, instruction_, optimizer):
        """Training loop for InstructBLIP fine-tuning."""
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for step_idx, (images, captions, _) in tqdm(enumerate(loader), desc=f"Training Epoch {epoch}", total=len(loader)):
            inputs, labels = self._process_batch(images, captions, instruction_, processor)

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            running_loss += loss.item()

            if step_idx % TRAIN_PARAMS['log_interval'] == 0:
                print(f"[TRAIN] Epoch {epoch}, Step {step_idx}, Loss: {loss.item():.4f}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss = running_loss / len(loader)
        print(f"[TRAIN] Epoch {epoch} completed. Average Loss: {epoch_loss:.4f}")

        # Save the model every 5 epochs
        if epoch % 5 == 0:
            print(f"[MODEL] Saving model checkpoint for Epoch {epoch}...")
            torch.save(model.state_dict(), f"{BEST_MODEL_PATH}_epoch_{epoch}.pt")
            print(f"[MODEL] Model saved to {BEST_MODEL_PATH}_epoch_{epoch}.pt")

    def validate(self, loader, model, processor, epoch, instruction_):
        """Validation loop for InstructBLIP fine-tuning."""
        model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for step_idx, (images, captions, _) in tqdm(enumerate(loader), desc=f"Validating Epoch {epoch}", total=len(loader)):
                inputs, labels = self._process_batch(images, captions, instruction_, processor)

                outputs = model(**inputs, labels=labels)
                loss = outputs.loss
                running_loss += loss.item()

                if step_idx % VALID_PARAMS['log_interval'] == 0:
                    print(f"[VALID] Epoch {epoch}, Step {step_idx}, Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(loader)
        print(f"[VALID] Epoch {epoch} completed. Average Loss: {epoch_loss:.4f}")

        # Early stopping logic
        self._early_stop(epoch_loss, model)

    def _early_stop(self, epoch_loss, model):
        """Check if early stopping criteria are met and save the best model."""
        if epoch_loss < self.best_loss:
            self.best_loss = epoch_loss
            self.early_stopping_counter = 0
            print(f"[EARLY STOP] New best loss: {self.best_loss:.4f}, saving model...")
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"[MODEL] Best model saved to {BEST_MODEL_PATH}")
        else:
            self.early_stopping_counter += 1
            print(f"[EARLY STOP] No improvement, counter: {self.early_stopping_counter}/3")

    def test(self, loader, model, processor, instruction_):
        """Testing loop for InstructBLIP fine-tuning."""
        model.eval()
        predictions = []

        with torch.no_grad():
            for step_idx, (images, captions, ids) in tqdm(enumerate(loader), desc=f"Testing", total=len(loader)):
                inputs = processor(images=images, text=[instruction_] * len(captions), return_tensors="pt").to(self.device)

                outputs = model.generate(
                    **inputs,
                    do_sample=False,
                    num_beams=5,
                    max_length=120,
                    min_length=5
                )
                generated_texts = processor.batch_decode(outputs, skip_special_tokens=True)
                predictions.extend(generated_texts)

        print(f"[TEST] Generated {len(predictions)} predictions.")
        return predictions

    def main(self):
        """Main entry point for InstructBLIP fine-tuning."""
        print("\n[MAIN] Starting InstructBLIP fine-tuning script...")
        
        # Set random seed for reproducibility
        print(f"\n[SETUP] Setting random seed: {SEED}")
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        # Load data
        print("\n[DATA] Loading ImageCLEF dataset...")
        captions_train, captions_valid, captions_test = load_imageclef_data()
        train_ids, dev_ids, test_ids = split_data(captions_train, captions_valid, captions_test)

        # Initialize model and processor
        print("\n[MODEL] Load pretrained InstructBLIP model...")
        self.model = InstructBlipForConditionalGeneration.from_pretrained(MODEL_NAME).to(self.device)
        self.processor = InstructBlipProcessor.from_pretrained(MODEL_NAME)

        # Freeze model layers
        self._freeze_model_layers()

        # Print frozen/trainable parameter counts
        self._print_param_counts()

        # Initialize datasets and dataloaders
        image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k", use_fast=True)
        train_dataset = CustomVisionDataset(captions_train, train_ids, image_processor, 'train')
        valid_dataset = CustomVisionDataset(captions_valid, dev_ids, image_processor, 'valid')
        test_dataset = CustomVisionDataset(captions_test, test_ids, image_processor, 'test')

        train_dataloader = DataLoader(train_dataset, **TRAIN_PARAMS['dataloader'])
        valid_dataloader = DataLoader(valid_dataset, **VALID_PARAMS['dataloader'])
        test_dataloader = DataLoader(test_dataset, **TEST_PARAMS['dataloader'])

        # Initialize optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
        print(f"[OPTIMIZER] Using learning rate: {LEARNING_RATE}")

        # Training loop
        print("\n[TRAINING] Starting instruction-based fine-tuning...")
        self.best_loss = float('inf')
        self.early_stopping_counter = 0

        for epoch in range(1, TRAIN_PARAMS['epochs'] + 1):
            print(f"\n[TRAINING] Epoch {epoch}/{TRAIN_PARAMS['epochs']}")
            self.train(train_dataloader, self.model, self.processor, epoch, INSTRUCTION, optimizer)

            print(f"\n[VALIDATION] Epoch {epoch}/{TRAIN_PARAMS['epochs']}")
            self.validate(valid_dataloader, self.model, self.processor, epoch, INSTRUCTION)

            # Early stopping check
            if self.early_stopping_counter >= 3 or epoch == TRAIN_PARAMS['epochs'] - 1:
                # Clean up memory
                del self.model
                del optimizer
                torch.cuda.empty_cache()

                print("[MODEL] Training complete, loading best model for testing...")
                self._load_best_model()

                break

        # Test the best model
        print('\n[TESTING] Generating predictions on test set...')
        predictions = self.test(test_dataloader, self.model, self.processor, INSTRUCTION)
        self._save_results(predictions, test_ids)


    def _freeze_model_layers(self):
        """Freeze model layers based on specified criteria."""
        print("\n[FREEZE] Freezing model layers...")
        for param in self.model.vision_model.encoder.layers.parameters():
            param.requires_grad = False
        for param in self.model.language_model.encoder.parameters():
            param.requires_grad = False
        for i, param in enumerate(self.model.language_model.decoder.parameters()):
            if i <= 334:
                param.requires_grad = False
        for i, param in enumerate(self.model.qformer.encoder.layer.parameters()):
            if i <= 190:
                param.requires_grad = False

    def _print_param_counts(self):
        """Print the count of frozen and trainable parameters."""
        trainable = sum(1 for param in self.model.parameters() if param.requires_grad)
        frozen = sum(1 for param in self.model.parameters() if not param.requires_grad)
        print(f'\n[MODEL] Trainable parameters: {trainable}')
        print(f'[MODEL] Frozen parameters: {frozen}')

    def _load_best_model(self):
        """Load the best model after training."""
        best_model = InstructBlipForConditionalGeneration.from_pretrained(MODEL_NAME)
        state_dict = torch.load(BEST_MODEL_PATH, map_location='cpu')
        best_model.load_state_dict(state_dict)
        self.model = best_model.to(self.device)

    def _save_results(self, predictions, test_ids):
        """Save predictions and results to files."""
        print("\n[SAVE] Saving best model and results...")
        with open(RESULTS_PATH, 'w') as out_test:
            for i, pred in enumerate(predictions):
                out_test.write(f'{test_ids[i]}|{pred}\n')
        print(f"[RESULTS] Saved to {RESULTS_PATH}")

        # Save generations dataframe
        final_df = pd.DataFrame({'ID': test_ids, 'Caption': predictions})
        final_df['Caption'] = final_df['Caption'].apply(lambda x: x.replace('\n', ' ').replace('\r', ' ').strip())
        final_df.to_csv(GENERATIONS_PATH, index=False)
        print(f"[RESULTS] Generations saved to {GENERATIONS_PATH}")


if __name__ == "__main__":
    instruct_blip = InstructBLIP()
    instruct_blip.main()


