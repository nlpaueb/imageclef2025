# ===========================
#   InstructBLIP Training Script
#     (Mixed CE + RL Loss)
# ===========================
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration, AutoImageProcessor
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
import nltk
nltk.download('stopwords')
from mixer_dataset import CustomVisionDataset
from sklearn import preprocessing

from evaluation.evaluator import load_evaluation_models, compute_weighted_average

from mixer_utils import load_imageclef_data, split_data

from mixer_config import (
    DEVICE, SEED, MODEL_NAME, INSTRUCTION,
    CHECKPOINT_PATH, BEST_MODEL_PATH,
    TRAIN_PARAMS, VALID_PARAMS, TEST_PARAMS, LEARNING_RATE,
    RESULTS_PATH, GENERATIONS_PATH,
)

class InstructBLIPMixer:
    def __init__(self):
        """Initialize the InstructBLIP Mixer model."""
        print("\n[INIT] Initializing Mixer...")
        self.device = DEVICE
        print(f"[INIT] Using device: {self.device}")

    def train(self, epoch, instruction_, model, processor, device, loader, optimizer, eval_models, max_alpha=1.0):
        """Training loop with mixed loss (CE + SCST)"""
        model.train()
        running_loss = 0
        batch_counter = 0
        accum_iter = 8  

        # Progressive alpha scheduling
        alpha = max_alpha * ((epoch) / TRAIN_PARAMS['epochs']) 
        print(f"\n[Training] Epoch {epoch}: Mixed Loss Alpha = {alpha:.4f}")

        optimizer.zero_grad()

        for step_idx, data in tqdm(enumerate(loader, 0), desc=f"Training Epoch {epoch}", total=len(loader)):
            images, captions, ids = data
            batch_counter += 1

            # Prepare instructions for batch
            instruction = [instruction_ for _ in range(len(captions))]
            inputs = processor(images=images, text=instruction, return_tensors="pt").to(device)

            # ===== 1. Generate greedy outputs =====
            with torch.no_grad():
                greedy_outputs = model.generate(
                    **inputs,
                    do_sample=False,
                    max_length=60,
                    min_length=5,
                )

            greedy_captions = processor.batch_decode(greedy_outputs, skip_special_tokens=True)
            print(f"[Debug] Greedy sample: {greedy_captions[0][:100]}...")

            # ===== 2. Generate sampled outputs =====
            with torch.no_grad():
                sampled_outputs = model.generate(
                    **inputs,
                    do_sample=True,
                    top_p=0.95,
                    temperature=0.6,
                    max_length=60,
                    min_length=5,
                )
                # sampled_outputs = model.generate(
                #     **inputs,
                #     num_beams=6,
                #     num_beam_groups=3,
                #     diversity_penalty=1.2,
                #     max_length=60,
                #     min_length=5,
                # )

            sampled_captions = processor.batch_decode(sampled_outputs, skip_special_tokens=True)
            print(f"[Debug] Sampled sample: {sampled_captions[0][:100]}...")

            # ===== 3. Calculate rewards =====
            rewards_sampled = []
            rewards_greedy = []

            for sampled, greedy, gt, id_ in zip(sampled_captions, greedy_captions, captions, ids):
                if not sampled or sampled.strip() == "":
                    sampled = "No caption"
                if not greedy or greedy.strip() == "":
                    greedy = "No caption"
                if not gt or gt.strip() == "":
                    gt = "No ground truth"

                try:
                    reward_sampled, _ = compute_weighted_average([id_], [sampled], [gt], eval_models)
                    reward_greedy, _ = compute_weighted_average([id_], [greedy], [gt], eval_models)
                except RuntimeError as e:
                    print(f"[Warning] Skipping ID {id_} due to AlignScore failure: {e}")
                    reward_sampled = 0.0
                    reward_greedy = 0.0

                print(f"[Debug] ID {id_} | Sampled reward: {reward_sampled:.4f} | Greedy reward: {reward_greedy:.4f}")
                
                rewards_sampled.append(reward_sampled)
                rewards_greedy.append(reward_greedy)

            rewards_sampled = torch.tensor(rewards_sampled, dtype=torch.float32).to(device)
            rewards_greedy = torch.tensor(rewards_greedy, dtype=torch.float32).to(device)

            # ===== 4. Calculate Cross Entropy Loss =====
            labels_gt = processor.tokenizer(
                captions,
                padding="max_length",
                max_length=60,
                truncation=True,
                return_tensors="pt"
            ).input_ids.to(device)

            labels_gt = torch.where(
                labels_gt == processor.tokenizer.pad_token_id,
                torch.full_like(labels_gt, -100),
                labels_gt
            )

            outputs_ce = model(**inputs, labels=labels_gt)
            ce_loss = outputs_ce.loss
            print(f"[Debug] CE loss: {ce_loss.item():.4f}")

            # ===== 5. Calculate SCST Loss =====
            labels_sampled = processor.tokenizer(
                sampled_captions,
                padding="max_length",
                max_length=60,
                truncation=True,
                return_tensors="pt"
            ).input_ids.to(device)

            labels_sampled = torch.where(
                labels_sampled == processor.tokenizer.pad_token_id,
                torch.full_like(labels_sampled, -100),
                labels_sampled
            )

            outputs_rl = model(**inputs, labels=labels_sampled)
            log_probs = -outputs_rl.loss.expand_as(rewards_sampled)

            advantage = (rewards_sampled - rewards_greedy).detach()

            scst_loss = (advantage * log_probs).mean()
            print(f"[Debug] SCST loss: {scst_loss.item():.4f}")

            # ===== 6. Combine losses =====
            total_loss = (1 - alpha) * ce_loss + alpha * scst_loss
            print(f"[Debug] Components - CE: {(1-alpha)*ce_loss.item():.4f}, SCST: {alpha*scst_loss.item():.4f}")

            # Normalize loss for accumulation
            total_loss = total_loss / accum_iter
            total_loss.backward()

            # Step optimizer every accum_iter steps or at end of epoch
            if (step_idx + 1) % accum_iter == 0 or (step_idx + 1) == len(loader):
                # Gradient checking
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            running_loss += total_loss.item() * accum_iter  # reverse the /accum_iter for correct reporting

            if step_idx % 100 == 0:  
                print(f"[Training] Epoch {epoch+1} | Step {step_idx+1} | "
                    f"Loss: {(total_loss.item() * accum_iter):.4f} (CE: {ce_loss.item():.4f}, "
                    f"SCST: {scst_loss.item():.4f}) | "
                    f"Reward diff: {(rewards_sampled.mean()-rewards_greedy.mean()).item():.4f}")

        avg_loss = running_loss / batch_counter
        print(f"[Training] Epoch {epoch} Complete | Avg Loss: {avg_loss:.4f}")

        # Save model checkpoint
        print(f"[Training] Saving model checkpoint for epoch {epoch}...")
        checkpoint_path = CHECKPOINT_PATH + f"checkpoint_epoch_{epoch}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"[Training] Model checkpoint saved to {checkpoint_path}")

    def validate(self, epoch, instruction_, model, processor, device, loader):
        """Validation loop"""
        print(f"\n[Validation] Starting validation for epoch {epoch}...")
        model.eval()
        val_batch_counter = 0
        running_val_loss = 0

        with torch.no_grad():
            for _, data in tqdm(enumerate(loader, 0), desc="Validating", total=len(loader)):
                val_batch_counter += 1
                image, caption, ids = data

                instruction = [instruction_ for _ in range(len(caption))]
                inputs = processor(images=image, text=instruction, return_tensors="pt").to(device)

                # Prepare labels
                labels = processor.tokenizer(
                    caption, 
                    padding="max_length", 
                    max_length=60, 
                    truncation=True, 
                    return_tensors="pt"
                )
                labels["input_ids"] = torch.tensor([
                    [-100 if x == processor.tokenizer.pad_token_id else x 
                     for x in labels["input_ids"][i].tolist()] 
                    for i in range(len(caption))
                ])
                labels = torch.tensor(labels.input_ids).to(device)

                val_outputs = model(**inputs, labels=labels)
                val_loss = val_outputs.loss
                running_val_loss += val_loss.item()
            
            # Clean up
            del inputs, labels

            epoch_val_loss = running_val_loss / val_batch_counter
            print(f'[Validation] Epoch: {epoch} | Val Loss: {epoch_val_loss:.4f} | Batches: {val_batch_counter}')
            
            # Early stopping check
            if epoch_val_loss < self.best_loss:
                self.best_loss = epoch_val_loss
                self.early_stopping_counter = 0
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                print(f"[Validation] New best model saved with loss: {epoch_val_loss:.4f}")
            else:
                self.early_stopping_counter += 1
                print(f"[Validation] Early stopping counter: {self.early_stopping_counter}/3")

    def test(self, instruction_, model, processor, device, loader, tokenizer):
        """Testing loop for generating predictions"""
        print(f"\n[Testing] Starting testing...")
        model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for step_idx, data in tqdm(enumerate(loader, 0), desc="Testing", total=len(loader)):
                image, caption, _ = data
                instruction = [instruction_ for _ in range(len(caption))]
                
                inputs = processor(images=image, text=instruction, return_tensors="pt").to(device)

                # Generate outputs
                outputs = model.generate(
                    **inputs,
                    do_sample=False,
                    num_beams=4,
                    max_length=60,
                    min_length=5,
                )

                generated_text = processor.batch_decode(outputs, skip_special_tokens=True)
                print(f"[Testing] Generated text sample: {generated_text[0][:50]}...")
                
                if step_idx % 100 == 0:
                    print(f'[Testing] Completed {step_idx} batches')
                    
                predictions.extend(generated_text)
                actuals.extend(caption)
        
        return predictions, actuals
    
    def _load_best_model(self):
        """Load the best model after training."""
        best_model = InstructBlipForConditionalGeneration.from_pretrained(MODEL_NAME)
        state_dict = torch.load(BEST_MODEL_PATH, map_location='cpu')
        best_model.load_state_dict(state_dict)
        self.model = best_model.to(self.device)

    def main(self):
        """Main entry point for InstructBLIP Mixer fine-tuning."""
        print("\n[MAIN] Starting InstructBLIP Mixer training pipeline...")

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

        # Initialize evaluation models
        print("\n[MODEL] Loading evaluation models...")
        self.evaluation_models = load_evaluation_models()

        # Freeze layers
        print("\n[MODEL] Freezing layers...")
        for i, param in enumerate(self.model.vision_model.encoder.layers.parameters()):
            param.requires_grad = False

        for i, param in enumerate(self.model.language_model.encoder.parameters()):
            param.requires_grad = False
        
        # Partial freeze of decoder
        c = 0
        for i, param in enumerate(self.model.language_model.decoder.parameters()):
            if i <= 334:
                param.requires_grad = False
            c += 1

        # Partial freeze of qformer
        c2 = 0
        for i, param in enumerate(self.model.qformer.encoder.layer.parameters()):
            c2 += 1
            if i <= 190:
                param.requires_grad = False

        # Print frozen/trainable parameter counts
        true_, false_ = 0, 0
        for param in self.model.parameters():   
            if param.requires_grad:
                true_ += 1
            else:
                false_ += 1
            
        print(f'\n[MODEL] Trainable parameters: {true_}')
        print(f'[MODEL] Frozen parameters: {false_}')

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

        # Initialize datasets
        print("\n[DATA] Creating datasets...")
        image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k", use_fast=True)
        self.train_dataset = CustomVisionDataset(captions_train, train_ids, image_processor, 'train')
        self.val_dataset = CustomVisionDataset(captions_valid, dev_ids, image_processor, 'validation')
        self.test_dataset = CustomVisionDataset(captions_test, test_ids, image_processor, 'test')

        # Training loop
        print("\n[TRAINING] Starting instruction-based fine-tuning...")
        self.best_loss = float('inf')
        self.early_stopping_counter = 0

        for epoch in range(1, TRAIN_PARAMS['epochs'] + 1):
            print(f"\n[TRAINING] Epoch {epoch}/{TRAIN_PARAMS['epochs']}")
            self.train(epoch, INSTRUCTION, self.model, self.processor, self.device, train_dataloader, optimizer, self.evaluation_models)

            print(f"\n[VALIDATION] Epoch {epoch}/{TRAIN_PARAMS['epochs']}")
            self.validate(epoch, INSTRUCTION, self.model, self.processor, self.device, valid_dataloader)

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
        predictions, _ = self.test(INSTRUCTION, self.model, self.processor, self.device, test_dataloader, self.processor.tokenizer)
        
        # Save results
        print("\n[RESULTS] Saving predictions...")
        with open(RESULTS_PATH, 'w') as out_test:
            for i, pred in enumerate(predictions):
                out_test.write(test_ids[i] + '|' + pred + '\n')
        print(f"[RESULTS] Saved to {RESULTS_PATH}")

        # Save generations dataframe
        final_df = pd.DataFrame({'Generated Text': predictions})
        final_df.to_csv(GENERATIONS_PATH, index=False)
        print(f"[RESULTS] Generations saved to {GENERATIONS_PATH}")


if __name__ == '__main__':
    print("[START] Launching InstructBLIP Mixer training...")
    instruct_blip_mixer = InstructBLIPMixer()
    instruct_blip_mixer.main()