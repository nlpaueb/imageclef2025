import wandb
from datasets import DatasetDict, Dataset
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration, 
    DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)
from fuser_utils import compute_metrics, load_and_preprocess_data, preprocess_function
from fuser_config import *

def train_model():
    """Train the FLAN-T5 model."""
    # Initialize Weights & Biases (wandb)
    if USE_WANDB:
        wandb.init(project='flanT5-2025')

    # Load and preprocess the data
    train_df, valid_df = load_and_preprocess_data()

    # Prepare datasets
    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)
    dataset = DatasetDict({'train': train_dataset, 'valid': valid_dataset})

    # Load the tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.to(DEVICE)

    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # Tokenize datasets
    tokenized_dataset = dataset.map(lambda x: preprocess_function(x, tokenizer))

    # Initialize training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        learning_rate=L_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH,
        save_strategy="epoch",
        save_total_limit=1,
        num_train_epochs=NUM_EPOCHS,
        predict_with_generate=True,
        logging_dir=LOGS_DIR
    )

    # Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["valid"],
        # tokenizer=tokenizer,
        # processing_class=preprocess_function,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()

    # Save the final model and tokenizer
    trainer.save_model(FINAL_MODEL_PATH)
    tokenizer.save_pretrained(FINAL_MODEL_PATH)

    # Finish the W&B run
    if USE_WANDB:
        wandb.finish()


if __name__ == "__main__":
    """Main function to initiate training."""
    train_model()
