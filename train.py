import torch
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset

MODEL_NAME = "google/flan-t5-small"
OUTPUT_DIR = "./models/humanizer"

# 1. Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

# 2. Load dataset (placeholder for now)
# We will replace this with HC3 / paraphrase later
dataset = load_dataset("quora")  # temporary, just to test pipeline

# 3. Preprocessing function
def preprocess(example):
    input_text = "Rewrite the following text in natural English:\n\n" + example["questions"]["text"][0]
    target_text = example["questions"]["text"][1]

    model_inputs = tokenizer(
        input_text,
        truncation=True,
        padding="max_length",
        max_length=256
    )

    labels = tokenizer(
        target_text,
        truncation=True,
        padding="max_length",
        max_length=256
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset["train"].map(preprocess, batched=False)

# 4. Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    num_train_epochs=1,
    learning_rate=5e-5,
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    report_to="none"
)

# 5. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

# 6. Train
trainer.train()

# 7. Save model
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
