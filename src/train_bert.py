import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)


# ================= CONFIG =================
MODEL_NAME = "bert-base-uncased"
DATA_PATH = "/Users/mac/Desktop/domain_adaptive_transformer_customer_feedback/data/processed/reviews_clean.csv"
OUTPUT_DIR = "/Users/mac/Desktop/domain_adaptive_transformer_customer_feedback/models/bert_sentiment"

MAX_SAMPLES = 50000
MAX_LENGTH = 128
BATCH_SIZE = 8
EPOCHS = 1
# =========================================

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# ===== Subset for fast training =====
df = df.sample(n=MAX_SAMPLES, random_state=42).reset_index(drop=True)
print("Using subset size:", len(df))

# Encode labels
label_encoder = LabelEncoder()
df["label_id"] = label_encoder.fit_transform(df["label"])

# Train / Validation split
train_df, val_df = train_test_split(
    df,
    test_size=0.1,
    random_state=42,
    stratify=df["label_id"]
)

train_ds = Dataset.from_pandas(train_df)
val_ds = Dataset.from_pandas(val_df)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

train_ds = train_ds.rename_column("label_id", "labels")
val_ds = val_ds.rename_column("label_id", "labels")

train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
val_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Model
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_encoder.classes_)
)

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

# Training arguments (خفيفة على Mac)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    logging_steps=100,
    load_best_model_at_end=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print("🚀 Starting BERT training...")
trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ BERT sentiment model saved to:", OUTPUT_DIR)
print("Labels:", list(label_encoder.classes_))