import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

# ================= CONFIG =================
BASE_MODEL = "gpt2"
OUTPUT_DIR = "/Users/mac/Desktop/domain_adaptive_transformer_customer_feedback/models/gpt_lora_support"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

MAX_SAMPLES = 2000
MAX_LENGTH = 128
BATCH_SIZE = 4
EPOCHS = 2
LR = 2e-4
# =========================================

# --------- Toy Support Dataset (Demo) ---------
data = [
    {
        "text": "Customer Issue: I was charged twice for my order.\n"
                "Category: Billing\n"
                "Support Response: We apologize for the inconvenience. "
                "Please share your order ID so we can resolve this immediately."
    },
    {
        "text": "Customer Issue: The product arrived damaged.\n"
                "Category: Complaint\n"
                "Support Response: We’re sorry to hear that. "
                "We will arrange a replacement or refund as soon as possible."
    },
    {
        "text": "Customer Issue: How can I reset my password?\n"
                "Category: Technical Issue\n"
                "Support Response: You can reset your password by clicking on "
                "'Forgot Password' on the login page."
    },
] * 700

dataset = Dataset.from_list(data).shuffle(seed=42).select(range(MAX_SAMPLES))

# --------- Tokenizer ---------
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(batch):
    tokens = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

    # ⭐ مهم جدًا لـ GPT
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)

# --------- Data Collator (مهم للاستقرار) ---------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# --------- Model ---------
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
model.config.pad_token_id = tokenizer.eos_token_id
model.to(DEVICE)

# --------- LoRA Config ---------
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["c_attn"]
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# --------- Training Args ---------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=2,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    logging_steps=50,
    save_strategy="epoch",
    report_to="none",
    fp16=False,
    remove_unused_columns=False  # ⭐ مهم مع GPT
)

# --------- Trainer ---------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

print("🚀 Starting GPT LoRA fine-tuning...")
trainer.train()

# --------- Save ---------
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ GPT LoRA model saved to:", OUTPUT_DIR)