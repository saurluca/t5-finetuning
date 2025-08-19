# %%
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
import torch
from math import ceil
import os
import re
from tqdm import tqdm

# Some Environment Setup
OUTPUT_DIR = "output"
LOG_DIR = "logs"
CACHE_DIR = "cache"
MODEL_NAME = "t5-small"
MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 4
LEARNING_RATE = 2e-5
NUM_EPOCHS = 1
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
SEED = 42

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR).to(
    device
)

# Load IMDb dataset
dataset = load_dataset("imdb")
train_valid = dataset["train"].train_test_split(test_size=0.1, seed=SEED)
train_ds = train_valid["train"]
valid_ds = train_valid["test"]
test_ds = dataset["test"]

label_id_to_text = {0: "negative", 1: "positive"}


def preprocess_function(examples):
    inputs = ["imdb sentiment: " + str(x) for x in examples["text"]]
    targets = [label_id_to_text[int(label_value)] for label_value in examples["label"]]
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length",
    )
    labels = tokenizer(
        text_target=targets,
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding="max_length",
        add_special_tokens=True,
    )
    pad_id = tokenizer.pad_token_id
    processed_labels = []
    for seq in labels["input_ids"]:
        processed_labels.append([tok if tok != pad_id else -100 for tok in seq])
    model_inputs["labels"] = processed_labels
    return model_inputs


tokenized_train = train_ds.map(
    preprocess_function, batched=True, batch_size=32, load_from_cache_file=True
)
tokenized_valid = valid_ds.map(
    preprocess_function, batched=True, batch_size=32, load_from_cache_file=True
)
tokenized_test = test_ds.map(
    preprocess_function, batched=True, batch_size=32, load_from_cache_file=True
)

# %%
# Evaluate base model before training
print("\n=== Base Model Evaluation ===")
base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR).to(
    device
)
base_model.eval()


def evaluate_model(model, dataset, model_name):
    batch_size = 64
    num_rows = len(dataset)
    num_batches = ceil(num_rows / batch_size)
    num_correct = 0
    num_total = 0

    for b in tqdm(range(num_batches)):
        sl = slice(b * batch_size, min((b + 1) * batch_size, num_rows))
        batch_texts = dataset[sl]["text"]
        batch_labels = [
            label_id_to_text[int(label_value)] for label_value in dataset[sl]["label"]
        ]
        inputs_text = build_inputs(batch_texts)
        enc = tokenizer(
            inputs_text, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            gen_ids = model.generate(enc["input_ids"], max_new_tokens=2, num_beams=1)
        batch_preds = [
            tokenizer.decode(ids, skip_special_tokens=True) for ids in gen_ids
        ]
        for p, t in zip(batch_preds, batch_labels):
            num_correct += int(normalize_sentiment(p) == normalize_sentiment(t))
            num_total += 1

    accuracy = num_correct / max(1, num_total)
    print(f"{model_name} accuracy on test set: {accuracy:.4f}")
    return accuracy


# Evaluate base model
base_accuracy = evaluate_model(base_model, test_ds, "Base T5-small")
del base_model
torch.cuda.empty_cache()

# %%

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    logging_dir=LOG_DIR,
    learning_rate=LEARNING_RATE,
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
)

trainer.train()

# Save final model
trainer.save_model(OUTPUT_DIR)

# %%

# Load latest checkpoint for evaluation
subdirs = next(os.walk(OUTPUT_DIR))[1]
checkpoint_dirs = [d for d in subdirs if re.match(r"checkpoint-\d+", d)]
if checkpoint_dirs:
    latest_checkpoint = max(checkpoint_dirs, key=lambda d: int(d.split("-")[-1]))
    latest_checkpoint_path = os.path.join(OUTPUT_DIR, latest_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(latest_checkpoint_path).to(device)
else:
    model = AutoModelForSeq2SeqLM.from_pretrained(OUTPUT_DIR).to(device)


def normalize_sentiment(text: str) -> str:
    s = str(text).strip().lower()
    if "positive" in s:
        return "positive"
    if "negative" in s:
        return "negative"
    return s


def build_inputs(texts):
    return ["imdb sentiment: " + str(t) for t in texts]


# Evaluate accuracy on test set via generation
print("\n=== IMDb Test accuracy ===")
batch_size = 64
num_rows = len(test_ds)
num_batches = ceil(num_rows / batch_size)
num_correct = 0
num_total = 0
for b in tqdm(range(num_batches)):
    sl = slice(b * batch_size, min((b + 1) * batch_size, num_rows))
    batch_texts = test_ds[sl]["text"]
    batch_labels = [
        label_id_to_text[int(label_value)] for label_value in test_ds[sl]["label"]
    ]
    inputs_text = build_inputs(batch_texts)
    enc = tokenizer(inputs_text, padding=True, truncation=True, return_tensors="pt").to(
        device
    )
    with torch.no_grad():
        gen_ids = model.generate(enc["input_ids"], max_new_tokens=2, num_beams=1)
    batch_preds = [tokenizer.decode(ids, skip_special_tokens=True) for ids in gen_ids]
    for p, t in zip(batch_preds, batch_labels):
        num_correct += int(normalize_sentiment(p) == normalize_sentiment(t))
        num_total += 1

accuracy = num_correct / max(1, num_total)
print(f"Accuracy on IMDb test set: {accuracy:.4f}")

# Quick demo
print("\n=== Demo ===")
demo_texts = [
    "This movie was a delightful surprise with great performances.",
    "The plot was boring and I fell asleep halfway through.",
    "Average film, some good moments but overall forgettable.",
]
demo_inputs = build_inputs(demo_texts)
enc = tokenizer(demo_inputs, padding=True, truncation=True, return_tensors="pt").to(
    device
)
with torch.no_grad():
    gen_ids = model.generate(enc["input_ids"], max_new_tokens=2, num_beams=1)
demo_preds = [tokenizer.decode(ids, skip_special_tokens=True) for ids in gen_ids]
for t, p in zip(demo_texts, demo_preds):
    print(f"Text: {t}\nPredicted: {normalize_sentiment(p)}\n")
