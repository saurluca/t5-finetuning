# %%
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import pandas as pd

# Some Environment Setup
OUTPUT_DIR = "output"
LOG_DIR = "logs"
CACHE_DIR = "cache"
DATA_PATH = "data/carousel_dataset.csv"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("t5-small", cache_dir=CACHE_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small", cache_dir=CACHE_DIR)

# Check if GPU is available and move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

input_text = "Translate English to German: The house is wonderful."

# Encode and generate response
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
output_ids = model.generate(input_ids, max_new_tokens=20)[0]

# Decode and print the output text
output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
print(output_text)

# %%

input_text = "question: what programming languauage is this?  code: `df <- read.csv('data.csv'); summary(df)`"

# Encode and generate response
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
output_ids = model.generate(input_ids, max_new_tokens=20)[0]

# Decode and print the output text
output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
print(output_text)

# Explicitly free demo model/tokenizer and clear GPU cache before training
del model
del tokenizer
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# %%

# read in data
df = pd.read_csv(DATA_PATH)
# drop rows with missing key fields to avoid type issues during tokenization
df = df.dropna(subset=["question", "correct_answer", "ground_truth"]).reset_index(
    drop=True
)
# filter for relevant columns
df = df[["question", "correct_answer", "student_answer", "ground_truth"]]
# print the first 5 rows
df.head()

# convert 0,1 to true, false for both answers and ground truth
df["correct_answer"] = df["correct_answer"].apply(
    lambda x: "true" if (str(x).strip() in {"1", "true", "True"}) else "false"
)
df["ground_truth"] = df["ground_truth"].apply(
    lambda x: "true" if (str(x).strip() in {"1", "true", "True"}) else "false"
)

# %%

# print number of unique questions in question column.
print("Number of unique questions:", len(df["question"].unique()))
print("average number of samples per question:", df.groupby("question").size().mean())

# %%
# Split into train, valid, test sets, sampling equally from all questions

# Get all unique questions
questions = df["question"].unique()

# Shuffle questions for randomness
questions = pd.Series(questions).sample(frac=1, random_state=42).reset_index(drop=True)

# Calculate split sizes
n_questions = len(questions)
n_train = int(0.6 * n_questions)
n_valid = int(0.2 * n_questions)
n_test = n_questions - n_train - n_valid

# Assign questions to splits
train_questions = questions[:n_train]
valid_questions = questions[n_train : n_train + n_valid]
test_questions = questions[n_train + n_valid :]

# Subset the dataframe for each split, ensuring all samples for a question are in the same split
train_df = df[df["question"].isin(train_questions)].reset_index(drop=True)
valid_df = df[df["question"].isin(valid_questions)].reset_index(drop=True)
test_df = df[df["question"].isin(test_questions)].reset_index(drop=True)

print(f"Train set: {len(train_df)} samples, {len(train_questions)} questions")
print(f"Valid set: {len(valid_df)} samples, {len(valid_questions)} questions")
print(f"Test set: {len(test_df)} samples, {len(test_questions)} questions")


# %%

from transformers import AutoTokenizer
from datasets import Dataset

tokenizer = AutoTokenizer.from_pretrained("t5-small")


def preprocess_function(examples, tokenizer):
    prefix = (
        "main_question: is the answer correct for the following question?  question: "
    )
    prefix_2 = "answer: "
    inputs = [
        prefix + str(question) + " " + prefix_2 + str(answer)
        for question, answer in zip(examples["question"], examples["correct_answer"])
    ]
    model_inputs = tokenizer(
        inputs, max_length=256, truncation=True, padding="max_length"
    )

    # Ensure labels are strings; tokenizer expects str or list[str]
    targets = [str(t) for t in examples["ground_truth"]]
    labels = tokenizer(
        text_target=targets,
        max_length=4,
        truncation=True,
        padding="max_length",
        add_special_tokens=True,
    )

    # Replace pad tokens in labels with -100 so they are ignored by loss
    pad_id = tokenizer.pad_token_id
    processed_labels = []
    for seq in labels["input_ids"]:
        processed_labels.append([tok if tok != pad_id else -100 for tok in seq])
    model_inputs["labels"] = processed_labels
    return model_inputs


# Convert pandas DataFrame to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
valid_dataset = Dataset.from_pandas(valid_df)
test_dataset = Dataset.from_pandas(test_df)

# Now use the Dataset's map method with the correct parameters
tokenized_train = train_dataset.map(
    preprocess_function,
    fn_kwargs={"tokenizer": tokenizer},
    batched=True,
    batch_size=16,
    load_from_cache_file=True,
)

tokenized_valid = valid_dataset.map(
    preprocess_function,
    fn_kwargs={"tokenizer": tokenizer},
    batched=True,
    batch_size=16,
    load_from_cache_file=True,
)

tokenized_test = test_dataset.map(
    preprocess_function,
    fn_kwargs={"tokenizer": tokenizer},
    batched=True,
    batch_size=16,
    load_from_cache_file=True,
)

# %%

import torch
import mlflow
from transformers import (
    Trainer,
    TrainingArguments,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
)

# Check if CUDA is available and set the device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_name = "t5-small"  # Or another T5 variant
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

# Define training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    # evaluation_strategy="steps",
    # eval_steps=50,
    logging_strategy="steps",
    logging_steps=50,
    save_strategy="epoch",
    logging_dir=LOG_DIR,
    learning_rate=2e-5,
    fp16=True,
)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    data_collator=data_collator,
)

# Start training
with mlflow.start_run():
    trainer.train()

# %%

import gc

del model
del tokenizer
del trainer
torch.cuda.empty_cache()
gc.collect()

# %%

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import os
import re
from math import ceil

# Load latest model checkpoint
# List all subdirectories in OUTPUT_DIR
subdirs = next(os.walk(OUTPUT_DIR))[1]

# Filter out directories that match the checkpoint pattern
checkpoint_dirs = [d for d in subdirs if re.match(r"checkpoint-\d+", d)]

# Find the latest checkpoint (highest number)
latest_checkpoint = max(checkpoint_dirs, key=lambda d: int(d.split("-")[-1]))

# Complete path to the latest checkpoint
latest_checkpoint_path = os.path.join(OUTPUT_DIR, latest_checkpoint)

# Load model from the latest checkpoint
model = AutoModelForSeq2SeqLM.from_pretrained(latest_checkpoint_path)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("t5-small")

# Check if GPU is available and move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


def build_inputs(questions, answers):
    prefix = (
        "main_question: is the answer correct for the following question?  question: "
    )
    suffix = " answer: "
    return [prefix + str(q) + suffix + str(a) for q, a in zip(questions, answers)]


# Show a few train examples
print("\n=== Train examples ===")
sample_train = train_df.sample(n=min(5, len(train_df)), random_state=42)
train_inputs = build_inputs(
    sample_train["question"].tolist(), sample_train["correct_answer"].tolist()
)
enc = tokenizer(train_inputs, padding=True, truncation=True, return_tensors="pt").to(
    device
)
with torch.no_grad():
    gen_ids = model.generate(enc["input_ids"], max_new_tokens=2, num_beams=1)
preds = [tokenizer.decode(ids, skip_special_tokens=True) for ids in gen_ids]
for inp, pred, gt in zip(train_inputs, preds, sample_train["ground_truth"].tolist()):
    print(f"Input: {inp}")
    print(f"Pred:  {pred} | Ground Truth: {gt}")

# Show a few test examples
print("\n=== Test examples ===")
sample_test = test_df.sample(n=min(5, len(test_df)), random_state=0)
test_inputs_preview = build_inputs(
    sample_test["question"].tolist(), sample_test["correct_answer"].tolist()
)
enc = tokenizer(
    test_inputs_preview, padding=True, truncation=True, return_tensors="pt"
).to(device)
with torch.no_grad():
    gen_ids = model.generate(enc["input_ids"], max_new_tokens=2, num_beams=1)
preds = [tokenizer.decode(ids, skip_special_tokens=True) for ids in gen_ids]
for inp, pred, gt in zip(
    test_inputs_preview, preds, sample_test["ground_truth"].tolist()
):
    print(f"Input: {inp}")
    print(f"Pred:  {pred} | GT: {gt}")

# Compute accuracy on full test set in batches
print("\n=== Test accuracy ===")
batch_size = 64
num_rows = len(test_df)
num_batches = ceil(num_rows / batch_size)
num_correct = 0
num_total = 0
for b in range(num_batches):
    sl = slice(b * batch_size, min((b + 1) * batch_size, num_rows))
    batch_questions = test_df.loc[sl, "question"].astype(str).tolist()
    batch_answers = test_df.loc[sl, "correct_answer"].astype(str).tolist()
    batch_targets = test_df.loc[sl, "ground_truth"].astype(str).tolist()
    inputs_text = build_inputs(batch_questions, batch_answers)
    enc = tokenizer(inputs_text, padding=True, truncation=True, return_tensors="pt").to(
        device
    )
    with torch.no_grad():
        gen_ids = model.generate(enc["input_ids"], max_new_tokens=2, num_beams=1)
    batch_preds = [tokenizer.decode(ids, skip_special_tokens=True) for ids in gen_ids]
    # Normalize predictions and targets to simple lowercase tokens like "true"/"false"
    norm = lambda s: str(s).strip().lower()
    for p, t in zip(batch_preds, batch_targets):
        num_correct += int(norm(p) == norm(t))
        num_total += 1

accuracy = num_correct / max(1, num_total)
print(f"Accuracy on test set: {accuracy:.4f}")
