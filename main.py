# %%
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from huggingface_hub import notebook_login

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

# %%

# read in data
df = pd.read_csv(DATA_PATH)
# filter for relevant columns
df = df[["question", "correct_answer", "student_answer", "ground_truth"]]
# print the first 5 rows
df.head()

# %%

# print number of unique questions in question column.
print("Number of unique questions:", len(df["question"].unique()))
print("average number of samples per question:", df.groupby("question").size().mean())

# %%

input_text = ": The house is wonderful."

# Encode and generate response
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
output_ids = model.generate(input_ids, max_new_tokens=20)[0]

# Decode and print the output text
output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
print(output_text)
