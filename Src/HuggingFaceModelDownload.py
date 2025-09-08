# from transformers import AutoTokenizer, AutoModelForCausalLM

# model_name = "google/flan-t5-small"
# local_dir = "../HuggingFace/Flan-T5-Small"

# Download model & tokenizer to local_dir
# tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=local_dir)
# model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=local_dir)

# # Save explicitly
# model.save_pretrained(local_dir)
# tokenizer.save_pretrained(local_dir)

# # You can reload offline anytime:

# model = AutoModelForCausalLM.from_pretrained(local_dir)
# tokenizer = AutoTokenizer.from_pretrained(local_dir)


import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Define the model name and local directory
model_name = "google/flan-t5-base "
local_dir = "../HuggingFace/flan-t5-base"

# Use the correct class for the T5 model
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=local_dir)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=local_dir)

print(f"Model and tokenizer for {model_name} downloaded and saved to {local_dir}")