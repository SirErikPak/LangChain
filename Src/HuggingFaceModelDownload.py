from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "mistralai/Mistral-7B-Instruct-v0.3"
local_dir = "../HuggingFace/Mistral-7B-Instruct-v0.3"

# Download model & tokenizer to local_dir
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=local_dir)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=local_dir)

# Save explicitly
model.save_pretrained(local_dir)
tokenizer.save_pretrained(local_dir)

# You can reload offline anytime:

model = AutoModelForCausalLM.from_pretrained(local_dir)
tokenizer = AutoTokenizer.from_pretrained(local_dir)