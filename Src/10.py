from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "meta-llama/Llama-3.1-8B-Instruct"
local_dir = "../HuggingFace/Llama-3.1-8B-Instrucear"

# Download model & tokenizer to local_dir
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=local_dir)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=local_dir)

# Save explicitly
model.save_pretrained(local_dir)
tokenizer.save_pretrained(local_dir)

# You can reload offline anytime:

model = AutoModelForCausalLM.from_pretrained(local_dir)
tokenizer = AutoTokenizer.from_pretrained(local_dir)