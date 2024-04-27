import gc
import os
import torch
from datasets import load_from_disk
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import (
    TrainingArguments,
    Trainer,
    default_data_collator,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from trl import DPOTrainer

# Constants
BASE_MODEL = "/home/zhiyu/code/llm/model/Meta-Llama-3-8B"
DATASET_NAME = "/mnt/nasdata/yongcheng/projects/research/DPO/direct-preference-optimization/.cache/root/Anthropic___hh-rlhf"
OUTPUT_DIR = "./results/"
SAVE_DIR = "./trained_model/"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
torch_dtype = torch.bfloat16
attn_implementation = "flash_attention_2"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation=attn_implementation,
)
tokenizer.pad_token = tokenizer.eos_token

# Load dataset
dataset = load_from_disk(DATASET_NAME)

# Define functions
def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = '\n\nAssistant:'
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[:search_term_idx + len(search_term)]

def split_prompt_and_responses(ex):
    prompt = extract_anthropic_prompt(ex['chosen'])
    chosen_response = ex['chosen'][len(prompt):]
    rejected_response = ex['rejected'][len(prompt):]
    return {'prompt': prompt, 'chosen': chosen_response, 'rejected': rejected_response}

# Process dataset
dataset = dataset.map(split_prompt_and_responses)

# Configure LoRA
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
)
model.add_adapter(peft_config)
model = prepare_model_for_kbit_training(model)

# Configure training
dpo_args = TrainingArguments(
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    warmup_steps=10,
    evaluation_strategy="steps",
    logging_strategy="steps",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=1,
    logging_steps=1,
    report_to="wandb",
    output_dir=OUTPUT_DIR,
    remove_unused_columns=False,
)
dpo_args.max_length = 1024
dpo_args.max_prompts_length = 512

# Train model
trainer = DPOTrainer(
    model=model,
    tokenizer=tokenizer,
    args=dpo_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=default_data_collator,
)
trainer.train()
trainer.save_model(SAVE_DIR)