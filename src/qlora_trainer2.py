# 하다가 말음. qlora/qlora.py 파일편집으로 이동.
import os
import bitsandbytes as bnb
import pandas as pd
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
from huggingface_hub import login
from peft import (
  LoraConfig,
  PeftConfig,
  get_peft_model,
  prepare_model_for_kbit_training,
)
from transformers import (
  AutoConfig,
  AutoModelForCausalLM,
  AutoTokenizer,
  BitsAndBytesConfig,
)
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig


def qlora_test2():
  OUTPUT_DIR = "./outputs/qlora_test1"
  model_id = "mistralai/Mistral-7B-Instruct-v0.2"
  dataset_name = 'Amod/mental_health_counseling_conversations'
  login(token=os.getenv('HF_TOKEN'))

  fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
  )

  accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

  # Load the dataset
  dataset = load_dataset(dataset_name, split="train")

  bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
  )

  model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    # trust_remote_code=True,
    quantization_config=bnb_config,
  )

  tokenizer = AutoTokenizer.from_pretrained(model_id)
  tokenizer.pad_token = tokenizer.eos_token
  
  model = prepare_model_for_kbit_training(model)

  def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit #if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
      if isinstance(module, cls):
        names = name.split('.')
        lora_module_names.add(names[0] if len(names) == 1 else names[-1])
      if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

  modules = find_all_linear_names(model)
  print('🗄️ Modules:', modules)

  config = LoraConfig(
    r = 8, # rank of update matrices
    lora_alpha = 32, # LoRA scaling factor
    target_modules = modules,
    lora_dropout = 0.05,
    bias = "none", # bias training - 'none', 'all', 'lora_only'
    task_type = "CAUSAL_LM"
  )

  model = get_peft_model(model, config)

#   def generate_prompt(data_point):
#     return f"""
# <Human>: {data_point["Context"]}
# <AI>: {data_point["Response"]}
#     """.strip()
  
  def generate_prompt(data_point):
    """
    Generates a formatted prompt for fine-tuning from a data point.
    """
    prefix_text = "This is a conversation from a mental health therapy chat. Respond empathetically and informatively." #instruction
    context = data_point['Context']
    response = data_point['Response']
    formatted_prompt = f"<s>[INST] {prefix_text} {context} [/INST]{response}</s>"
    return formatted_prompt

  def generate_and_tokenize_prompt(data_point):
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenizer(
      full_prompt,
      padding = 'max_length',
      truncation = True,
      max_length = 512,
    )
    return tokenized_full_prompt

  dataset = load_dataset(dataset_name, split="train")
  dataset = dataset.shuffle().map(generate_and_tokenize_prompt)

  training_args = transformers.TrainingArguments(
    # auto_find_batch_size = True,
    per_device_train_batch_size = 8,
    num_train_epochs = 4,
    learning_rate = 2e-4,
    bf16 = True,
    save_total_limit = 4,
    logging_steps = 10,
    output_dir = OUTPUT_DIR,
    save_strategy = 'epoch'
  )

  trainer = transformers.Trainer(
    model = model,
    train_dataset = dataset
  )
  model.config.use_cache = False
  trainer.train()
