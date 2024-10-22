import os
from os.path import exists, join, isdir
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from peft import PeftModel
from peft.tuners.lora import LoraLayer
import transformers
from typing import Dict

DEFAULT_PAD_TOKEN = "[PAD]"

def __smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
  """토크나이저와 임베딩을 리사이징합니다.

  Note: 최적화되지 않은 버전으로, 임베딩 사이즈가 64로 나누어 떨어지지 않을 수 있습니다.
  """
  num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
  model.resize_token_embeddings(len(tokenizer))
  
  if num_new_tokens > 0:
    input_embeddings_data = model.get_input_embeddings().weight.data
    output_embeddings_data = model.get_output_embeddings().weight.data

    input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
    output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

    input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
    output_embeddings_data[-num_new_tokens:] = output_embeddings_avg


def get_last_checkpoint(checkpoint_dir):
    if isdir(checkpoint_dir):
        is_completed = exists(join(checkpoint_dir, 'completed'))
        if is_completed: return None, True # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
                max_step = max(max_step, int(filename.replace('checkpoint-', '')))
        if max_step == 0: return None, is_completed # training started, but no checkpoint
        checkpoint_dir = join(checkpoint_dir, f'checkpoint-{max_step}')
        print(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed # checkpoint found!
    return None, False # first training


# TODO: Update variables
max_new_tokens = 64
top_p = 0.9
temperature=0.7

# Base model
model_name_or_path = '/home/yglee/hdd1/nosync/models/codegen-6B'
# Adapter name on HF hub or local checkpoint path.
# adapter_path, _ = get_last_checkpoint('qlora/output/guanaco-7b')
adapter_path = '/home/yglee/hdd0/WorkspaceLab/p14/external/clm/nosync/output/codegen-6B_peft3/checkpoint-4808'

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
# Fixing some of the early LLaMA HF conversion issues.
tokenizer.bos_token_id = 1

# Load the model (use bf16 for faster inference)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    # load_in_4bit=True,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
    )
)

if tokenizer._pad_token is None:
  __smart_tokenizer_and_embedding_resize(
    special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
    tokenizer=tokenizer,
    model=model,
  )

model = PeftModel.from_pretrained(model, adapter_path)
model.eval()


def generate(model, max_new_tokens=max_new_tokens, top_p=top_p, temperature=temperature):
    inputs = tokenizer("public int getAlignmentEnd() {\n    if (getReadUnmappedFlag() {\n        return NO_ALIGNMENT_START;\n    } else if (this.mAlignmentEnd == NO_ALIGNMENT_START) {\n// buggy lines start:\n        this.mAlignmentEnd = getCigar().getReferenceLength() - 1;\n// buggy lines end\n    }\n    return this.mAlignmentEnd;\n}\n// fixed lines:\n<|endoftext|>", return_tensors="pt").to('cuda')
    print(inputs.input_ids.dtype)
    print(inputs.attention_mask.dtype)
    outputs = model.generate(
        **inputs, 
        generation_config=GenerationConfig(
            do_sample=True,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            temperature=temperature,
        )
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(text)
    return text

generate(model)
import pdb; pdb.set_trace()