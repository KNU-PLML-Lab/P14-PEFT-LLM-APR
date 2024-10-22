import argparse
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

def get_tokenizer(
  args: argparse.Namespace,
  model: transformers.PreTrainedTokenizer,
  force_model: str, # 'code_llama'
):
  tokenizer = None

  # AutoTokenizer가 CodeLlamaTokenizer를 감지하지 못해서 따로 처리
  if force_model == 'code_llama':
    tokenizer = transformers.CodeLlamaTokenizer.from_pretrained(
      args.model_name_or_path,
      cache_dir=args.cache_dir,
      padding_side="right",
      use_fast=False, # Fast tokenizer giving issues.
      tokenizer_type='llama' if 'llama' in args.model_name_or_path.lower() else None, # Needed for HF name change
      trust_remote_code=args.trust_remote_code,
      token=args.token,
    )
  else:
    # Tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
      args.model_name_or_path,
      cache_dir=args.cache_dir,
      padding_side="right",
      use_fast=False, # Fast tokenizer giving issues.
      tokenizer_type='llama' if 'llama' in args.model_name_or_path.lower() else None, # Needed for HF name change
      trust_remote_code=args.trust_remote_code,
      token=args.token,
    )

  if tokenizer._pad_token is None:
    __smart_tokenizer_and_embedding_resize(
      special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
      tokenizer=tokenizer,
      model=model,
    )


  if 'llama' in args.model_name_or_path.lower() or isinstance(tokenizer, transformers.CodeLlamaTokenizer):
    # LLaMa 토크나이저는 올바른 특수 토큰이 설정되어 있지 않을 수 있습니다.
    # 누락된 경우 다른 토큰으로 분석되는 것을 방지하기 위해 추가합니다.
    # 이들은 vocabulary에 포함되어 있습니다.
    # 또한 `model.config.pad_token_id`는 `<unk>` 토큰에 해당하는 0입니다.
    print('LLaMa Detected> Adding special tokens.')
    tokenizer.add_special_tokens({
      "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
      "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
      "unk_token": tokenizer.convert_ids_to_tokens(
        model.config.pad_token_id if (model.config.pad_token_id != None and model.config.pad_token_id != -1) else tokenizer.pad_token_id
      ),
    })