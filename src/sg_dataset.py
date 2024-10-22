import codecs
import copy
from dataclasses import dataclass
import json
import multiprocessing
import os
from typing import Dict, Sequence, Union
import random

from datasets import Dataset
import torch
import tqdm
import transformers

import sg_tools

IGNORE_INDEX = -100
DEFAULT_PAD = 2 # "<pad>"


# ========== CLM 데이터셋 클래스 ==========
# 직접 finetune을 진행하는 경우 사용하는 데이터 로더 클래스 (clm에서 사용)
class SgDataset(torch.utils.data.Dataset):
  def __init__(
    self,
    file_path: str,
    tokenizer,
    max_length=1024,
    shuffle=False,
    load_range=None
  ):
    tokenizer_name = sg_tools.nomalize_name_or_path_to_name(tokenizer.name_or_path)
    # if '/' in tokenizer_name:
    #   file_name_with_extension = os.path.basename(tokenizer_name)
    #   tokenizer_name = os.path.splitext(file_name_with_extension)[0]  # 확장자를 제거한 파일 이름

    EOS_TOKEN = tokenizer.eos_token
    if 'incoder' in tokenizer_name:
      print('🚨 Using incoder tokenizer. Setting EOS token to "<|endofmask|>"')
      EOS_TOKEN = '<|endofmask|>'

    self.data = []
    self.max_length = max_length

    self.ext = file_path.split('.')[-1] if '.' in file_path else ''
    self.origin_file_path_wo_ext = file_path.replace(f'.{self.ext}', '') if self.ext else file_path
    self.preprocessed_file_path = f"{self.origin_file_path_wo_ext}_name_{tokenizer_name}_cut_{self.max_length}.{self.ext}"

    #  캐싱된 전처리 파일이 있으면 그걸 사용하고, 없으면 전처리 후 저장
    if os.path.exists(self.preprocessed_file_path):
      print(f"🍳 Found preprocessed dataset at {self.preprocessed_file_path}. Using that...")
    else:
      print(f"🥚 Preprocessing dataset at {file_path}...")
      file_streamer = codecs.open(file_path, 'r', 'utf-8')
      with open(self.preprocessed_file_path, 'w') as f:
        for i, line in enumerate(file_streamer.readlines()):
          line = eval(line)
          inputs = (
            line['buggy function before'] + '// buggy lines start:\n' + line['buggy line'] + '// buggy lines end\n' +
            line['buggy function after'] + '// fixed lines:\n' + line['fixed line'] + EOS_TOKEN
          )
          outputs = line['fixed line'] + EOS_TOKEN
          f.write(json.dumps({'input': inputs, 'output': outputs}) + '\n')

          if i % 10000 == 0:
            print('🥚 Preprocessing... ', i)

      file_streamer.close()

    # 전처리 된 데이터 불러오기
    print('ℹ️ Dont afraid of sequence length warning. It will be filtered by max_length.')
    preprocessed_file_streamer = codecs.open(self.preprocessed_file_path, 'r', 'utf-8')
    for line in preprocessed_file_streamer.readlines():
      line = eval(line)
      inputs = tokenizer.encode(line['input'], return_tensors='pt')
      outputs = tokenizer.encode(line['output'], return_tensors='pt')
      if inputs.size(1) > self.max_length:
        continue
      self.data.append({
        'input_ids': inputs,
        'labels': torch.cat(
          [torch.zeros(1, inputs.size(1) - outputs.size(1)).fill_(IGNORE_INDEX).long(), outputs], dim=1
        ),
        'attention_mask': torch.ones(inputs.size()).long()
      })

      if len(self.data) % 10000 == 0:
        print(f"🍞 {self.preprocessed_file_path} Tokenizing... {len(self.data)}")

      if load_range is not None and len(self.data) == load_range[1]:
        break
    preprocessed_file_streamer.close()

    if shuffle:
      random.seed(7)
      random.shuffle(self.data)

    print(f"🍞 Total size ({file_path}): {len(self.data)}")
    if load_range is not None:
      self.data = self.data[load_range[0]:]
  
  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, item):
    return self.data[item]



class SgDataset2(torch.utils.data.Dataset):
  def __init__(
    self,
    file_path: str,
    tokenizer,
    max_length=1024,
    shuffle=False,
    load_range=None,
    num_proc=None
  ):
    self.tokenizer = tokenizer
    self.tokenizer_name = sg_tools.nomalize_name_or_path_to_name(tokenizer.name_or_path)
    self.max_length = max_length
    self.num_proc = num_proc or os.cpu_count()

    # 전처리 된 데이터 파일 이름
    ext = file_path.split('.')[-1] if '.' in file_path else ''
    origin_file_path_wo_ext = file_path.replace(f'.{ext}', '') if ext else file_path
    self.preprocessed_file_path = f"{origin_file_path_wo_ext}_name_{self.tokenizer_name}_cut_{self.max_length}.{ext}"

    self.data = self._load_and_process_data(file_path, load_range, shuffle)
    print(f'🍞 Total size ({file_path}): {len(self.data)}')


  def __len__(self):
    return len(self.data)


  def __getitem__(self, item):
    return self.data[item]


  def _preprocess_line(self, line):
    """단일 라인 전처리 함수"""
    line = eval(line)
    inputs = (
      line['buggy function before'] + '// buggy lines start:\n' +
      line['buggy line'] + '// buggy lines end\n' +
      line['buggy function after'] + '// fixed lines:\n' +
      line['fixed line'] + self.tokenizer.eos_token
    )
    outputs = line['fixed line'] + self.tokenizer.eos_token
    return {'input': inputs, 'output': outputs}


  def _tokenize_item(self, item):
    """단일 아이템 토큰화 함수"""
    try: 
      inputs = self.tokenizer.encode(item['input'], return_tensors='pt')
      outputs = self.tokenizer.encode(item['output'], return_tensors='pt')
      
      # 최대 길이를 넘어가는 경우는 제외
      if inputs.size(1) > self.max_length:
        return None
      return {
        'input_ids': inputs,
        'labels': torch.cat(
          [
            torch.zeros(1, inputs.size(1) - outputs.size(1)).fill_(IGNORE_INDEX).long(),
            outputs
          ],
          dim=1
        ),
        'attention_mask': torch.ones(inputs.size()).long()
      }
    except Exception as e:
      print(f"Error tokenizing item: {e}")
      return None


  def _load_and_process_data(self, file_path, load_range, shuffle):
    """데이터 로딩 및 전처리 병렬 메인"""
    if os.path.exists(self.preprocessed_file_path):
      print(f"🍳 Found preprocessed dataset at {self.preprocessed_file_path}. Using that...")
    else:
      print(f"🥚 Preprocessing dataset at {file_path}...")
      file_streamer = codecs.open(file_path, 'r', 'utf-8')
      with open(self.preprocessed_file_path, 'w') as f:
        for i, line in enumerate(file_streamer.readlines()):
          f.write(json.dumps(self._preprocess_line(line)) + '\n')
          if i % 10000 == 0:
            print('🥚 Preprocessing... ', i)
      file_streamer.close()

    # 전처리된 데이터 로딩 및 토큰화
    print('ℹ️ Loading and tokenizing data...')
    processed_data = []
        
    with open(self.preprocessed_file_path, 'r', encoding='utf-8') as f:
      lines = f.readlines()
      if load_range is not None:
        lines = lines[load_range[0]:load_range[1]]

      # 병렬 토큰화
      print('ℹ️ Dont afraid of sequence length warning. It will be filtered by max_length.')
      with multiprocessing.Pool(self.num_proc) as pool:
        for item in tqdm.tqdm(
          pool.imap(
            self._tokenize_item,
            (json.loads(line) for line in lines),
            chunksize=1000
          ),
          total=len(lines),
          desc="🍞 Tokenizing"
        ):
          if item is not None:
            processed_data.append(item)

    if shuffle:
      random.seed(7)
      random.shuffle(processed_data)

    return processed_data



def custom_collate(batch):
  batch_data = {'input_ids': [], 'labels': [], 'attention_mask': []}
  max_len = max([b['input_ids'].size(1) for b in batch])
  for b in batch:
    batch_data['input_ids'].append(torch.cat([
      b['input_ids'],
      torch.zeros(1, max_len - b['input_ids'].size(1)).fill_(DEFAULT_PAD).long()
    ], dim=1))
    batch_data['labels'].append(torch.cat([
      b['labels'],
      torch.zeros(1, max_len - b['labels'].size(1)).fill_(IGNORE_INDEX).long()
    ], dim=1))
    batch_data['attention_mask'].append(torch.cat([
      b['attention_mask'],
      torch.zeros(1, max_len - b['attention_mask'].size(1))
    ], dim=1))

  batch_data['input_ids'] = torch.cat(batch_data['input_ids'], dim=0)
  batch_data['labels'] = torch.cat(batch_data['labels'], dim=0)
  batch_data['attention_mask'] = torch.cat(batch_data['attention_mask'], dim=0)

  return batch_data
# ========== CLM 데이터셋 클래스 끝 ==========



# ========== Trainer용 데이터 모듈 생성 함수 ==========
@dataclass
class DataCollatorForCausalLM(object):
  tokenizer: transformers.PreTrainedTokenizer
  source_max_len: int
  target_max_len: int
  train_on_source: bool
  predict_with_generate: bool

  def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
    # Extract elements
    sources = [f"{self.tokenizer.bos_token}{example['input']}" for example in instances]
    targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in instances]

    # Tokenize
    tokenized_sources_with_prompt = self.tokenizer(
      sources,
      # max_length=self.source_max_len,
      # truncation=True,
      truncation=False,
      padding=False,
      add_special_tokens=False,
    )
    tokenized_targets = self.tokenizer(
      targets,
      # max_length=self.target_max_len,
      # truncation=True,
      truncation=False,
      padding=False,
      add_special_tokens=False,
    )

    # Build the input and labels for causal LM
    input_ids = []
    labels = []
    for tokenized_source, tokenized_target in zip(
      tokenized_sources_with_prompt['input_ids'],
      tokenized_targets['input_ids']
    ):
      if not self.predict_with_generate:
        input_ids.append(torch.tensor(tokenized_source + tokenized_target))
        if not self.train_on_source:
          labels.append(
            torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
          )
        else:
          labels.append(torch.tensor(copy.deepcopy(tokenized_source + tokenized_target)))
      else:
        input_ids.append(torch.tensor(tokenized_source))
    # Apply padding
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX) if not self.predict_with_generate else None
    data_dict = {
      'input_ids': input_ids,
      'attention_mask':input_ids.ne(self.tokenizer.pad_token_id),
    }
    if labels is not None:
      data_dict['labels'] = labels
    return data_dict

def make_data_module(tokenizer: transformers.PreTrainedTokenizer, args) -> Dict:
  def data_length_filter(datas: Dataset, tokenizer: transformers.PreTrainedTokenizer, max_length: int):
    def check_length(example):
      source = f"{tokenizer.bos_token}{example['input']}"
      target = f"{example['output']}{tokenizer.eos_token}"
      
      tok_source = tokenizer(source, truncation=False, padding=False, add_special_tokens=False)
      tok_target = tokenizer(target, truncation=False, padding=False, add_special_tokens=False)
      
      return len(tok_source['input_ids']) + len(tok_target['input_ids']) <= max_length

    # 길이 조건을 만족하는 데이터만 필터링
    filtered_dataset = datas.filter(check_length, num_proc=(os.cpu_count() or 4))
    print(f"✂️ Original/Filtered dataset by length: {len(datas)}/{len(filtered_dataset)}")
    return filtered_dataset

  def local_dataset(dataset_name_or_path: str, tokenizer: transformers.PreTrainedTokenizer, max_length: Union[int|None] = None):
    """
      기존 CLM 논문의 데이터를 HuggingFace Dataset으로 캐싱하거나 변환하는 함수
    """
    new_data_path = dataset_name_or_path.replace('.jsonl', '_hfdataset.jsonl')

    if os.path.exists(new_data_path):
      print(f"📦 Found processed dataset at {new_data_path}. Using that.")
      full_dataset = Dataset.from_json(path_or_paths=new_data_path)
    else:
      file_streamer = codecs.open(dataset_name_or_path, 'r', 'utf-8')
      with open(new_data_path, 'w') as f:
        for i, line in enumerate(file_streamer.readlines()):
          line = eval(line)
          input = (
            line['buggy function before'] + '// buggy lines start:\n' + line['buggy line'] +
            '// buggy lines end\n' + line['buggy function after'] + '// fixed lines:\n'
          )
          output = line['fixed line']
          f.write(json.dumps({'input': input, 'output': output}) + '\n')

          if i % 10000 == 0:
            print('🥚 Reading...(1) ', i)
      file_streamer.close()

    full_dataset = Dataset.from_json(path_or_paths=new_data_path)
    if args.max_length is not None:
      print('📢 Dont afraid of sequence length warning. It will be filtered by max_length.')
      full_dataset = data_length_filter(full_dataset, tokenizer, max_length)

    return full_dataset

  def load_data(dataset_name, tokenizer):
    if os.path.exists(dataset_name):
      try:
        args.dataset_format = args.dataset_format if args.dataset_format else "input-output"
        full_dataset = local_dataset(dataset_name, tokenizer, args.max_length)
        return full_dataset
      except:
        raise ValueError(f"Error loading dataset from {dataset_name}")
    else:
      raise NotImplementedError(f"Dataset {dataset_name} not implemented yet.")

  # Load dataset.
  train_dataset = load_data(args.dataset, tokenizer)
  validation_dataset = None
  if args.validation_dataset:
    validation_dataset = load_data(args.validation_dataset, tokenizer)

  if validation_dataset is None:
    print('⚠️ No validation dataset provided. Splitting train dataset in train and validation according to `eval_dataset_size`')
    __tmp_dataset = train_dataset.train_test_split(
      test_size=args.eval_dataset_size, shuffle=True, seed=42
    )
    train_dataset = __tmp_dataset['train']
    validation_dataset = __tmp_dataset['test']

  # Split train/eval, reduce size
  eval_dataset = None
  if args.do_eval or args.do_predict:
    eval_dataset = validation_dataset
    # 최대 평가 데이터셋 크기 제한
    if args.max_eval_samples is not None and len(eval_dataset) > args.max_eval_samples:
      print(f"⚠️ Truncating eval dataset to {args.max_eval_samples} samples")
      eval_dataset = eval_dataset.select(range(args.max_eval_samples))
    # group_by_length의 경우 length 파라미터 추가해줌
    if args.group_by_length:
      print('🧂 args.group_by_length(eval): Adding length params...')
      eval_dataset = eval_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})

  if args.do_train:
    # 최대 훈련 데이터셋 크기 제한
    if args.max_train_samples is not None and len(train_dataset) > args.max_train_samples:
      print(f"⚠️ Truncating train dataset to {args.max_train_samples} samples")
      train_dataset = train_dataset.select(range(args.max_train_samples))
    # group_by_length의 경우 length 파라미터 추가해줌
    if args.group_by_length:
      print('🧂 args.group_by_length(train): Adding length params...')
      train_dataset = train_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})

  data_collator = DataCollatorForCausalLM(
    tokenizer=tokenizer,
    source_max_len=args.source_max_len,
    target_max_len=args.target_max_len,
    train_on_source=args.train_on_source,
    predict_with_generate=args.predict_with_generate,
  )
  return dict(
    train_dataset=train_dataset if args.do_train else None,
    eval_dataset=eval_dataset if args.do_eval else None,
    predict_dataset=eval_dataset if args.do_predict else None,
    data_collator=data_collator
  )

# ========== Trainer용 데이터 모듈 생성 함수 끝 ==========
