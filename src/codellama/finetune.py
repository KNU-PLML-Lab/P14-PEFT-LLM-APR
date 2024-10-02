import time
import traceback
import torch
import torch.nn as nn
import tqdm
from dataset import Dataset, custom_collate
from transformers import (
  AutoTokenizer,
  CodeGenForCausalLM,
  get_cosine_schedule_with_warmup,
  Adafactor
)

def step_validation(model, device_ids, validation_loader, save_dir, parallel=False):
  print('🧫 Validation...')
  validation_loss = []
  # no_grad를 설정해 validation 과정에서 gradient가 계산되지 않도록 함
  with torch.no_grad():
    for _, data in enumerate(validation_loader):
      data = {
        'input_ids': data['input_ids'].to(device_ids[0]),
        'labels': data['labels'].to(device_ids[0]),
        'attention_mask': data['attention_mask'].to(device_ids[0])
      }
      output = model(
        input_ids=data['input_ids'],
        labels=data['labels'],
        attention_mask=data['attention_mask'],
        return_dict=True
      )
      loss = output.loss
      # loss의 평균을 계산하여 validation_loss에 추가
      validation_loss.append(loss.mean().item())
  # validation_loss의 평균을 계산하여 출력
  print('🔬 Validation loss:', round(sum(validation_loss) / len(validation_loss), 4))
  # 모델 저장
  if not parallel:
    model.module.save_pretrained(save_dir)
  else:
    model.save_pretrained(save_dir)
  # 모델을 다시 train 모드로 전환
  model.train()

def run_finetune(
  training_file, # 학습 데이터 파일 경로
  validation_file, # 검증 데이터 파일 경로
  vocabulary_file, # 어휘 사전 파일 경로
  pretrained_file, # 사전 학습된 모델 파일 경로
  device_ids,
  epochs,
  batch_size,
  save_dir,
  parallel=False,
  load_range=None
):
  # tokenizer, model, optimizer, scheduler
  tokenizer = AutoTokenizer.from_pretrained(vocabulary_file)
  model = CodeGenForCausalLM.from_pretrained(pretrained_file)
  print('🧠 Model parameters:', sum(param.numel() for param in model.parameters()))
  # parallel이 False인 경우 DataParallel을 사용하여 model을 병렬화
  if not parallel:
    model = nn.DataParallel(model, device_ids=device_ids).to(device_ids[0])
  # parallel이 True인 경우 
  else:
    # 현재 구현되지 않음. 예외 반환하고 종료
    raise NotImplementedError('Parallel training is not implemented yet.')
    # if 'codegen-350M' in pretrained_file:
    #   model.parallelize(device_map = {
    #     0: [_ for _ in range(0, 20)]
    #   })
    # elif 'codegen-2B' in pretrained_file:
    #   model.parallelize(device_map = {
    #     0: [_ for _ in range(0, 7)],
    #     1: [_ for _ in range(7, 16)],
    #     2: [_ for _ in range(16, 25)],
    #     3: [_ for _ in range(25, 32)]
    #   })
    # else:
    #   model.parallelize(device_map = {
    #     0: [_ for _ in range(0, 4)], 
    #     1: [_ for _ in range(4, 8)],
    #     2: [_ for _ in range(8, 12)],
    #     3: [_ for _ in range(12, 16)],
    #     4: [_ for _ in range(16, 20)],
    #     5: [_ for _ in range(20, 24)],
    #     6: [_ for _ in range(24, 29)],
    #     7: [_ for _ in range(29, 33)]
    #   })
  
  # 학습 데이터셋, 검증 데이터셋 로드
  # CodeLlama에서는 메모리와 속도 문제로 max_length를 768로 설정
  train_dataset = Dataset(
    training_file,
    tokenizer,
    max_length=768,
    shuffle=False,
    load_range=load_range
  )
  validation_dataset = Dataset(
    validation_file,
    tokenizer,
    max_length=768,
    load_range=None
  )

  training_sampler = torch.utils.data.SequentialSampler(train_dataset)
  validation_sampler = torch.utils.data.SequentialSampler(validation_dataset)

  # 데이터 로더 생성
  train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0, # 데이터 로딩을 위한 프로세스 수
    pin_memory=True, # 학습 데이터를 고정 메모리에 올림
    sampler=training_sampler, # 데이터 로딩 시 사용할 샘플러
    collate_fn=custom_collate, # 데이터 로딩 시 사용할 함수
  )
  validation_loader = torch.utils.data.DataLoader(
    dataset=validation_dataset,
    batch_size=3*batch_size, # 검증 데이터셋의 배치 크기는 학습 데이터셋의 3배
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    sampler=validation_sampler,
    collate_fn=custom_collate,
  )

  # optimizer와 scheduler 설정
  optimizer = Adafactor(
    model.parameters(),
    lr=1e-5, # 학습률
    scale_parameter=False, # 🍞
    relative_step=False # 🍞
  )
  scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=int(epochs * len(train_loader))
  )

  # tqdm을 사용하여 학습 진행 상황을 출력
  for epoch in range(epochs):
    model.train()
    training_loss = []
    start_time = time.time()
    oom = 0
    print('🚀 Epoch start:', epoch + 1, '/', epochs)
    for i, data in enumerate(tqdm.tqdm(train_loader, total=len(train_loader))):
      data = {
        'input_ids': data['input_ids'].to(device_ids[0]),
        'labels': data['labels'].to(device_ids[0]),
        'attention_mask': data['attention_mask'].to(device_ids[0])
      }
      try:
        optimizer.zero_grad()
        output = model(
          input_ids=data['input_ids'],
          labels=data['labels'],
          attention_mask=data['attention_mask'],
          return_dict=True
        )
        loss = output.loss
        # loss의 평균을 계산하여 training_loss에 추가
        training_loss.append(loss.mean().item())
        loss.mean().backward() # 역전파
        optimizer.step()
        scheduler.step()
      except RuntimeError as e:
        if 'out of memory' in str(e):
          oom += 1
          print(f'💥 Out of Memory: {oom}')
          model.zero_grad()
          optimizer.zero_grad()
          scheduler.step()
          del data

          torch.cuda.empty_cache()
          time.sleep(5)
        else:
          print('💥 RuntimeError:', e)
          traceback.print_exc()
          break
      
      if i % 1000 == 0:
        # 로그 출력
        print('📝 Epoch: {}, 👣 step: {}/{}, 🔥 loss: {}, 📈 lr: {}, 💥 oom: {}, ⌛ time: {}s'.format(
          epoch + 1, i, len(train_loader),
          round(sum(training_loss) / len(training_loss), 4),
          round(scheduler.get_last_lr()[0], 7),
          oom,
          int(time.time() - start_time)
        ))
        start_time = time.time()
        oom = 0
      if i % 10000 == 0 and i > 0:
        # 검증 과정 실행
        step_validation(model, device_ids, validation_loader, save_dir, parallel=parallel)
    # 최종 검증 과정 실행
    step_validation(model, device_ids, validation_loader, save_dir, parallel)

if __name__ == '__main__':
  device_ids = [0]
  training_file = ''      # fine-tuning 데이터 경로로 변경
  validation_file = ''    # fine-tuning 데이터 경로로 변경
  vocabulary_file = 'CodeLlama-7b-hf/'
  pretrained_file = 'CodeLlama-7b-hf/'

  run_finetune(
    training_file,
    validation_file,
    vocabulary_file,
    pretrained_file,
    device_ids,
    epochs=1,
    batch_size=1,
    save_dir='../../outputs/CodeLlama-7b-hf',
    parallel=False,
    load_range=None
  )