import argparse
import os
import time

import torch
import tqdm
import transformers
import bitsandbytes

import sg_args
import sg_model
import sg_dataset
import sg_tools



def validation_step(model, validation_loader, save_dir, device_ids_first, parallel=False, global_step=0, wandb=None):
  validation_loss = []
  model.eval()
  with torch.no_grad():
    try:
      for i, data in enumerate(validation_loader):
        data = {
          'input_ids': data['input_ids'].to(device_ids_first),
          'labels': data['labels'].to(device_ids_first),
          'attention_mask': data['attention_mask'].to(device_ids_first)
        }
        output = model(input_ids=data['input_ids'], labels=data['labels'], attention_mask=data['attention_mask'], return_dict=True)
        loss = output.loss
        validation_loss.append(loss.mean().item())
    except Exception as e:
      torch.cuda.empty_cache()
      pass
  print('🧫 Validation loss:', round(sum(validation_loss) / len(validation_loss), 4))
  if (wandb):
    wandb.log({'validation_loss': round(sum(validation_loss) / len(validation_loss), 4), 'global_step': global_step})


  checkpoint_folder = os.path.join(
    save_dir,
    f"{transformers.trainer_utils.PREFIX_CHECKPOINT_DIR}-{global_step}"
  )
  peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
  pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")

  if not parallel:
    model.module.save_pretrained(peft_model_path)
  else:
    model.save_pretrained(peft_model_path)
  if os.path.exists(pytorch_model_path):
    os.remove(pytorch_model_path)
  

  print('📦 Checkpoint saved')
  model.train()

def finetune(
    args: argparse.Namespace,
    model,
    training_dataset: sg_dataset.SgDataset,
    validation_dataset: sg_dataset.SgDataset,
    device_ids: list[int] | None,
    parallel=False,
    wandb=None,
  ):
  print('🧬 Model parameters:', sum(param.numel() for param in model.parameters()))
  tmp_device = model.device
  if not parallel:
    model = torch.nn.DataParallel(model, device_ids=device_ids).to(tmp_device)
  else:
    raise NotImplementedError('❌ Parallel training not implemented')
  
  training_sampler = torch.utils.data.RandomSampler(training_dataset)
  validation_sampler = torch.utils.data.SequentialSampler(validation_dataset)
  # TODO: 🍞 코드 분석
  training_loader = torch.utils.data.DataLoader(
    dataset=training_dataset, batch_size=args.per_device_train_batch_size, shuffle=False,
    num_workers=0, pin_memory=True, sampler=training_sampler, collate_fn=sg_dataset.custom_collate
  )
  validation_loader = torch.utils.data.DataLoader(
    dataset=validation_dataset, batch_size=args.per_device_eval_batch_size, shuffle=False,
    num_workers=0, pin_memory=True, sampler=validation_sampler, collate_fn=sg_dataset.custom_collate
  )

  optimizer: torch.optim.Adam | bitsandbytes.adam.PagedAdam32bit = None
  if (args.optim == 'paged_adamw_32bit'):
    optimizer = bitsandbytes.adam.PagedAdam32bit(
      model.parameters(),
      lr=args.learning_rate,
      weight_decay=args.weight_decay,
      betas=(args.adam_beta1, args.adam_beta2), # Not exists in args?
      # eps=args.adam_epsilon, # Not exists in args
      # is_paged=True, # 옵티마이저 이름은 Paged이지만 실제로 내부 값은 False 인것 같음. 이유를 모르겠음
    )
  else:
    raise NotImplementedError(f'❌ Optimizer {args.optim} not implemented')
  
  scheduler = transformers.get_cosine_schedule_with_warmup(
    optimizer=optimizer, num_warmup_steps=0, num_training_steps=int(
      args.num_train_epochs * len(training_loader)
    )
  )
  
  # Wandb 로깅 사용
  if (wandb):
    wandb.watch(model)

  for epoch in range(args.num_train_epochs):
    model.train()
    training_loss = []
    start_time = time.time()
    oom = 0

    for i, data in enumerate(tqdm.tqdm(training_loader, desc=f'🚂 Epoch {epoch} / {args.num_train_epochs}')):
      data = {
        'input_ids': data['input_ids'].to(tmp_device),
        'labels': data['labels'].to(tmp_device),
        'attention_mask': data['attention_mask'].to(tmp_device)
      }
      try:
        # TODO: 🍞 코드 분석
        optimizer.zero_grad()
        output = model(
          input_ids=data['input_ids'],
          labels=data['labels'],
          attention_mask=data['attention_mask'],
          return_dict=True
          )
        loss = output.loss

        loss.mean().backward()
        # TODO: 🍞 코드 분석
        torch.nn.utils.clip_grad_value_(model.parameters(), 0.3)
        optimizer.step()
        scheduler.step()
        training_loss.append(loss.mean().item())
      except Exception as e:
        print(str(e))
        if 'out of memory' in str(e):
          oom += 1
        model.zero_grad()
        optimizer.zero_grad()
        scheduler.step()
        del data
        torch.cuda.empty_cache()

      if i % args.logging_steps == 0:
        log_loss = round(sum(training_loss) / len(training_loss), 4)
        log_lr = round(scheduler.get_last_lr()[0], 7)
        log_time = int(time.time() - start_time)

        print(
          'epoch: {}, step: {}/{}, loss: {}, lr: {}, oom: {}, time: {}s'.format(
            epoch + 1, i, len(training_loader),
            log_loss, log_lr, oom, log_time
          )
        )

        if (wandb):
          wandb.log({
            'train_loss': log_loss,
            'learning_rate': log_lr,
            'oom': oom,
            'time': log_time,
          })

        start_time = time.time()
        oom = 0
      
      if i % args.eval_steps == 0 and i > 0:
        validation_step(
          model,
          validation_loader,
          args.output_dir,
          device_ids_first=tmp_device,
          parallel=parallel,
          global_step=i,
          wandb=wandb
        )
    validation_step(
      model,
      validation_loader,
      args.output_dir,
      device_ids_first=tmp_device,
      parallel=parallel,
      global_step=i,
      wandb=wandb
    )



def main():
  (
    args,
    _,
    _,
    _,
    _,
    _,
  ) = sg_args.parse_args()

  if args.report_to == 'wandb':
    import wandb
    print('📊 Logging to wandb')
    args_dict = {k: v for k, v in vars(args).items() if k != 'asdf'}
    wandb.init(project='qlora-clm-apr', name=(args.run_name or None), config=args_dict, save_code=False, reinit=True)

  # AutoTokenizer가 CodeLlamaTokenizer를 인식하지 못함
  force_model = None
  if ('code_llama' in args.model_name_or_path.lower()) or ('codellama' in args.model_name_or_path.lower()):
    force_model = 'code_llama'
  model, tokenizer = sg_model.get_model_tokenizer(args, force_model)
  
  # 모델 구조 디버깅
  sg_tools.save_model_struct(
    model,
    model_name=sg_tools.nomalize_name_or_path_to_name(args.model_name_or_path) if args.model_name_or_path else None,
  )
  # return exit(0)

  model.config.use_cache = False

  training_dataset = sg_dataset.SgDataset(
    file_path=args.dataset, tokenizer=tokenizer, max_length=args.max_length,
    # load_range=[0, 1000]
  )
  validation_dataset = sg_dataset.SgDataset(
    file_path=args.validation_dataset, tokenizer=tokenizer, max_length=args.max_length,
    load_range=[0, 1000]
  )

  # Verifying the datatypes and parameter counts before training.
  sg_tools.print_trainable_parameters(args, model)
  sg_tools.print_model_named_parameters(model)

  finetune(
    args=args,
    model=model,
    training_dataset=training_dataset,
    validation_dataset=validation_dataset,
    device_ids=None,
    parallel=False,
    wandb=wandb if args.report_to == 'wandb' else None,
  )

if __name__ == '__main__':
  main()
