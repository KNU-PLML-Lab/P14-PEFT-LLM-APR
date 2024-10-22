import json
import os

import numpy
import transformers

import sg_args
import sg_model
import sg_dataset
import sg_tools

def main():
  print('🧹 Parse arguments...')
  (
    args,
    _,
    _,
    training_args,
    _,
    _,
  ) = sg_args.parse_args()

  # AutoTokenizer가 CodeLlamaTokenizer를 인식하지 못함
  print('⭐ Load model and tokenizer...')
  force_model = None
  if ('code_llama' in args.model_name_or_path.lower()) or ('codellama' in args.model_name_or_path.lower()):
    force_model = 'code_llama'
  model, tokenizer = sg_model.get_model_tokenizer(args, force_model)
  
  # 모델 구조 디버깅
  # sg_tools.save_model_struct(model)
  # return exit(0)

  model.config.use_cache = False

  print('📚 Load datasets...')
  data_module = sg_dataset.make_data_module(tokenizer=tokenizer, args=args)

  print('🚄 Prepareing trainer...')
  trainer = transformers.Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    **{k:v for k,v in data_module.items() if k != 'predict_dataset'},
  )

  # Callbacks
  print('🛄 Setup callbacks...')
  if not args.full_finetune:
    trainer.add_callback(sg_tools.SavePeftModelCallback)
  
  # Verifying the datatypes and parameter counts before training.
  sg_tools.print_trainable_parameters(args, model)
  sg_tools.print_model_named_parameters(model)

  all_metrics = {"run_name": args.run_name}
  # Training
  if args.do_train:
    print("*** Train ***")
    # Note: `resume_from_checkpoint` not supported for adapter checkpoints by HF.
    # Currently adapter checkpoint is reloaded as expected but optimizer/scheduler states are not.
    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    all_metrics.update(metrics)
  # Evaluation
  if args.do_eval:
    print("*** Evaluate ***")
    metrics = trainer.evaluate(metric_key_prefix="eval")
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    all_metrics.update(metrics)
  # Prediction
  if args.do_predict:
    print("*** Predict ***")
    prediction_output = trainer.predict(test_dataset=data_module['predict_dataset'],metric_key_prefix="predict")
    prediction_metrics = prediction_output.metrics
    predictions = prediction_output.predictions
    predictions = numpy.where(predictions != -100, predictions, tokenizer.pad_token_id)
    predictions = tokenizer.batch_decode(
      predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    with open(os.path.join(args.output_dir, 'predictions.jsonl'), 'w') as fout:
      for i, example in enumerate(data_module['predict_dataset']):
        example['prediction_with_input'] = predictions[i].strip()
        example['prediction'] = predictions[i].replace(example['input'], '').strip()
        fout.write(json.dumps(example) + '\n')
    print(prediction_metrics)
    trainer.log_metrics("predict", prediction_metrics)
    trainer.save_metrics("predict", prediction_metrics)
    all_metrics.update(prediction_metrics)

  if (args.do_train or args.do_eval or args.do_predict):
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as fout:
      fout.write(json.dumps(all_metrics))



if __name__ == '__main__':
  main()