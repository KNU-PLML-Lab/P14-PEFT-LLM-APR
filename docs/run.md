## Fine-tuning

#### codegen_6b_v11
```bash
CUDA_VISIBLE_DEVICES="0" python ./src/sg_finetune.py \
    --run_name codegen_6b_v11 \
    --model_name_or_path ~/WorkspaceLabModels/codegen_6b \
    --output_dir ~/WorkspaceLabModels/codegen_6b_v11 \
    --dataset ./data/finetune_training.jsonl \
    --validation_dataset ./data/finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0002 \
    --seed 142 \
    --max_length 768 \
    --per_device_train_batch_size 8 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --eval_steps 1000 \
    --save_steps 999999 \
    --specific_save_steps "13,125,1250,6250" \
    --eval_dataset_size 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --report_to wandb
```

#### incoder_6b_v11
```bash
CUDA_VISIBLE_DEVICES="1" python ./src/sg_finetune.py \
    --run_name incoder_6b_v11 \
    --model_name_or_path ~/WorkspaceLabModels/incoder_6b \
    --output_dir ~/WorkspaceLabModels/incoder_6b_v11 \
    --dataset ./data/finetune_training.jsonl \
    --validation_dataset ./data/finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0002 \
    --seed 142 \
    --max_length 1024 \
    --per_device_train_batch_size 8 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --eval_steps 1000 \
    --save_steps 999999 \
    --specific_save_steps "13,125,1250,6250" \
    --eval_dataset_size 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --report_to wandb
```

#### codellama_7b_v11
```bash
CUDA_VISIBLE_DEVICES="2" python ./src/sg_finetune.py \
    --run_name codellama_7b_v11 \
    --model_name_or_path ~/WorkspaceLabModels/codellama_7b \
    --output_dir ~/WorkspaceLabModels/codellama_7b_v11 \
    --dataset ./data/finetune_training.jsonl \
    --validation_dataset ./data/finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0002 \
    --seed 142 \
    --max_length 1024 \
    --per_device_train_batch_size 8 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --eval_steps 1000 \
    --save_steps 999999 \
    --specific_save_steps "13,125,1250,6250" \
    --eval_dataset_size 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --report_to wandb
```

### starcoder2_7b_v11
```bash
CUDA_VISIBLE_DEVICES="3" python ./src/sg_finetune.py \
    --run_name starcoder2_7b_v11 \
    --model_name_or_path ~/WorkspaceLabModels/starcoder2_7b \
    --output_dir ~/WorkspaceLabModels/starcoder2_7b_v11 \
    --dataset ./data/finetune_training.jsonl \
    --validation_dataset ./data/finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0002 \
    --seed 142 \
    --max_length 1024 \
    --per_device_train_batch_size 8 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --eval_steps 1000 \
    --save_steps 999999 \
    --specific_save_steps "13,125,1250,6250" \
    --eval_dataset_size 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --report_to wandb
```

### qwen2.5_coder_7b_v11
```bash
CUDA_VISIBLE_DEVICES="0" python ./src/sg_finetune.py \
    --run_name qwen2.5_coder_7b_v11 \
    --model_name_or_path ~/WorkspaceLabModels/qwen2.5_coder_7b \
    --output_dir ~/WorkspaceLabModels/qwen2.5_coder_7b_v11 \
    --dataset ./data/finetune_training.jsonl \
    --validation_dataset ./data/finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0002 \
    --seed 142 \
    --max_length 1024 \
    --per_device_train_batch_size 4 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --eval_steps 2000 \
    --save_steps 999999 \
    --specific_save_steps "25,250,2500,12500" \
    --eval_dataset_size 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --report_to wandb
```

#### codellama_13b_v11
```bash
CUDA_VISIBLE_DEVICES="1" python ./src/sg_finetune.py \
    --run_name codellama_13b_v11 \
    --model_name_or_path ~/WorkspaceLabModels/codellama_13b \
    --output_dir ~/WorkspaceLabModels/codellama_13b_v11 \
    --dataset ./data/finetune_training.jsonl \
    --validation_dataset ./data/finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0002 \
    --seed 142 \
    --max_length 1024 \
    --per_device_train_batch_size 4 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --eval_steps 2000 \
    --save_steps 999999 \
    --specific_save_steps "25,250,2500,12500" \
    --eval_dataset_size 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --report_to wandb
```

#### codegen_16b_v11
```bash
CUDA_VISIBLE_DEVICES="0" python ./src/sg_finetune.py \
    --run_name codegen_16b_v11 \
    --model_name_or_path ~/WorkspaceLabModels/codegen_16b \
    --output_dir ~/WorkspaceLabModels/codegen_16b_v11 \
    --dataset ./data/finetune_training.jsonl \
    --validation_dataset ./data/finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0002 \
    --seed 142 \
    --max_length 768 \
    --per_device_train_batch_size 4 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --eval_steps 2000 \
    --save_steps 999999 \
    --specific_save_steps "25,250,2500,12500" \
    --eval_dataset_size 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --report_to wandb
```

#### codegen_6b_v12
```bash
CUDA_VISIBLE_DEVICES="2" python ./src/sg_finetune.py \
    --run_name codegen_6b_v12 \
    --model_name_or_path ~/WorkspaceLabModels/codegen_6b \
    --output_dir ~/WorkspaceLabModels/codegen_6b_v12 \
    --dataset ./data/finetune_training.jsonl \
    --validation_dataset ./data/finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0002 \
    --seed 120462 \
    --max_length 768 \
    --per_device_train_batch_size 8 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --eval_steps 1000 \
    --save_steps 999999 \
    --specific_save_steps "13,125,1250,6250" \
    --eval_dataset_size 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --report_to wandb
```

#### incoder_6b_v12
```bash
CUDA_VISIBLE_DEVICES="3" python ./src/sg_finetune.py \
    --run_name incoder_6b_v12 \
    --model_name_or_path ~/WorkspaceLabModels/incoder_6b \
    --output_dir ~/WorkspaceLabModels/incoder_6b_v12 \
    --dataset ./data/finetune_training.jsonl \
    --validation_dataset ./data/finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0002 \
    --seed 120462 \
    --max_length 1024 \
    --per_device_train_batch_size 8 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --eval_steps 1000 \
    --save_steps 999999 \
    --specific_save_steps "13,125,1250,6250" \
    --eval_dataset_size 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --report_to wandb
```

#### codellama_7b_v12
```bash
CUDA_VISIBLE_DEVICES="0" python ./src/sg_finetune.py \
    --run_name codellama_7b_v12 \
    --model_name_or_path ~/WorkspaceLabModels/codellama_7b \
    --output_dir ~/WorkspaceLabModels/codellama_7b_v12 \
    --dataset ./data/finetune_training.jsonl \
    --validation_dataset ./data/finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0002 \
    --seed 120462 \
    --max_length 1024 \
    --per_device_train_batch_size 8 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --eval_steps 1000 \
    --save_steps 999999 \
    --specific_save_steps "13,125,1250,6250" \
    --eval_dataset_size 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --report_to wandb
```

### starcoder2_7b_v12
```bash
CUDA_VISIBLE_DEVICES="1" python ./src/sg_finetune.py \
    --run_name starcoder2_7b_v12 \
    --model_name_or_path ~/WorkspaceLabModels/starcoder2_7b \
    --output_dir ~/WorkspaceLabModels/starcoder2_7b_v12 \
    --dataset ./data/finetune_training.jsonl \
    --validation_dataset ./data/finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0002 \
    --seed 120462 \
    --max_length 1024 \
    --per_device_train_batch_size 8 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --eval_steps 1000 \
    --save_steps 999999 \
    --specific_save_steps "13,125,1250,6250" \
    --eval_dataset_size 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --report_to wandb
```

### qwen2.5_coder_7b_v12
```bash
CUDA_VISIBLE_DEVICES="2" python ./src/sg_finetune.py \
    --run_name qwen2.5_coder_7b_v12 \
    --model_name_or_path ~/WorkspaceLabModels/qwen2.5_coder_7b \
    --output_dir ~/WorkspaceLabModels/qwen2.5_coder_7b_v12 \
    --dataset ./data/finetune_training.jsonl \
    --validation_dataset ./data/finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0002 \
    --seed 120462 \
    --max_length 1024 \
    --per_device_train_batch_size 4 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --eval_steps 2000 \
    --save_steps 999999 \
    --specific_save_steps "25,250,2500,12500" \
    --eval_dataset_size 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --report_to wandb
```

#### codellama_13b_v12
```bash
CUDA_VISIBLE_DEVICES="3" python ./src/sg_finetune.py \
    --run_name codellama_13b_v12 \
    --model_name_or_path ~/WorkspaceLabModels/codellama_13b \
    --output_dir ~/WorkspaceLabModels/codellama_13b_v12 \
    --dataset ./data/finetune_training.jsonl \
    --validation_dataset ./data/finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0002 \
    --seed 120462 \
    --max_length 1024 \
    --per_device_train_batch_size 4 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --eval_steps 2000 \
    --save_steps 999999 \
    --specific_save_steps "25,250,2500,12500" \
    --eval_dataset_size 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --report_to wandb
```

#### codegen_16b_v12
```bash
CUDA_VISIBLE_DEVICES="1" python ./src/sg_finetune.py \
    --run_name codegen_16b_v12 \
    --model_name_or_path ~/WorkspaceLabModels/codegen_16b \
    --output_dir ~/WorkspaceLabModels/codegen_16b_v12 \
    --dataset ./data/finetune_training.jsonl \
    --validation_dataset ./data/finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0002 \
    --seed 120462 \
    --max_length 768 \
    --per_device_train_batch_size 4 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --eval_steps 2000 \
    --save_steps 999999 \
    --specific_save_steps "25,250,2500,12500" \
    --eval_dataset_size 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --report_to wandb
```
