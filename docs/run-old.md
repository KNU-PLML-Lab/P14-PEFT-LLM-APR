## Fine-tuning

### v5
#### codegen_6b_v5 (4090)
```bash
CUDA_VISIBLE_DEVICES="0" python sg_finetune.py \
    --run_name codegen_6b_v5 \
    --model_name_or_path ~/WorkspaceLabModels/codegen-6B \
    --output_dir ../../nosync/output/codegen_6b_v5 \
    --dataset ../../finetune_training.jsonl \
    --validation_dataset ../../finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0002 \
    --seed 0 \
    --max_length 768 \
    --per_device_train_batch_size 8 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --eval_steps 1000 \
    --eval_dataset_size 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --report_to wandb
```

#### incoder_6b_v5 (4090)
```bash
CUDA_VISIBLE_DEVICES="1" python sg_finetune.py \
    --run_name incoder_6b_v5 \
    --model_name_or_path ~/WorkspaceLabModels/incoder-6B \
    --output_dir ../../nosync/output/incoder_6b_v5 \
    --dataset ../../finetune_training.jsonl \
    --validation_dataset ../../finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0002 \
    --seed 0 \
    --max_length 1024 \
    --per_device_train_batch_size 8 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --eval_steps 1000 \
    --eval_dataset_size 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --report_to wandb
```

#### codellama_7b_v5 (4090)
```bash
CUDA_VISIBLE_DEVICES="2" python sg_finetune.py \
    --run_name codellama_7b_v5 \
    --model_name_or_path ~/WorkspaceLabModels/code_llama-7b-hf \
    --output_dir ../../nosync/output/codellama_7b_v5 \
    --dataset ../../finetune_training.jsonl \
    --validation_dataset ../../finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0002 \
    --seed 0 \
    --max_length 1024 \
    --per_device_train_batch_size 8 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --eval_steps 1000 \
    --eval_dataset_size 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --report_to wandb
```

#### codellama_13b_v5 (a6000)
```bash
CUDA_VISIBLE_DEVICES="0" python sg_finetune.py \
    --run_name codellama_13b_v5 \
    --model_name_or_path ~/WorkspaceLabModels/code_llama-13b-hf \
    --output_dir ../../nosync/output/codellama_13b_v5 \
    --dataset ../../finetune_training.jsonl \
    --validation_dataset ../../finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0002 \
    --seed 0 \
    --max_length 1024 \
    --per_device_train_batch_size 8 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --eval_steps 2000 \
    --eval_dataset_size 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --report_to wandb
```

#### codellama_34b_v5 (a6000)
```bash
CUDA_VISIBLE_DEVICES="1" python sg_finetune.py \
    --run_name codellama_34b_v5 \
    --model_name_or_path ~/WorkspaceLabModels/code_llama-34b-hf \
    --output_dir ../../nosync/output/codellama_34b_v5 \
    --dataset ../../finetune_training.jsonl \
    --validation_dataset ../../finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0001 \
    --seed 0 \
    --max_length 1024 \
    --per_device_train_batch_size 4 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --eval_steps 4000 \
    --eval_dataset_size 1000 \
    --max_eval_samples 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --lora_dropout 0.05 \
    --report_to wandb
```

### v8 (bf16 seed72)

#### codegen_6b_v8 (4090)
```bash
CUDA_VISIBLE_DEVICES="0" python ./src/sg_finetune.py \
    --run_name codegen_6b_v8 \
    --model_name_or_path ~/WorkspaceLabModels/codegen_6b \
    --output_dir ~/WorkspaceLabModels/codegen_6b_v8 \
    --dataset ./data/finetune_training.jsonl \
    --validation_dataset ./data/finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0002 \
    --seed 72 \
    --max_length 768 \
    --per_device_train_batch_size 8 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --eval_steps 1000 \
    --save_steps 1000 \
    --specific_save_steps "13,125,1250,6250" \
    --eval_dataset_size 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --report_to wandb
```

#### incoder_6b_v8_2 (4090) (code fix)
```bash
CUDA_VISIBLE_DEVICES="1" python ./src/sg_finetune.py \
    --run_name incoder_6b_v8_2 \
    --model_name_or_path ~/WorkspaceLabModels/incoder_6b \
    --output_dir ~/WorkspaceLabModels/incoder_6b_v8_2 \
    --dataset ./data/finetune_training.jsonl \
    --validation_dataset ./data/finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0002 \
    --seed 72 \
    --max_length 1024 \
    --per_device_train_batch_size 8 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --eval_steps 1000 \
    --save_steps 1000 \
    --specific_save_steps "13,125,1250,6250" \
    --eval_dataset_size 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --report_to wandb
```

#### codellama_7b_v8 (4090)
```bash
CUDA_VISIBLE_DEVICES="2" python ./src/sg_finetune.py \
    --run_name codellama_7b_v8 \
    --model_name_or_path ~/WorkspaceLabModels/codellama_7b \
    --output_dir ~/WorkspaceLabModels/codellama_7b_v8 \
    --dataset ./data/finetune_training.jsonl \
    --validation_dataset ./data/finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0002 \
    --seed 72 \
    --max_length 1024 \
    --per_device_train_batch_size 8 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --eval_steps 1000 \
    --save_steps 1000 \
    --specific_save_steps "13,125,1250,6250" \
    --eval_dataset_size 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --report_to wandb
```

#### codellama_13b_v8 (A6000)
```bash
CUDA_VISIBLE_DEVICES="0" python ./src/sg_finetune.py \
    --run_name codellama_13b_v8 \
    --model_name_or_path ~/WorkspaceLabModels/codellama_13b \
    --output_dir ~/WorkspaceLabModels/codellama_13b_v8 \
    --dataset ./data/finetune_training.jsonl \
    --validation_dataset ./data/finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0002 \
    --seed 72 \
    --max_length 1024 \
    --per_device_train_batch_size 8 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --eval_steps 1000 \
    --save_steps 1000 \
    --specific_save_steps "13,125,1250,6250" \
    --eval_dataset_size 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --report_to wandb
```

#### codellama_34b_v8 (A6000)
```bash
CUDA_VISIBLE_DEVICES="1" python ./src/sg_finetune.py \
    --run_name codellama_34b_v8 \
    --model_name_or_path ~/WorkspaceLabModels/codellama_34b \
    --output_dir ~/WorkspaceLabModels/codellama_34b_v8 \
    --dataset ./data/finetune_training.jsonl \
    --validation_dataset ./data/finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0001 \
    --seed 72 \
    --max_length 1024 \
    --per_device_train_batch_size 4 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --eval_steps 2000 \
    --save_steps 2000 \
    --specific_save_steps "25,250,2500,12500" \
    --eval_dataset_size 1000 \
    --max_eval_samples 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --lora_dropout 0.05 \
    --report_to wandb
```

#### codellama_34b_v8 (A6000)
```bash
CUDA_VISIBLE_DEVICES="1" python ./src/sg_finetune.py \
    --run_name codellama_34b_v8 \
    --model_name_or_path ~/WorkspaceLabModels/codellama_34b \
    --output_dir ~/WorkspaceLabModels/codellama_34b_v8 \
    --dataset ./data/finetune_training.jsonl \
    --validation_dataset ./data/finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0001 \
    --seed 72 \
    --max_length 1024 \
    --per_device_train_batch_size 4 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --eval_steps 2000 \
    --save_steps 2000 \
    --specific_save_steps "25,250,2500,12500" \
    --eval_dataset_size 1000 \
    --max_eval_samples 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --lora_dropout 0.05 \
    --report_to wandb
```

### deepseek_coder_v2_lite_base_v8 (A6000)
```bash
CUDA_VISIBLE_DEVICES="0" python ./src/sg_finetune.py \
    --run_name deepseek_coder_v2_lite_base_v8 \
    --model_name_or_path ~/WorkspaceLabModels/deepseek_coder_v2_lite_base \
    --output_dir ~/WorkspaceLabModels/deepseek_coder_v2_lite_base_v8 \
    --dataset ./data/finetune_training.jsonl \
    --validation_dataset ./data/finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0002 \
    --seed 72 \
    --max_length 1024 \
    --per_device_train_batch_size 8 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --eval_steps 1000 \
    --save_steps 1000 \
    --specific_save_steps "13,125,1250,6250" \
    --eval_dataset_size 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --trust_remote_code \
    --report_to wandb
```

### v9 (bf16 seed1761)

#### codegen_6b_v9 (4090)
```bash
CUDA_VISIBLE_DEVICES="0" python ./src/sg_finetune.py \
    --run_name codegen_6b_v9 \
    --model_name_or_path ~/WorkspaceLabModels/codegen_6b \
    --output_dir ~/WorkspaceLabModels/codegen_6b_v9 \
    --dataset ./data/finetune_training.jsonl \
    --validation_dataset ./data/finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0002 \
    --seed 1761 \
    --max_length 768 \
    --per_device_train_batch_size 8 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --eval_steps 1000 \
    --save_steps 1000 \
    --specific_save_steps "13,125,1250,6250" \
    --eval_dataset_size 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --report_to wandb
```

#### incoder_6b_v9 (4090)
```bash
CUDA_VISIBLE_DEVICES="1" python ./src/sg_finetune.py \
    --run_name incoder_6b_v9 \
    --model_name_or_path ~/WorkspaceLabModels/incoder_6b \
    --output_dir ~/WorkspaceLabModels/incoder_6b_v9 \
    --dataset ./data/finetune_training.jsonl \
    --validation_dataset ./data/finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0002 \
    --seed 1761 \
    --max_length 1024 \
    --per_device_train_batch_size 8 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --eval_steps 1000 \
    --save_steps 1000 \
    --specific_save_steps "13,125,1250,6250" \
    --eval_dataset_size 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --report_to wandb
```

#### codellama_7b_v9 (4090)
```bash
CUDA_VISIBLE_DEVICES="2" python ./src/sg_finetune.py \
    --run_name codellama_7b_v9 \
    --model_name_or_path ~/WorkspaceLabModels/codellama_7b \
    --output_dir ~/WorkspaceLabModels/codellama_7b_v9 \
    --dataset ./data/finetune_training.jsonl \
    --validation_dataset ./data/finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0002 \
    --seed 1761 \
    --max_length 1024 \
    --per_device_train_batch_size 8 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --eval_steps 1000 \
    --save_steps 1000 \
    --specific_save_steps "13,125,1250,6250" \
    --eval_dataset_size 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --report_to wandb
```

### deepseek_coder_v2_lite_base_v9 (A6000)
```bash
CUDA_VISIBLE_DEVICES="1" python ./src/sg_finetune.py \
    --run_name deepseek_coder_v2_lite_base_v9 \
    --model_name_or_path ~/WorkspaceLabModels/deepseek_coder_v2_lite_base \
    --output_dir ~/WorkspaceLabModels/deepseek_coder_v2_lite_base_v9 \
    --dataset ./data/finetune_training.jsonl \
    --validation_dataset ./data/finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0002 \
    --seed 1761 \
    --max_length 1024 \
    --per_device_train_batch_size 8 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --eval_steps 1000 \
    --save_steps 1000 \
    --specific_save_steps "13,125,1250,6250" \
    --eval_dataset_size 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --trust_remote_code \
    --report_to wandb
```

#### codellama_34b_v8_ab (A6000)
```bash
CUDA_VISIBLE_DEVICES="0" python ./src/sg_finetune.py \
    --run_name codellama_34b_v8_ab \
    --model_name_or_path ~/WorkspaceLabModels/codellama_34b \
    --output_dir ~/WorkspaceLabModels/codellama_34b_v8_ab \
    --dataset ./data/finetune_training.jsonl \
    --validation_dataset ./data/finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0002 \
    --seed 72 \
    --max_length 1024 \
    --per_device_train_batch_size 4 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --eval_steps 2000 \
    --save_steps 2000 \
    --specific_save_steps "25,250,2500,12500" \
    --eval_dataset_size 1000 \
    --max_eval_samples 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --report_to wandb
```

#### codellama_34b_v9 (A6000)
```bash
CUDA_VISIBLE_DEVICES="0" python ./src/sg_finetune.py \
    --run_name codellama_34b_v9 \
    --model_name_or_path ~/WorkspaceLabModels/codellama_34b \
    --output_dir ~/WorkspaceLabModels/codellama_34b_v9 \
    --dataset ./data/finetune_training.jsonl \
    --validation_dataset ./data/finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0001 \
    --seed 1761 \
    --max_length 1024 \
    --per_device_train_batch_size 4 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --eval_steps 2000 \
    --save_steps 2000 \
    --specific_save_steps "25,250,2500,12500" \
    --eval_dataset_size 1000 \
    --max_eval_samples 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --lora_dropout 0.05 \
    --report_to wandb
```

#### codegen_16b_v8 (4090)
```bash
CUDA_VISIBLE_DEVICES="0" python ./src/sg_finetune.py \
    --run_name codegen_16b_v8 \
    --model_name_or_path ~/WorkspaceLabModels/codegen_16b \
    --output_dir ~/WorkspaceLabModels/codegen_16b_v8 \
    --dataset ./data/finetune_training.jsonl \
    --validation_dataset ./data/finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0002 \
    --seed 72 \
    --max_length 768 \
    --per_device_train_batch_size 4 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --eval_steps 2000 \
    --save_steps 2000 \
    --specific_save_steps "25,250,2500,12500" \
    --eval_dataset_size 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --report_to wandb
```

#### codegen_16b_v9 (4090)
```bash
CUDA_VISIBLE_DEVICES="1" python ./src/sg_finetune.py \
    --run_name codegen_16b_v9 \
    --model_name_or_path ~/WorkspaceLabModels/codegen_16b \
    --output_dir ~/WorkspaceLabModels/codegen_16b_v9 \
    --dataset ./data/finetune_training.jsonl \
    --validation_dataset ./data/finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0002 \
    --seed 1761 \
    --max_length 768 \
    --per_device_train_batch_size 4 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --eval_steps 2000 \
    --save_steps 2000 \
    --specific_save_steps "25,250,2500,12500" \
    --eval_dataset_size 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --report_to wandb
```

### qwen2.5_coder_7b_v8 (4090)
```bash
CUDA_VISIBLE_DEVICES="2" python ./src/sg_finetune.py \
    --run_name qwen2.5_coder_7b_v8 \
    --model_name_or_path ~/WorkspaceLabModels/qwen2.5_coder_7b \
    --output_dir ~/WorkspaceLabModels/qwen2.5_coder_7b_v8 \
    --dataset ./data/finetune_training.jsonl \
    --validation_dataset ./data/finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0002 \
    --seed 72 \
    --max_length 1024 \
    --per_device_train_batch_size 4 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --eval_steps 2000 \
    --save_steps 2000 \
    --specific_save_steps "25,250,2500,12500" \
    --eval_dataset_size 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --report_to wandb
```

### starcoder2_7b_v8 (4090)
```bash
CUDA_VISIBLE_DEVICES="3" python ./src/sg_finetune.py \
    --run_name starcoder2_7b_v8 \
    --model_name_or_path ~/WorkspaceLabModels/starcoder2_7b \
    --output_dir ~/WorkspaceLabModels/starcoder2_7b_v8 \
    --dataset ./data/finetune_training.jsonl \
    --validation_dataset ./data/finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0002 \
    --seed 72 \
    --max_length 1024 \
    --per_device_train_batch_size 8 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --eval_steps 1000 \
    --save_steps 1000 \
    --specific_save_steps "13,125,1250,6250" \
    --eval_dataset_size 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --report_to wandb
```

### starcoder2_7b_v9 (4090)
```bash
CUDA_VISIBLE_DEVICES="0" python ./src/sg_finetune.py \
    --run_name starcoder2_7b_v9 \
    --model_name_or_path ~/WorkspaceLabModels/starcoder2_7b \
    --output_dir ~/WorkspaceLabModels/starcoder2_7b_v9 \
    --dataset ./data/finetune_training.jsonl \
    --validation_dataset ./data/finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0002 \
    --seed 1761 \
    --max_length 1024 \
    --per_device_train_batch_size 8 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --eval_steps 1000 \
    --save_steps 1000 \
    --specific_save_steps "13,125,1250,6250" \
    --eval_dataset_size 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --report_to wandb
```

#### codegen_16b_v8_2 (4090)
```bash
CUDA_VISIBLE_DEVICES="0" python ./src/sg_finetune.py \
    --run_name codegen_16b_v8_2 \
    --model_name_or_path ~/WorkspaceLabModels/codegen_16b \
    --output_dir ~/WorkspaceLabModels/codegen_16b_v8_2 \
    --dataset ./data/finetune_training.jsonl \
    --validation_dataset ./data/finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0002 \
    --seed 72 \
    --max_length 1024 \
    --per_device_train_batch_size 4 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --eval_steps 2000 \
    --save_steps 2000 \
    --specific_save_steps "25,250,2500,12500" \
    --eval_dataset_size 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --report_to wandb
```

#### codegen_16b_v9_2 (4090)
```bash
CUDA_VISIBLE_DEVICES="1" python ./src/sg_finetune.py \
    --run_name codegen_16b_v9_2 \
    --model_name_or_path ~/WorkspaceLabModels/codegen_16b \
    --output_dir ~/WorkspaceLabModels/codegen_16b_v9_2 \
    --dataset ./data/finetune_training.jsonl \
    --validation_dataset ./data/finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0002 \
    --seed 1761 \
    --max_length 1024 \
    --per_device_train_batch_size 4 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --eval_steps 2000 \
    --save_steps 2000 \
    --specific_save_steps "25,250,2500,12500" \
    --eval_dataset_size 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --report_to wandb
```

### deepseek_coder_v2_lite_base_v10 (A6000)
```bash
CUDA_VISIBLE_DEVICES="1" python ./src/sg_finetune.py \
    --run_name deepseek_coder_v2_lite_base_v10 \
    --model_name_or_path ~/WorkspaceLabModels/deepseek_coder_v2_lite_base \
    --output_dir ~/WorkspaceLabModels/deepseek_coder_v2_lite_base_v10 \
    --dataset ./data/finetune_training.jsonl \
    --validation_dataset ./data/finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0002 \
    --seed 122 \
    --max_length 1024 \
    --per_device_train_batch_size 8 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --eval_steps 1000 \
    --save_steps 1000 \
    --specific_save_steps "13,125,1250,6250" \
    --eval_dataset_size 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --trust_remote_code \
    --report_to wandb
```

### qwen2.5_coder_7b_v9 (4090)
```bash
CUDA_VISIBLE_DEVICES="0" python ./src/sg_finetune.py \
    --run_name qwen2.5_coder_7b_v9 \
    --model_name_or_path ~/WorkspaceLabModels/qwen2.5_coder_7b \
    --output_dir ~/WorkspaceLabModels/qwen2.5_coder_7b_v9 \
    --dataset ./data/finetune_training.jsonl \
    --validation_dataset ./data/finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0002 \
    --seed 1761 \
    --max_length 1024 \
    --per_device_train_batch_size 4 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --eval_steps 2000 \
    --save_steps 2000 \
    --specific_save_steps "25,250,2500,12500" \
    --eval_dataset_size 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --report_to wandb
```

#### codellama_13b_v10 (4090)
```bash
CUDA_VISIBLE_DEVICES="3" python ./src/sg_finetune.py \
    --run_name codellama_13b_v10 \
    --model_name_or_path ~/WorkspaceLabModels/codellama_13b \
    --output_dir ~/WorkspaceLabModels/codellama_13b_v10 \
    --dataset ./data/finetune_training.jsonl \
    --validation_dataset ./data/finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0002 \
    --seed 122 \
    --max_length 1024 \
    --per_device_train_batch_size 4 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --eval_steps 2000 \
    --save_steps 2000 \
    --specific_save_steps "25,250,2500,12500" \
    --eval_dataset_size 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --report_to wandb
```

#### codegen_16b_v10 (4090)
```bash
CUDA_VISIBLE_DEVICES="1" python ./src/sg_finetune.py \
    --run_name codegen_16b_v10 \
    --model_name_or_path ~/WorkspaceLabModels/codegen_16b \
    --output_dir ~/WorkspaceLabModels/codegen_16b_v10 \
    --dataset ./data/finetune_training.jsonl \
    --validation_dataset ./data/finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0002 \
    --seed 122 \
    --max_length 768 \
    --per_device_train_batch_size 4 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --eval_steps 2000 \
    --save_steps 2000 \
    --specific_save_steps "25,250,2500,12500" \
    --eval_dataset_size 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --report_to wandb
```

### starcoder2_7b_v10 (4090)
```bash
CUDA_VISIBLE_DEVICES="2" python ./src/sg_finetune.py \
    --run_name starcoder2_7b_v10 \
    --model_name_or_path ~/WorkspaceLabModels/starcoder2_7b \
    --output_dir ~/WorkspaceLabModels/starcoder2_7b_v10 \
    --dataset ./data/finetune_training.jsonl \
    --validation_dataset ./data/finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0002 \
    --seed 122 \
    --max_length 1024 \
    --per_device_train_batch_size 8 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --eval_steps 1000 \
    --save_steps 1000 \
    --specific_save_steps "13,125,1250,6250" \
    --eval_dataset_size 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --report_to wandb
```

### qwen2.5_coder_7b_v10 (4090)
```bash
CUDA_VISIBLE_DEVICES="3" python ./src/sg_finetune.py \
    --run_name qwen2.5_coder_7b_v10 \
    --model_name_or_path ~/WorkspaceLabModels/qwen2.5_coder_7b \
    --output_dir ~/WorkspaceLabModels/qwen2.5_coder_7b_v10 \
    --dataset ./data/finetune_training.jsonl \
    --validation_dataset ./data/finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0002 \
    --seed 122 \
    --max_length 1024 \
    --per_device_train_batch_size 4 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --eval_steps 2000 \
    --save_steps 2000 \
    --specific_save_steps "25,250,2500,12500" \
    --eval_dataset_size 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --report_to wandb
```

#### codegen_6b_v10 (4090)
```bash
CUDA_VISIBLE_DEVICES="0" python ./src/sg_finetune.py \
    --run_name codegen_6b_v10 \
    --model_name_or_path ~/WorkspaceLabModels/codegen_6b \
    --output_dir ~/WorkspaceLabModels/codegen_6b_v10 \
    --dataset ./data/finetune_training.jsonl \
    --validation_dataset ./data/finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0002 \
    --seed 122 \
    --max_length 768 \
    --per_device_train_batch_size 8 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --eval_steps 1000 \
    --save_steps 1000 \
    --specific_save_steps "13,125,1250,6250" \
    --eval_dataset_size 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --report_to wandb
```

#### incoder_6b_v10 (4090)
```bash
CUDA_VISIBLE_DEVICES="1" python ./src/sg_finetune.py \
    --run_name incoder_6b_v10 \
    --model_name_or_path ~/WorkspaceLabModels/incoder_6b \
    --output_dir ~/WorkspaceLabModels/incoder_6b_v10 \
    --dataset ./data/finetune_training.jsonl \
    --validation_dataset ./data/finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0002 \
    --seed 122 \
    --max_length 1024 \
    --per_device_train_batch_size 8 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --eval_steps 1000 \
    --save_steps 1000 \
    --specific_save_steps "13,125,1250,6250" \
    --eval_dataset_size 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --report_to wandb
```

#### codellama_7b_v10 (4090)
```bash
CUDA_VISIBLE_DEVICES="2" python ./src/sg_finetune.py \
    --run_name codellama_7b_v10 \
    --model_name_or_path ~/WorkspaceLabModels/codellama_7b \
    --output_dir ~/WorkspaceLabModels/codellama_7b_v10 \
    --dataset ./data/finetune_training.jsonl \
    --validation_dataset ./data/finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0002 \
    --seed 122 \
    --max_length 1024 \
    --per_device_train_batch_size 8 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --eval_steps 1000 \
    --save_steps 1000 \
    --specific_save_steps "13,125,1250,6250" \
    --eval_dataset_size 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --report_to wandb
```

#### codellama_13b_v10 (4090)
```bash
CUDA_VISIBLE_DEVICES="3" python ./src/sg_finetune.py \
    --run_name codellama_13b_v10 \
    --model_name_or_path ~/WorkspaceLabModels/codellama_13b \
    --output_dir ~/WorkspaceLabModels/codellama_13b_v10 \
    --dataset ./data/finetune_training.jsonl \
    --validation_dataset ./data/finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0002 \
    --seed 122 \
    --max_length 1024 \
    --per_device_train_batch_size 4 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --eval_steps 2000 \
    --save_steps 2000 \
    --specific_save_steps "25,250,2500,12500" \
    --eval_dataset_size 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --report_to wandb
```

#### codellama_13b_v9 (4090)
```bash
CUDA_VISIBLE_DEVICES="2" python ./src/sg_finetune.py \
    --run_name codellama_13b_v9 \
    --model_name_or_path ~/WorkspaceLabModels/codellama_13b \
    --output_dir ~/WorkspaceLabModels/codellama_13b_v9 \
    --dataset ./data/finetune_training.jsonl \
    --validation_dataset ./data/finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0002 \
    --seed 1761 \
    --max_length 1024 \
    --per_device_train_batch_size 4 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --eval_steps 2000 \
    --save_steps 2000 \
    --specific_save_steps "25,250,2500,12500" \
    --eval_dataset_size 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --report_to wandb
```

#### codellama_34b_v10_2 (A6000)
```bash
CUDA_VISIBLE_DEVICES="1" python ./src/sg_finetune.py \
    --run_name codellama_34b_v10_2 \
    --model_name_or_path ~/WorkspaceLabModels/codellama_34b \
    --output_dir ~/WorkspaceLabModels/codellama_34b_v10_2 \
    --dataset ./data/finetune_training.jsonl \
    --validation_dataset ./data/finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0001 \
    --seed 122 \
    --max_length 1024 \
    --per_device_train_batch_size 4 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --eval_steps 2000 \
    --save_steps 2000 \
    --specific_save_steps "25,250,2500,12500" \
    --eval_dataset_size 1000 \
    --max_eval_samples 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --lora_dropout 0.05 \
    --report_to wandb
```

#### codellama_7b_v10_paramtest (4090)
```bash
CUDA_VISIBLE_DEVICES="0" python ./src/sg_finetune.py \
    --run_name codellama_7b_v10_paramtest \
    --model_name_or_path ~/WorkspaceLabModels/codellama_7b \
    --output_dir ~/WorkspaceLabModels/codellama_7b_v10_paramtest \
    --dataset ./data/finetune_training.jsonl \
    --validation_dataset ./data/finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0002 \
    --seed 122 \
    --max_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --eval_steps 1000 \
    --save_steps 1000 \
    --specific_save_steps "13,125,1250,6250" \
    --eval_dataset_size 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10
```





## Benchmarks
```bash
# --do_humaneval
# --do_quixbugs
# --do_defects4j --strict_defects4j --validate_result_split_defects4j

# --do_generate
# --do_validate

# do_defects4j 추가
CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_7b \
  --output_dir ~/WorkspaceLabModels/codellama_7b_v5 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_generate \
  --do_validate



CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_7b \
  --output_dir ~/WorkspaceLabModels/codellama_7b_v5 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_defects4j \
  --do_validate

# codellama_13b_v5 집중 테스트
CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_13b \
  --output_dir ~/WorkspaceLabModels/codellama_13b_v5 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_generate \
  --do_validate

CUDA_VISIBLE_DEVICES="1" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_13b \
  --output_dir ~/WorkspaceLabModels/codellama_13b_v5 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_quixbugs \
  --do_generate \
  --do_validate

CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_13b \
  --output_dir ~/WorkspaceLabModels/codellama_13b_v5 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_defects4j \
  --do_generate \
  --do_validate

# codellama_34b_v5 집중 테스트
CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_34b \
  --output_dir ~/WorkspaceLabModels/codellama_34b_v5 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_generate \
  --do_validate

CUDA_VISIBLE_DEVICES="1" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_34b \
  --output_dir ~/WorkspaceLabModels/codellama_34b_v5 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_quixbugs \
  --do_generate \
  --do_validate

CUDA_VISIBLE_DEVICES="1" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_34b \
  --output_dir ~/WorkspaceLabModels/codellama_34b_v5 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_defects4j \
  --do_generate \
  --do_validate

# v8
CUDA_VISIBLE_DEVICES="2" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_7b \
  --output_dir ~/WorkspaceLabModels/codellama_7b_v8 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_defects4j --strict_defects4j \
  --do_generate \
  --do_validate

CUDA_VISIBLE_DEVICES="1" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_13b \
  --output_dir ~/WorkspaceLabModels/codellama_13b_v5 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_defects4j \
  --strict_defects4j \
  --do_validate

CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_34b \
  --output_dir ~/WorkspaceLabModels/codellama_34b_v5 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_defects4j \
  --strict_defects4j \
  --do_validate

# codellama_13b_v8
CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_13b \
  --output_dir ~/WorkspaceLabModels/codellama_13b_v8 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_quixbugs \
  --do_defects4j --strict_defects4j \
  --do_generate

CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_13b \
  --output_dir ~/WorkspaceLabModels/codellama_13b_v8 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_validate

CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_13b \
  --output_dir ~/WorkspaceLabModels/codellama_13b_v8 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_quixbugs \
  --do_validate

CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_13b \
  --output_dir ~/WorkspaceLabModels/codellama_13b_v8 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_defects4j --strict_defects4j --validate_result_split_defects4j \
  --do_validate

# v9
CUDA_VISIBLE_DEVICES="2" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_7b \
  --output_dir ~/WorkspaceLabModels/codellama_7b_v9 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_quixbugs \
  --do_defects4j --strict_defects4j \
  --do_generate

# v9 validate
CUDA_VISIBLE_DEVICES="1" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_7b \
  --output_dir ~/WorkspaceLabModels/codellama_7b_v9 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_validate

CUDA_VISIBLE_DEVICES="1" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_7b \
  --output_dir ~/WorkspaceLabModels/codellama_7b_v9 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_quixbugs \
  --do_validate

CUDA_VISIBLE_DEVICES="1" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_7b \
  --output_dir ~/WorkspaceLabModels/codellama_7b_v9 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_defects4j --strict_defects4j --validate_result_split_defects4j \
  --do_validate

# old v5 validate splitter
CUDA_VISIBLE_DEVICES="1" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_7b \
  --output_dir ~/WorkspaceLabModels/codellama_7b_v5 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_defects4j --validate_result_split_defects4j

CUDA_VISIBLE_DEVICES="2" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_13b \
  --output_dir ~/WorkspaceLabModels/codellama_13b_v5 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_defects4j --validate_result_split_defects4j

# codellama_7b_v9 humaneval (v5이 중간값이지만, 시드가 없으며, checkpoint가 누락되어서 v9 대신 사용) 스탭별 테스트
# gen
CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_7b \
  --output_dir ~/WorkspaceLabModels/codellama_7b_v9/checkpoint-13 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_generate

CUDA_VISIBLE_DEVICES="1" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_7b \
  --output_dir ~/WorkspaceLabModels/codellama_7b_v9/checkpoint-125 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_generate

CUDA_VISIBLE_DEVICES="2" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_7b \
  --output_dir ~/WorkspaceLabModels/codellama_7b_v9/checkpoint-1250 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_generate

CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_7b \
  --output_dir ~/WorkspaceLabModels/codellama_7b_v9/checkpoint-6250 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_generate

# val
CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_7b \
  --output_dir ~/WorkspaceLabModels/codellama_7b_v9/checkpoint-13 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_validate

CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_7b \
  --output_dir ~/WorkspaceLabModels/codellama_7b_v9/checkpoint-125 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_validate

CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_7b \
  --output_dir ~/WorkspaceLabModels/codellama_7b_v9/checkpoint-1250 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_validate

CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_7b \
  --output_dir ~/WorkspaceLabModels/codellama_7b_v9/checkpoint-6250 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_validate

# codellama_34b_v8
CUDA_VISIBLE_DEVICES="1" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_34b \
  --output_dir ~/WorkspaceLabModels/codellama_34b_v8 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_quixbugs \
  --do_defects4j --strict_defects4j \
  --do_generate

CUDA_VISIBLE_DEVICES="1" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_34b \
  --output_dir ~/WorkspaceLabModels/codellama_34b_v8 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_validate

CUDA_VISIBLE_DEVICES="1" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_34b \
  --output_dir ~/WorkspaceLabModels/codellama_34b_v8 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_quixbugs \
  --do_validate

CUDA_VISIBLE_DEVICES="1" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_34b \
  --output_dir ~/WorkspaceLabModels/codellama_34b_v8 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_defects4j --strict_defects4j --validate_result_split_defects4j \
  --do_validate

# deepseek_coder_v2_lite_base_v8
CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/deepseek_coder_v2_lite_base \
  --output_dir ~/WorkspaceLabModels/deepseek_coder_v2_lite_base_v8 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_humaneval \
  --do_quixbugs \
  --do_defects4j --strict_defects4j \
  --do_generate

CUDA_VISIBLE_DEVICES="2" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/deepseek_coder_v2_lite_base \
  --output_dir ~/WorkspaceLabModels/deepseek_coder_v2_lite_base_v8 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_humaneval \
  --do_validate

CUDA_VISIBLE_DEVICES="2" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/deepseek_coder_v2_lite_base \
  --output_dir ~/WorkspaceLabModels/deepseek_coder_v2_lite_base_v8 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_quixbugs \
  --do_validate

CUDA_VISIBLE_DEVICES="2" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/deepseek_coder_v2_lite_base \
  --output_dir ~/WorkspaceLabModels/deepseek_coder_v2_lite_base_v8 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_defects4j --strict_defects4j --validate_result_split_defects4j \
  --do_validate

# deepseek_coder_v2_lite_base_v9
CUDA_VISIBLE_DEVICES="1" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/deepseek_coder_v2_lite_base \
  --output_dir ~/WorkspaceLabModels/deepseek_coder_v2_lite_base_v9 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_humaneval \
  --do_quixbugs \
  --do_defects4j --strict_defects4j \
  --do_generate

CUDA_VISIBLE_DEVICES="2" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/deepseek_coder_v2_lite_base \
  --output_dir ~/WorkspaceLabModels/deepseek_coder_v2_lite_base_v9 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_humaneval \
  --do_validate

CUDA_VISIBLE_DEVICES="2" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/deepseek_coder_v2_lite_base \
  --output_dir ~/WorkspaceLabModels/deepseek_coder_v2_lite_base_v9 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_quixbugs \
  --do_validate

CUDA_VISIBLE_DEVICES="2" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/deepseek_coder_v2_lite_base \
  --output_dir ~/WorkspaceLabModels/deepseek_coder_v2_lite_base_v9 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_defects4j --strict_defects4j --validate_result_split_defects4j \
  --do_validate

# codegen_16b_v8
CUDA_VISIBLE_DEVICES="2" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codegen_16b \
  --output_dir ~/WorkspaceLabModels/codegen_16b_v8 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_humaneval \
  --do_quixbugs \
  --do_defects4j --strict_defects4j \
  --do_generate

CUDA_VISIBLE_DEVICES="2" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codegen_16b \
  --output_dir ~/WorkspaceLabModels/codegen_16b_v8 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_humaneval \
  --do_validate

CUDA_VISIBLE_DEVICES="2" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codegen_16b \
  --output_dir ~/WorkspaceLabModels/codegen_16b_v8 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_quixbugs \
  --do_validate

CUDA_VISIBLE_DEVICES="2" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codegen_16b \
  --output_dir ~/WorkspaceLabModels/codegen_16b_v8 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_defects4j --strict_defects4j --validate_result_split_defects4j \
  --do_validate

# qwen2.5_coder_7b_v8
CUDA_VISIBLE_DEVICES="2" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/qwen2.5_coder_7b \
  --output_dir ~/WorkspaceLabModels/qwen2.5_coder_7b_v8 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_humaneval \
  --do_quixbugs \
  --do_defects4j --strict_defects4j \
  --do_generate

CUDA_VISIBLE_DEVICES="2" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/qwen2.5_coder_7b \
  --output_dir ~/WorkspaceLabModels/qwen2.5_coder_7b_v8 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_humaneval \
  --do_validate

CUDA_VISIBLE_DEVICES="2" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/qwen2.5_coder_7b \
  --output_dir ~/WorkspaceLabModels/qwen2.5_coder_7b_v8 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_quixbugs \
  --do_validate

CUDA_VISIBLE_DEVICES="2" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/qwen2.5_coder_7b \
  --output_dir ~/WorkspaceLabModels/qwen2.5_coder_7b_v8 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_defects4j --strict_defects4j --validate_result_split_defects4j \
  --do_validate

# starcoder2_7b_v8
CUDA_VISIBLE_DEVICES="2" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/starcoder2_7b \
  --output_dir ~/WorkspaceLabModels/starcoder2_7b_v8 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_humaneval \
  --do_quixbugs \
  --do_defects4j --strict_defects4j \
  --do_generate

CUDA_VISIBLE_DEVICES="2" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/starcoder2_7b \
  --output_dir ~/WorkspaceLabModels/starcoder2_7b_v8 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_humaneval \
  --do_validate

CUDA_VISIBLE_DEVICES="2" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/starcoder2_7b \
  --output_dir ~/WorkspaceLabModels/starcoder2_7b_v8 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_quixbugs \
  --do_validate

CUDA_VISIBLE_DEVICES="2" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/starcoder2_7b \
  --output_dir ~/WorkspaceLabModels/starcoder2_7b_v8 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_defects4j --strict_defects4j --validate_result_split_defects4j \
  --do_validate

# codegen_16b_v8_2
CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codegen_16b \
  --output_dir ~/WorkspaceLabModels/codegen_16b_v8_2 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_humaneval \
  --do_quixbugs \
  --do_defects4j --strict_defects4j \
  --do_generate

CUDA_VISIBLE_DEVICES="2" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codegen_16b \
  --output_dir ~/WorkspaceLabModels/codegen_16b_v8_2 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_humaneval \
  --do_validate

CUDA_VISIBLE_DEVICES="2" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codegen_16b \
  --output_dir ~/WorkspaceLabModels/codegen_16b_v8_2 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_quixbugs \
  --do_validate

CUDA_VISIBLE_DEVICES="2" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codegen_16b \
  --output_dir ~/WorkspaceLabModels/codegen_16b_v8_2 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_defects4j --strict_defects4j --validate_result_split_defects4j \
  --do_validate

# codegen_16b_v9_2
CUDA_VISIBLE_DEVICES="1" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codegen_16b \
  --output_dir ~/WorkspaceLabModels/codegen_16b_v9_2 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_humaneval \
  --do_quixbugs \
  --do_defects4j --strict_defects4j \
  --do_generate

CUDA_VISIBLE_DEVICES="2" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codegen_16b \
  --output_dir ~/WorkspaceLabModels/codegen_16b_v9_2 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_humaneval \
  --do_validate

CUDA_VISIBLE_DEVICES="2" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codegen_16b \
  --output_dir ~/WorkspaceLabModels/codegen_16b_v9_2 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_quixbugs \
  --do_validate

CUDA_VISIBLE_DEVICES="2" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codegen_16b \
  --output_dir ~/WorkspaceLabModels/codegen_16b_v9_2 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_defects4j --strict_defects4j --validate_result_split_defects4j \
  --do_validate

# starcoder2_7b_v9
CUDA_VISIBLE_DEVICES="2" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/starcoder2_7b \
  --output_dir ~/WorkspaceLabModels/starcoder2_7b_v9 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_humaneval \
  --do_quixbugs \
  --do_defects4j --strict_defects4j \
  --do_generate

CUDA_VISIBLE_DEVICES="2" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/starcoder2_7b \
  --output_dir ~/WorkspaceLabModels/starcoder2_7b_v9 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_humaneval \
  --do_validate

CUDA_VISIBLE_DEVICES="2" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/starcoder2_7b \
  --output_dir ~/WorkspaceLabModels/starcoder2_7b_v9 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_quixbugs \
  --do_validate

CUDA_VISIBLE_DEVICES="2" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/starcoder2_7b \
  --output_dir ~/WorkspaceLabModels/starcoder2_7b_v9 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_defects4j --strict_defects4j --validate_result_split_defects4j \
  --do_validate

# qwen2.5_coder_7b_v9
CUDA_VISIBLE_DEVICES="3" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/qwen2.5_coder_7b \
  --output_dir ~/WorkspaceLabModels/qwen2.5_coder_7b_v9 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_humaneval \
  --do_quixbugs \
  --do_defects4j --strict_defects4j \
  --do_generate

CUDA_VISIBLE_DEVICES="2" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/qwen2.5_coder_7b \
  --output_dir ~/WorkspaceLabModels/qwen2.5_coder_7b_v9 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_humaneval \
  --do_validate

CUDA_VISIBLE_DEVICES="2" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/qwen2.5_coder_7b \
  --output_dir ~/WorkspaceLabModels/qwen2.5_coder_7b_v9 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_quixbugs \
  --do_validate

CUDA_VISIBLE_DEVICES="2" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/qwen2.5_coder_7b \
  --output_dir ~/WorkspaceLabModels/qwen2.5_coder_7b_v9 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_defects4j --strict_defects4j --validate_result_split_defects4j \
  --do_validate

# codellama_34b_v8_ab
CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_34b \
  --output_dir ~/WorkspaceLabModels/codellama_34b_v8_ab \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_humaneval \
  --do_quixbugs \
  --do_defects4j --strict_defects4j \
  --do_generate

CUDA_VISIBLE_DEVICES="1" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_34b \
  --output_dir ~/WorkspaceLabModels/codellama_34b_v8_ab \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_humaneval \
  --do_validate

CUDA_VISIBLE_DEVICES="1" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_34b \
  --output_dir ~/WorkspaceLabModels/codellama_34b_v8_ab \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_quixbugs \
  --do_validate

CUDA_VISIBLE_DEVICES="1" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_34b \
  --output_dir ~/WorkspaceLabModels/codellama_34b_v8_ab \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_defects4j --strict_defects4j --validate_result_split_defects4j \
  --do_validate

# starcoder2_7b_v8 humaneval 스탭별 테스트
# gen
CUDA_VISIBLE_DEVICES="3" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/starcoder2_7b \
  --output_dir ~/WorkspaceLabModels/starcoder2_7b_v8/checkpoint-13 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_generate

CUDA_VISIBLE_DEVICES="2" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/starcoder2_7b \
  --output_dir ~/WorkspaceLabModels/starcoder2_7b_v8/checkpoint-125 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_generate

CUDA_VISIBLE_DEVICES="2" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/starcoder2_7b \
  --output_dir ~/WorkspaceLabModels/starcoder2_7b_v8/checkpoint-1250 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_generate

CUDA_VISIBLE_DEVICES="3" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/starcoder2_7b \
  --output_dir ~/WorkspaceLabModels/starcoder2_7b_v8/checkpoint-6250 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_generate

# val
CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/starcoder2_7b \
  --output_dir ~/WorkspaceLabModels/starcoder2_7b_v8/checkpoint-13 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_validate

CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/starcoder2_7b \
  --output_dir ~/WorkspaceLabModels/starcoder2_7b_v8/checkpoint-125 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_validate

CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/starcoder2_7b \
  --output_dir ~/WorkspaceLabModels/starcoder2_7b_v8/checkpoint-1250 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_validate

CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/starcoder2_7b \
  --output_dir ~/WorkspaceLabModels/starcoder2_7b_v8/checkpoint-6250 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_validate

# qwen2.5_coder_7b_v8 humaneval 스탭별 테스트
# gen
CUDA_VISIBLE_DEVICES="3" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/qwen2.5_coder_7b \
  --output_dir ~/WorkspaceLabModels/qwen2.5_coder_7b_v8/checkpoint-25 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_generate

CUDA_VISIBLE_DEVICES="2" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/qwen2.5_coder_7b \
  --output_dir ~/WorkspaceLabModels/qwen2.5_coder_7b_v8/checkpoint-250 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_generate

CUDA_VISIBLE_DEVICES="3" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/qwen2.5_coder_7b \
  --output_dir ~/WorkspaceLabModels/qwen2.5_coder_7b_v8/checkpoint-2500 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_generate

CUDA_VISIBLE_DEVICES="2" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/qwen2.5_coder_7b \
  --output_dir ~/WorkspaceLabModels/qwen2.5_coder_7b_v8/checkpoint-12500 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_generate

# val
CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/qwen2.5_coder_7b \
  --output_dir ~/WorkspaceLabModels/qwen2.5_coder_7b_v8/checkpoint-25 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_validate

CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/qwen2.5_coder_7b \
  --output_dir ~/WorkspaceLabModels/qwen2.5_coder_7b_v8/checkpoint-250 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_validate

CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/qwen2.5_coder_7b \
  --output_dir ~/WorkspaceLabModels/qwen2.5_coder_7b_v8/checkpoint-2500 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_validate

CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/qwen2.5_coder_7b \
  --output_dir ~/WorkspaceLabModels/qwen2.5_coder_7b_v8/checkpoint-12500 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_validate

# codegen_16b_v8_2 humaneval 스탭별 테스트
# gen
CUDA_VISIBLE_DEVICES="3" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codegen_16b \
  --output_dir ~/WorkspaceLabModels/codegen_16b_v8_2/checkpoint-25 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_generate

CUDA_VISIBLE_DEVICES="2" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codegen_16b \
  --output_dir ~/WorkspaceLabModels/codegen_16b_v8_2/checkpoint-250 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_generate

CUDA_VISIBLE_DEVICES="3" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codegen_16b \
  --output_dir ~/WorkspaceLabModels/codegen_16b_v8_2/checkpoint-2500 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_generate

CUDA_VISIBLE_DEVICES="2" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codegen_16b \
  --output_dir ~/WorkspaceLabModels/codegen_16b_v8_2/checkpoint-12500 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_generate
#여기까지
# val
CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codegen_16b \
  --output_dir ~/WorkspaceLabModels/codegen_16b_v8_2/checkpoint-25 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_validate

CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codegen_16b \
  --output_dir ~/WorkspaceLabModels/codegen_16b_v8_2/checkpoint-250 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_validate

CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codegen_16b \
  --output_dir ~/WorkspaceLabModels/codegen_16b_v8_2/checkpoint-2500 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_validate

CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codegen_16b \
  --output_dir ~/WorkspaceLabModels/codegen_16b_v8_2/checkpoint-12500 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_validate

# codellama_13b_v8 humaneval 스탭별 테스트
# gen
CUDA_VISIBLE_DEVICES="3" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_13b \
  --output_dir ~/WorkspaceLabModels/codellama_13b_v8/checkpoint-13 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_generate

CUDA_VISIBLE_DEVICES="2" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_13b \
  --output_dir ~/WorkspaceLabModels/codellama_13b_v8/checkpoint-125 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_generate

CUDA_VISIBLE_DEVICES="3" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_13b \
  --output_dir ~/WorkspaceLabModels/codellama_13b_v8/checkpoint-1250 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_generate

CUDA_VISIBLE_DEVICES="2" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_13b \
  --output_dir ~/WorkspaceLabModels/codellama_13b_v8/checkpoint-6250 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_generate

# val
CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_13b \
  --output_dir ~/WorkspaceLabModels/codellama_13b_v8/checkpoint-13 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_validate

CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_13b \
  --output_dir ~/WorkspaceLabModels/codellama_13b_v8/checkpoint-125 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_validate

CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_13b \
  --output_dir ~/WorkspaceLabModels/codellama_13b_v8/checkpoint-1250 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_validate

CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_13b \
  --output_dir ~/WorkspaceLabModels/codellama_13b_v8/checkpoint-6250 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_validate

# deepseek_coder_v2_lite_base_v8 humaneval 스탭별 테스트
# gen
CUDA_VISIBLE_DEVICES="3" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/deepseek_coder_v2_lite_base \
  --output_dir ~/WorkspaceLabModels/deepseek_coder_v2_lite_base_v8/checkpoint-13 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_humaneval \
  --do_generate

CUDA_VISIBLE_DEVICES="2" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/deepseek_coder_v2_lite_base \
  --output_dir ~/WorkspaceLabModels/deepseek_coder_v2_lite_base_v8/checkpoint-125 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_humaneval \
  --do_generate

CUDA_VISIBLE_DEVICES="3" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/deepseek_coder_v2_lite_base \
  --output_dir ~/WorkspaceLabModels/deepseek_coder_v2_lite_base_v8/checkpoint-1250 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_humaneval \
  --do_generate

CUDA_VISIBLE_DEVICES="2" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/deepseek_coder_v2_lite_base \
  --output_dir ~/WorkspaceLabModels/deepseek_coder_v2_lite_base_v8/checkpoint-6250 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_humaneval \
  --do_generate
#여기까지
# val
CUDA_VISIBLE_DEVICES="1" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/deepseek_coder_v2_lite_base \
  --output_dir ~/WorkspaceLabModels/deepseek_coder_v2_lite_base_v8/checkpoint-13 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_humaneval \
  --do_validate &&\
CUDA_VISIBLE_DEVICES="1" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/deepseek_coder_v2_lite_base \
  --output_dir ~/WorkspaceLabModels/deepseek_coder_v2_lite_base_v8/checkpoint-125 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_humaneval \
  --do_validate &&\
CUDA_VISIBLE_DEVICES="1" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/deepseek_coder_v2_lite_base \
  --output_dir ~/WorkspaceLabModels/deepseek_coder_v2_lite_base_v8/checkpoint-1250 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_humaneval \
  --do_validate &&\
CUDA_VISIBLE_DEVICES="1" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/deepseek_coder_v2_lite_base \
  --output_dir ~/WorkspaceLabModels/deepseek_coder_v2_lite_base_v8/checkpoint-6250 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_humaneval \
  --do_validate

# codellama_34b_v8 humaneval 스탭별 테스트
# gen
CUDA_VISIBLE_DEVICES="1" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_34b \
  --output_dir ~/WorkspaceLabModels/codellama_34b_v8/checkpoint-25 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_generate

CUDA_VISIBLE_DEVICES="1" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_34b \
  --output_dir ~/WorkspaceLabModels/codellama_34b_v8/checkpoint-250 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_generate

CUDA_VISIBLE_DEVICES="1" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_34b \
  --output_dir ~/WorkspaceLabModels/codellama_34b_v8/checkpoint-2500 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_generate

CUDA_VISIBLE_DEVICES="1" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_34b \
  --output_dir ~/WorkspaceLabModels/codellama_34b_v8/checkpoint-12500 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_generate

# val
CUDA_VISIBLE_DEVICES="1" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_34b \
  --output_dir ~/WorkspaceLabModels/codellama_34b_v8/checkpoint-25 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_validate

CUDA_VISIBLE_DEVICES="1" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_34b \
  --output_dir ~/WorkspaceLabModels/codellama_34b_v8/checkpoint-250 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_validate

CUDA_VISIBLE_DEVICES="1" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_34b \
  --output_dir ~/WorkspaceLabModels/codellama_34b_v8/checkpoint-2500 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_validate

CUDA_VISIBLE_DEVICES="1" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_34b \
  --output_dir ~/WorkspaceLabModels/codellama_34b_v8/checkpoint-12500 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_validate

# codellama_7b_v10
CUDA_VISIBLE_DEVICES="2" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_7b \
  --output_dir ~/WorkspaceLabModels/codellama_7b_v10 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_quixbugs \
  --do_defects4j --strict_defects4j \
  --do_generate

CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_7b \
  --output_dir ~/WorkspaceLabModels/codellama_7b_v10 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_validate &&\
CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_7b \
  --output_dir ~/WorkspaceLabModels/codellama_7b_v10 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_quixbugs \
  --do_validate

CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_7b \
  --output_dir ~/WorkspaceLabModels/codellama_7b_v10 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_defects4j --strict_defects4j --validate_result_split_defects4j \
  --do_validate

# starcoder2_7b_v10
CUDA_VISIBLE_DEVICES="2" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/starcoder2_7b \
  --output_dir ~/WorkspaceLabModels/starcoder2_7b_v10 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_humaneval \
  --do_quixbugs \
  --do_defects4j --strict_defects4j \
  --do_generate

CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/starcoder2_7b \
  --output_dir ~/WorkspaceLabModels/starcoder2_7b_v10 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_humaneval \
  --do_validate &&\
CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/starcoder2_7b \
  --output_dir ~/WorkspaceLabModels/starcoder2_7b_v10 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_quixbugs \
  --do_validate

CUDA_VISIBLE_DEVICES="2" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/starcoder2_7b \
  --output_dir ~/WorkspaceLabModels/starcoder2_7b_v10 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_defects4j --strict_defects4j --validate_result_split_defects4j \
  --do_validate

# qwen2.5_coder_7b_v10
CUDA_VISIBLE_DEVICES="1" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/qwen2.5_coder_7b \
  --output_dir ~/WorkspaceLabModels/qwen2.5_coder_7b_v10 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_humaneval \
  --do_quixbugs \
  --do_defects4j --strict_defects4j \
  --do_generate

CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/qwen2.5_coder_7b \
  --output_dir ~/WorkspaceLabModels/qwen2.5_coder_7b_v10 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_humaneval \
  --do_validate &&\
CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/qwen2.5_coder_7b \
  --output_dir ~/WorkspaceLabModels/qwen2.5_coder_7b_v10 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_quixbugs \
  --do_validate

CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/qwen2.5_coder_7b \
  --output_dir ~/WorkspaceLabModels/qwen2.5_coder_7b_v10 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_defects4j --strict_defects4j --validate_result_split_defects4j \
  --do_validate

# codegen_16b_v10
CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codegen_16b \
  --output_dir ~/WorkspaceLabModels/codegen_16b_v10 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_humaneval \
  --do_quixbugs \
  --do_defects4j --strict_defects4j \
  --do_generate

CUDA_VISIBLE_DEVICES="1" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codegen_16b \
  --output_dir ~/WorkspaceLabModels/codegen_16b_v10 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_humaneval \
  --do_validate &&\
CUDA_VISIBLE_DEVICES="1" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codegen_16b \
  --output_dir ~/WorkspaceLabModels/codegen_16b_v10 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_quixbugs \
  --do_validate

CUDA_VISIBLE_DEVICES="1" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codegen_16b \
  --output_dir ~/WorkspaceLabModels/codegen_16b_v10 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_defects4j --strict_defects4j --validate_result_split_defects4j \
  --do_validate

# deepseek_coder_v2_lite_base_v10
CUDA_VISIBLE_DEVICES="1" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/deepseek_coder_v2_lite_base \
  --output_dir ~/WorkspaceLabModels/deepseek_coder_v2_lite_base_v10 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_humaneval \
  --do_quixbugs \
  --do_defects4j --strict_defects4j \
  --do_generate

CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/deepseek_coder_v2_lite_base \
  --output_dir ~/WorkspaceLabModels/deepseek_coder_v2_lite_base_v10 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_humaneval \
  --do_validate &&\
CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/deepseek_coder_v2_lite_base \
  --output_dir ~/WorkspaceLabModels/deepseek_coder_v2_lite_base_v10 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_quixbugs \
  --do_validate

CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/deepseek_coder_v2_lite_base \
  --output_dir ~/WorkspaceLabModels/deepseek_coder_v2_lite_base_v10 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_defects4j --strict_defects4j --validate_result_split_defects4j \
  --do_validate
```
