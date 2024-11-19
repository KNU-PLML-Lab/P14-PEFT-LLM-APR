## Fine-tuning

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
> 여기까지 진행중

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

#### codellama_13b_v10 (A6000?)
```bash
CUDA_VISIBLE_DEVICES="0" python ./src/sg_finetune.py \
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

### qwen2.5_coder_7b_v9 (4090)
```bash
CUDA_VISIBLE_DEVICES="2" python ./src/sg_finetune.py \
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
CUDA_VISIBLE_DEVICES="2" python ./src/sg_finetune.py \
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

## Benchmarks
```bash
# --do_humaneval
# --do_quixbugs
# --do_defects4j --strict_defects4j --validate_result_split_defects4j

# --do_generate
# --do_validate

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
# 여기까지
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
```

### DEBUG QuixBugs
```bash
javac -cp .:java_programs:junit4-4.12.jar:hamcrest-all-1.3.jar java_testcases/junit/GCD_TEST.java
java -cp .:java_programs:junit4-4.12.jar:hamcrest-all-1.3.jar org.junit.runner.JUnitCore java_testcases.junit.GCD_TEST
```

### DEBUG defects4j
```bash
defects4j checkout -p Chart -v 4b -w /home/yglee/wl/p14/nosync/defects4j_tmp853/tmp
```
