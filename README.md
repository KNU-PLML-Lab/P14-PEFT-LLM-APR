# PL&ML YKL Project 14
## Dependencies
- conda
(inherit from Defects4J 2.0.1)
- Java 1.8
- Git >= 1.9
- SVN >= 1.8
- Perl >= 5.0.12

## How to start
- My env: Win11 > wsl2 > Ubuntu 24.04 LTS
```bash
git clone [THIS_REPOSITORY]
# init submodules (Defects4J 2.0.1)
git submodule update --init --recursive

# Download Java 1.8
# Manual download: https://www.oracle.com/java/technologies/downloads/?er=221886#java8
# or
sudo add-apt-repository ppa:openjdk-r/ppa
sudo apt update
sudo apt install openjdk-8-jdk

# Init Defects4J
cd defects4j
sudo apt install cpanminus unzip build-essential
cpanm --installdeps .
./init.sh

# Register Defects4J PATH
export PATH=$PATH:$(pwd)/framework/bin
# ... or add to .bashrc
#echo 'export PATH="$PATH:'$(pwd)'/framework/bin"' >> ~/.bashrc
#source ~/.bashrc

# Install Perl modules
sudo cpanm DBI
sudo cpanm DBD::SQLite

# Check Defects4J
defects4j info -p Lang
cd ..

# Init Conda
conda env update --prefix ./.conda --file environment.yml
conda activate ./.conda

# Run the project
python main.py --task train
```

## Additional tools & Notes
- https://bitbucket.org/rjust/fault-localization-data/src/master/

```bash
# Sample Defects4J (Lang project, 1st bug)
defects4j checkout -p Lang -v 1b -w Lang_1_buggy
defects4j checkout -p Lang -v 1f -w Lang_1_fixed
```

extra...
```bash
# for matplotlib
sudo apt install libxcb-cursor0
```

## Reference
- https://llama.meta.com/docs/how-to-guides/fine-tuning/
- https://github.com/microsoft/CodeXGLUE


## Memo
```
conda env update --name p14 --file environment.yml
pip install git+https://github.com/huggingface/transformers
pip install git+https://github.com/bitsandbytes-foundation/bitsandbytes
pip install transformers -U
pip install 'accelerate>=0.26.0'
pip install 'bitsandbytes>=0.41.3'
pip install flash-attn --no-build-isolation --no-cache-dir
pip install fuzzywuzzy
pip install bitsandbytes --no-build-isolation --no-cache-dir
pip install peft --no-build-isolation --no-cache-dir
pip install sentencepiece
pip install protobuf

wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/3bf863cc.pub
sudo add-apt-repository 'deb https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/ /'
sudo apt-get update
sudo apt-get -y install cuda
sudo apt install nvidia-cuda-toolkit
nvcc --version

ln -s /usr/lib/wsl/lib/libcuda.so /home/stgr/miniconda3/envs/p14_3/lib/libcuda.so
```

## clm
목차
- **codegen_peft**
  - 코드: lin-tan/clm
  - 데이터셋: lin-tan/clm
  - codegen-6b-multi
  - qlora finetune
- **codegen_peft2**
  - 코드: artidoro/qlora
  - 데이터셋: lin-tan/clm
  - codegen-6b-multi
  - qlora finetune
- **codegen_peft3**
  - 코드: artidoro/qlora
  - 데이터셋: lin-tan/clm
  - codegen-6b-multi
  - qlora finetune
  - 수정된 prompt / lr조정
- **codellama_peft**
  - 코드: artidoro/qlora
  - 데이터셋: lin-tan/clm
  - code_llama-13b-hf
  - qlora finetune
- **codellama_peft2**
  - 코드: artidoro/qlora
  - 데이터셋: lin-tan/clm
  - code_llama-13b-hf
  - qlora finetune
  - lr 2e-4
- **codellama_peft3**
  - 코드: artidoro/qlora
  - 데이터셋: lin-tan/clm
  - code_llama-7b-instruct-hf
  - qlora finetune
- **codellama_7b_peft**
  - 코드: artidoro/qlora
  - 데이터셋: lin-tan/clm
  - code_llama-7b-hf
  - qlora finetune
  - lr 2e-4
  - token 2048 / 512
- **codellama_34b_peft**
  - 코드: artidoro/qlora
  - 데이터셋: lin-tan/clm
  - code_llama-34b-hf
  - qlora finetune
  - lr 2e-4
  - token 2048 / 512
- **deepseek-coder-v2-lite-base_peft**
  - 코드: artidoro/qlora
  - 데이터셋: lin-tan/clm
  - DeepSeek-Coder-V2-Lite-Base (16B / 2.4B active params)
  - qlora finetune
  - lr 2e-4

20241015 이후
- codegen_6b_v2
  - 코드: lin-tan/clm 변형
  - 데이터셋: lin-tan/clm (768길이초과 제외)
  - codegen-6B
  - qlora finetune (r: 64 / a: 16 / lora_dropoutput 0.1)
  - lr 2e-4
  - batch 16
  - LoraLayer fp32
  - 시간이 60시간이나 걸림
<!-- - **codegen_6b_v2_np**
  - 위와 동일. 하지만 bitsandbytes.adam.PagedAdam32bit param중 is_paged=True가 없음
  - 시간 차이 없어서 실험 취소 -->
- codegen_6b_v3
  - 코드: artidoro/qlora 변형
  - 데이터셋: lin-tan/clm (768길이초과 제외)
  - codegen-6B
  - qlora finetune (r: 64 / a: 16 / lora_dropoutput 0.1)
  - lr 2e-4
  - batch 1
- codegen_13b_v3
  - 코드: artidoro/qlora 변형
  - 데이터셋: lin-tan/clm (768길이초과 제외)
  - codegen-13B
  - qlora finetune (r: 64 / a: 16 / lora_dropoutput 0.1)
  - lr 2e-4
  - batch 16
- codellama_7b_v3
  - 코드: artidoro/qlora 변형
  - 데이터셋: lin-tan/clm (1024길이초과 제외)
  - code_llama_7b
  - qlora finetune (r: 64 / a: 16 / lora_dropoutput 0.1)
  - lr 2e-4
  - batch 1
- codellama_13b_v3
  - 코드: artidoro/qlora 변형
  - 데이터셋: lin-tan/clm (1024길이초과 제외)
  - code_llama_13b
  - qlora finetune (r: 64 / a: 16 / lora_dropoutput 0.1)
  - lr 2e-4
  - batch 16
- codellama_34b_v3
  - 코드: artidoro/qlora 변형
  - 데이터셋: lin-tan/clm (1024길이초과 제외)
  - code_llama_34b
  - qlora finetune (r: 64 / a: 16 / lora_dropoutput 0.05)
  - lr 1e-4
  - batch 6
<!-- - **dscoder_16b_v3**
  - 코드: artidoro/qlora 변형
  - 데이터셋: lin-tan/clm (1024길이초과 제외)
  - deepseek-coder-v2-lite-base
  - qlora finetune (r: 64 / a: 16 / lora_dropoutput 0.1)
  - lr 2e-4
  - batch 16 -->
- codegen_6b_v4
  - 코드: artidoro/qlora 변형
  - 데이터셋: lin-tan/clm (768길이초과 제외)
  - codegen-6B
  - qlora finetune (r: 64 / a: 16 / lora_dropoutput 0.1)
  - lr 2e-4
  - batch 16
  - gradient_checkpointing
- codellama_7b_v2
  - 코드: lin-tan/clm 변형
  - 데이터셋: lin-tan/clm (1024길이초과 제외)
  - code_llama_7b
  - qlora finetune (r: 64 / a: 16 / lora_dropoutput 0.1)
  - lr 2e-4
  - batch 16


## 20241015 이후

### codellama_34b_v5 (a6000)
```bash
CUDA_VISIBLE_DEVICES=1 python sg_finetune.py \
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

### codellama_13b_v5 (a6000)
```bash
CUDA_VISIBLE_DEVICES=0 python sg_finetune.py \
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

### incoder_6b_v5 (4090)
```bash
CUDA_VISIBLE_DEVICES="3" python sg_finetune.py \
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

### codellama_7b_v5 (4090)
```bash
CUDA_VISIBLE_DEVICES=1 python sg_finetune.py \
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

### codegen_6b_v5 (4090)
```bash
CUDA_VISIBLE_DEVICES=2 python sg_finetune.py \
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

### dscoder_16b_v2 (4090) 구현중
```bash
CUDA_VISIBLE_DEVICES="2,3" python sg_finetune.py \
    --run_name dscoder_16b_v2 \
    --model_name_or_path ~/WorkspaceLabModels/deepseek-coder-v2-lite-base \
    --output_dir ../../nosync/output/dscoder_16b_v2 \
    --dataset ../../finetune_training.jsonl \
    --validation_dataset ../../finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0002 \
    --seed 0 \
    --max_length 1024 \
    --per_device_train_batch_size 1 \
    --num_train_epochs 1 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --use_reentrant \
    --save-steps 16000 \
    --eval_steps 16000 \
    --max_eval_samples 1000 \
    --eval_dataset_size 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --trust_remote_code \
    --report_to wandb
```

### incoder_6b_v2 (4090)
```bash
CUDA_VISIBLE_DEVICES="2" python sg_finetune.py \
    --run_name incoder_6b_v2 \
    --model_name_or_path ~/WorkspaceLabModels/incoder-6B \
    --output_dir ../../nosync/output/incoder_6b_v2 \
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

### codellama_13b_v2 (a6000)
```bash
CUDA_VISIBLE_DEVICES=1 python sg_finetune.py \
    --run_name codellama_13b_v2 \
    --model_name_or_path ~/WorkspaceLabModels/code_llama-13b-hf \
    --output_dir ../../nosync/output/codellama_13b_v2 \
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

### codellama_7b_v2 (4090)
```bash
CUDA_VISIBLE_DEVICES=1 python sg_finetune.py \
    --run_name codellama_7b_v2 \
    --model_name_or_path ~/WorkspaceLabModels/code_llama-7b-hf \
    --output_dir ../../nosync/output/codellama_7b_v2 \
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

### dscoder_16b_v3 (4090)
```bash
CUDA_VISIBLE_DEVICES="0,3" python sg_finetune_trainer.py \
    --run_name dscoder_16b_v3 \
    --model_name_or_path ~/WorkspaceLabModels/deepseek-coder-v2-lite-base \
    --output_dir ../../nosync/output/dscoder_16b_v3 \
    --dataset ../../finetune_training.jsonl \
    --validation_dataset ../../finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0002 \
    --seed 0 \
    --max_length 1024 \
    --per_device_train_batch_size 1 \
    --num_train_epochs 1 \
    --gradient_checkpointing \
    --gradient_accumulation_steps \
    --use_reentrant \
    --save-steps 20000 \
    --eval_steps 20000 \
    --max_eval_samples 1000 \
    --eval_dataset_size 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --trust_remote_code \
    --report_to wandb
```

### codellama_34b_v3 (a6000)
```bash
CUDA_VISIBLE_DEVICES=1 python sg_finetune_trainer.py \
    --run_name codellama_34b_v3 \
    --model_name_or_path ~/WorkspaceLabModels/code_llama-34b-hf \
    --output_dir ../../nosync/output/codellama_34b_v3 \
    --dataset ../../finetune_training.jsonl \
    --validation_dataset ../../finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0001 \
    --seed 0 \
    --max_length 1024 \
    --per_device_train_batch_size 6 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --eval_steps 1000 \
    --max_eval_samples 1000 \
    --eval_dataset_size 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --lora_dropout 0.05 \
    --report_to wandb
    
```

### codellama_13b_v3 (a6000)
```bash
CUDA_VISIBLE_DEVICES=0 python sg_finetune_trainer.py \
    --run_name codellama_13b_v3 \
    --model_name_or_path ~/WorkspaceLabModels/code_llama-13b-hf \
    --output_dir ../../nosync/output/codellama_13b_v3 \
    --dataset ../../finetune_training.jsonl \
    --validation_dataset ../../finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0002 \
    --seed 0 \
    --max_length 1024 \
    --per_device_train_batch_size 1 \
    --num_train_epochs 1 \
    --save_steps 10000 \
    --eval_steps 10000 \
    --max_eval_samples 1000 \
    --eval_dataset_size 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --report_to wandb
```

### codellama_7b_v3 (4090)
```bash
CUDA_VISIBLE_DEVICES=2 python sg_finetune_trainer.py \
    --run_name codellama_7b_v3 \
    --model_name_or_path ~/WorkspaceLabModels/code_llama-7b-hf \
    --output_dir ../../nosync/output/codellama_7b_v3 \
    --dataset ../../finetune_training.jsonl \
    --validation_dataset ../../finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0002 \
    --seed 0 \
    --max_length 1024 \
    --per_device_train_batch_size 1 \
    --num_train_epochs 1 \
    --save_steps 10000 \
    --eval_steps 10000 \
    --max_eval_samples 1000 \
    --eval_dataset_size 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --report_to wandb
```

### codegen_6b_v4 (4090)
```bash
CUDA_VISIBLE_DEVICES=3 python sg_finetune_trainer.py \
    --run_name codegen_6b_v4 \
    --model_name_or_path ~/WorkspaceLabModels/codegen-6B \
    --output_dir ../../nosync/output/codegen_6b_v4 \
    --dataset ../../finetune_training.jsonl \
    --validation_dataset ../../finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0002 \
    --seed 0 \
    --max_length 768 \
    --per_device_train_batch_size 12 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --save_steps 2000 \
    --eval_steps 2000 \
    --max_eval_samples 1000 \
    --eval_dataset_size 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --report_to wandb
```

### codegen_6b_v3 (4090)
```bash
CUDA_VISIBLE_DEVICES=1 python sg_finetune_trainer.py \
    --run_name codegen_6b_v3 \
    --model_name_or_path ~/WorkspaceLabModels/codegen-6B \
    --output_dir ../../nosync/output/codegen_6b_v3 \
    --dataset ../../finetune_training.jsonl \
    --validation_dataset ../../finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0002 \
    --seed 0 \
    --max_length 768 \
    --per_device_train_batch_size 1 \
    --num_train_epochs 1 \
    --save_steps 10000 \
    --eval_steps 10000 \
    --max_eval_samples 1000 \
    --eval_dataset_size 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --report_to wandb
```

### codegen_6b_v2
```bash
CUDA_VISIBLE_DEVICES=1 python sg_finetune.py \
--model_name_or_path ~/hdd1/WorkspaceLabModels/codegen-6B \
    --output_dir ../../nosync/output/codegen_6b_v2 \
    --dataset ../../finetune_training.jsonl \
    --validation_dataset ../../finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0002 \
    --seed 0 \
    --max_length 768 \
    --per_device_train_batch_size 16 \
    --num_train_epochs 1 \
    --eval_steps 1000 \
    --eval_dataset_size 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --report_to wandb
    
```

### codegen_6b_v2_np
```bash
CUDA_VISIBLE_DEVICES=1 python sg_finetune.py \
--model_name_or_path ~/hdd1/WorkspaceLabModels/codegen-6B \
    --output_dir ../../nosync/output/codegen_6b_v2_np \
    --dataset ../../finetune_training.jsonl \
    --validation_dataset ../../finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.0002 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --seed 0 \
    --max_length 768 \
    --per_device_train_batch_size 16 \
    --num_train_epochs 1 \
    --eval_steps 1000 \
    --eval_dataset_size 1000 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --report_to wandb \
    --max_memory_MB 44000
```


### codegen_peft3_3 wandb 를 통해 속도 비교용
```bash
cd clm-apr/codegen_peft3

# --max_steps 1875 \ --num_train_epochs 1 \
# --eval_steps 187 \ 
CUDA_VISIBLE_DEVICES=1 python qlora.py \
--model_name_or_path ~/hdd1/WorkspaceLabModels/codegen-6B \
    --output_dir ../../nosync/output/codegen-6B_peft3_4 \
    --dataset ../../finetune_training.jsonl \
    --data_seed 42 \
    --dataloader_num_workers 1 \
    --group_by_length \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_modules all \
    --double_quant \
    --quant_type nf4 \
    --bf16 \
    --bits 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing \
    --learning_rate 0.0002 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.1 \
    --weight_decay 0.0 \
    --seed 0 \
    --source_max_len 768 \
    --target_max_len 768 \
    --max_new_tokens 256 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --eval_dataset_size 1000 \
    --max_eval_samples 1000 \
    --per_device_eval_batch_size 1 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 40 \
    --report_to wandb \
    --max_memory_MB 44000
```

### lin-tan/clm 기반 lin-tan/clm 데이터셋 codegen-6b-multi qlora finetune
```bash
cd clm-apr/codegen_peft
CUDA_VISIBLE_DEVICES=0 python finetune.py
```
```
mat1 and mat2 must have the same dtype, but got Float and BFloat16
Traceback (most recent call last):
  File "/mnt/hdd0/yglee/WorkspaceLab/p14/external/clm/clm-apr/codegen_peft/finetune.py", line 217, in <module>
    fine_tune(
  File "/mnt/hdd0/yglee/WorkspaceLab/p14/external/clm/clm-apr/codegen_peft/finetune.py", line 199, in fine_tune
    round(sum(training_loss) / len(training_loss), 4),
ZeroDivisionError: division by zero
```
### artidoro/qlora 기반 lin-tan/clm 데이터셋 codegen-6b-multi qlora finetune
```bash
cd clm-apr/codegen_peft2

# --max_steps 1875 \ --num_train_epochs 1 \
# --eval_steps 187 \ 
CUDA_VISIBLE_DEVICES=0 python qlora.py \
--model_name_or_path ~/hdd1/models/codegen-6B \
    --output_dir ../../nosync/output/codegen-6B_peft2 \
    --dataset ../../finetune_training.jsonl \
    --data_seed 42 \
    --dataloader_num_workers 1 \
    --group_by_length \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_modules all \
    --double_quant \
    --quant_type nf4 \
    --bf16 \
    --bits 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing \
    --learning_rate 0.0002 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.1 \
    --weight_decay 0.0 \
    --seed 0 \
    --source_max_len 512 \
    --target_max_len 256 \
    --max_new_tokens 256 \
    --per_device_train_batch_size 24 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --eval_dataset_size 1000 \
    --max_eval_samples 1000 \
    --per_device_eval_batch_size 1 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 40 \

```

### artidoro/qlora 기반 lin-tan/clm 데이터셋 codegen-6b-multi qlora finetune - 수정된 prompt / lr조정
```bash
cd clm-apr/codegen_peft3

# --max_steps 1875 \ --num_train_epochs 1 \
# --eval_steps 187 \ 
CUDA_VISIBLE_DEVICES=1 python qlora.py \
--model_name_or_path ~/hdd1/WorkspaceLabModels/codegen-6B \
    --output_dir ../../nosync/output/codegen-6B_peft3_2 \
    --dataset ../../finetune_training.jsonl \
    --data_seed 42 \
    --dataloader_num_workers 1 \
    --group_by_length \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_modules all \
    --double_quant \
    --quant_type nf4 \
    --bf16 \
    --bits 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing \
    --learning_rate 0.0002 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.1 \
    --weight_decay 0.0 \
    --seed 0 \
    --source_max_len 512 \
    --target_max_len 256 \
    --max_new_tokens 256 \
    --per_device_train_batch_size 24 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --eval_dataset_size 1000 \
    --max_eval_samples 1000 \
    --per_device_eval_batch_size 1 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 40 \

```

### artidoro/qlora 기반 lin-tan/clm 데이터셋 codegen-6b-multi qlora finetune - 수정된 prompt
```bash
cd clm-apr/codegen_peft3

# --max_steps 1875 \ --num_train_epochs 1 \
# --eval_steps 187 \ 
CUDA_VISIBLE_DEVICES=1 python qlora.py \
--model_name_or_path ~/hdd1/nosync/models/codegen-6B \
    --output_dir ../../nosync/output/codegen-6B_peft3 \
    --dataset ../../finetune_training.jsonl \
    --data_seed 42 \
    --dataloader_num_workers 1 \
    --group_by_length \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_modules all \
    --double_quant \
    --quant_type nf4 \
    --bf16 \
    --bits 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing \
    --learning_rate 0.00001 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.1 \
    --weight_decay 0.0 \
    --seed 0 \
    --source_max_len 512 \
    --target_max_len 256 \
    --max_new_tokens 256 \
    --per_device_train_batch_size 24 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --eval_dataset_size 1000 \
    --max_eval_samples 1000 \
    --per_device_eval_batch_size 1 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 40 \


CUDA_VISIBLE_DEVICES=1 python sg_inference_adapter.py \
--model_name_or_path ~/hdd1/nosync/models/codegen-6B \
    --output_dir ../../nosync/output/codegen-6B_peft3 \
    --dataset ../../finetune_training.jsonl \
    --data_seed 42 \
    --dataloader_num_workers 1 \
    --group_by_length \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_modules all \
    --double_quant \
    --quant_type nf4 \
    --bf16 \
    --bits 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing \
    --learning_rate 0.00001 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.1 \
    --weight_decay 0.0 \
    --seed 0 \
    --source_max_len 512 \
    --target_max_len 256 \
    --max_new_tokens 256 \
    --per_device_train_batch_size 24 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --eval_dataset_size 1000 \
    --max_eval_samples 1000 \
    --per_device_eval_batch_size 1 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 40 \

```

### artidoro/qlora 기반 lin-tan/clm 데이터셋 codellama-13b-hf qlora finetune
```bash
cd clm-apr/codellama_peft

# --max_steps 1875 \ --num_train_epochs 1 \
# --eval_steps 187 \ 
CUDA_VISIBLE_DEVICES=1 python qlora.py \
--model_name_or_path ~/hdd1/nosync/models/code_llama-13b-hf \
    --output_dir ../../nosync/output/code_llama-13b-hf_peft \
    --dataset ../../finetune_training.jsonl \
    --data_seed 42 \
    --dataloader_num_workers 1 \
    --group_by_length \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_modules all \
    --double_quant \
    --quant_type nf4 \
    --bf16 \
    --bits 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing \
    --learning_rate 0.00001 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.1 \
    --weight_decay 0.0 \
    --seed 0 \
    --source_max_len 512 \
    --target_max_len 256 \
    --max_new_tokens 256 \
    --per_device_train_batch_size 24 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --eval_dataset_size 1000 \
    --max_eval_samples 1000 \
    --per_device_eval_batch_size 1 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 40 \

```

### artidoro/qlora 기반 lin-tan/clm 데이터셋 codellama-13b-hf / qlora finetune / lr 2e-4
```bash
cd clm-apr/codellama_peft

# --max_steps 1875 \ --num_train_epochs 1 \
# --eval_steps 187 \ 
CUDA_VISIBLE_DEVICES=1 python qlora.py \
--model_name_or_path ~/hdd1/nosync/models/code_llama-13b-hf \
    --output_dir ../../nosync/output/code_llama-13b-hf_peft2 \
    --dataset ../../finetune_training.jsonl \
    --data_seed 42 \
    --dataloader_num_workers 1 \
    --group_by_length \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_modules all \
    --double_quant \
    --quant_type nf4 \
    --bf16 \
    --bits 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing \
    --learning_rate 0.0002 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.1 \
    --weight_decay 0.0 \
    --seed 0 \
    --source_max_len 512 \
    --target_max_len 256 \
    --max_new_tokens 256 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --eval_dataset_size 1000 \
    --max_eval_samples 1000 \
    --per_device_eval_batch_size 1 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 40 \

```

### artidoro/qlora 기반 lin-tan/clm 데이터셋 codellama-7b-instruct-hf qlora finetune
```bash
cd clm-apr/codellama_peft

# --max_steps 1875 \ --num_train_epochs 1 \
# --eval_steps 187 \ 
CUDA_VISIBLE_DEVICES=1 python qlora.py \
--model_name_or_path ~/hdd1/nosync/models/code_llama-7b-instruct-hf \
    --output_dir ../../nosync/output/code_llama-7b-instruct-hf_peft \
    --dataset ../../finetune_training.jsonl \
    --data_seed 42 \
    --dataloader_num_workers 1 \
    --group_by_length \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_modules all \
    --double_quant \
    --quant_type nf4 \
    --bf16 \
    --bits 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing \
    --learning_rate 0.0002 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.1 \
    --weight_decay 0.0 \
    --seed 0 \
    --source_max_len 512 \
    --target_max_len 256 \
    --max_new_tokens 256 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --eval_dataset_size 1000 \
    --max_eval_samples 1000 \
    --per_device_eval_batch_size 1 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 40 \

```
### codellama_7b_peft
```bash
cd clm-apr/codellama_7b_peft

CUDA_VISIBLE_DEVICES=1 python qlora.py \
--model_name_or_path ~/hdd1/nosync/models/code_llama-7b-hf \
    --output_dir ../../nosync/output/code_llama-7b-hf_peft \
    --dataset ../../finetune_training.jsonl \
    --data_seed 42 \
    --dataloader_num_workers 1 \
    --group_by_length \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_modules all \
    --double_quant \
    --quant_type nf4 \
    --bf16 \
    --bits 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing \
    --learning_rate 0.0002 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.1 \
    --weight_decay 0.0 \
    --seed 0 \
    --source_max_len 2048 \
    --target_max_len 512 \
    --max_new_tokens 512 \
    --per_device_train_batch_size 12 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --eval_dataset_size 1000 \
    --max_eval_samples 1000 \
    --per_device_eval_batch_size 1 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 40 \

```

### codellama_34b_peft
```bash
cd clm-apr/codellama_34b_peft

CUDA_VISIBLE_DEVICES=0 python qlora.py \
--model_name_or_path ~/hdd1/nosync/models/code_llama-34b-hf \
    --output_dir ../../nosync/output/code_llama-34b-hf_peft \
    --dataset ../../finetune_training.jsonl \
    --data_seed 42 \
    --dataloader_num_workers 1 \
    --group_by_length \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_modules all \
    --double_quant \
    --quant_type nf4 \
    --bf16 \
    --bits 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing \
    --learning_rate 0.0002 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.1 \
    --weight_decay 0.0 \
    --seed 0 \
    --source_max_len 2048 \
    --target_max_len 512 \
    --max_new_tokens 512 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --eval_dataset_size 1000 \
    --max_eval_samples 1000 \
    --per_device_eval_batch_size 1 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 5000 \
    --save_total_limit 40 \

```

### deepseek-coder-v2-lite-base_peft
```bash
cd clm-apr/deepseek-coder-v2-lite-base_peft
#  File "/home/yglee/miniconda3/envs/p14_5/lib/python3.10/site-packages/accelerate/state.py", line 287, in __init__
#    raise NotImplementedError(
#NotImplementedError: Using RTX 4000 series doesn't support faster communication broadband via P2P or IB. Please set #`NCCL_P2P_DISABLE="1"` and `NCCL_IB_DISABLE="1" or use `accelerate launch` which will do this automatically.
NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" CUDA_VISIBLE_DEVICES="0,1,2,3" python qlora.py \
--model_name_or_path ~/nosync/models/deepseek-coder-v2-lite-base \
    --output_dir ../../nosync/output/deepseek-coder-v2-lite-base_peft \
    --dataset ../../finetune_training.jsonl \
    --data_seed 42 \
    --dataloader_num_workers 1 \
    --group_by_length \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_modules all \
    --double_quant \
    --quant_type nf4 \
    --bf16 \
    --bits 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing \
    --learning_rate 0.0002 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.1 \
    --weight_decay 0.0 \
    --seed 0 \
    --source_max_len 2048 \
    --target_max_len 512 \
    --max_new_tokens 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --eval_dataset_size 1000 \
    --max_eval_samples 1000 \
    --per_device_eval_batch_size 1 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 10000 \
    --save_total_limit 40 \
    --max_memory_MB 22000 \
    --trust_remote_code
```

### 오리지널 codegen
```bash
cd clm-apr/codegen_finetune

# --max_steps 1875 \ --num_train_epochs 1 \
# --eval_steps 187 \ 
CUDA_VISIBLE_DEVICES=0 python finetune.py

```

### apex 설치
```bash
git clone https://github.com/NVIDIA/apex
cd apex
python setup.py install
```
