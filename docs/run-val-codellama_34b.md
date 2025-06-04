## old
```shell
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
```



## generation only
```shell
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

CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_34b \
  --output_dir ~/WorkspaceLabModels/codellama_34b_v9 \
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
  --output_dir ~/WorkspaceLabModels/codellama_34b_v10_2 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_quixbugs \
  --do_defects4j --strict_defects4j \
  --do_generate
```



## validation only
```shell
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


CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_34b \
  --output_dir ~/WorkspaceLabModels/codellama_34b_v9 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_validate &&\
CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_34b \
  --output_dir ~/WorkspaceLabModels/codellama_34b_v9 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_quixbugs \
  --do_validate

CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_34b \
  --output_dir ~/WorkspaceLabModels/codellama_34b_v9 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_defects4j --strict_defects4j --validate_result_split_defects4j \
  --do_validate


# 여기까지
CUDA_VISIBLE_DEVICES="1" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_34b \
  --output_dir ~/WorkspaceLabModels/codellama_34b_v10_2 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_generate \
  --do_validate

CUDA_VISIBLE_DEVICES="1" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_34b \
  --output_dir ~/WorkspaceLabModels/codellama_34b_v10_2 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_quixbugs \
  --do_generate \
  --do_validate

CUDA_VISIBLE_DEVICES="1" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_34b \
  --output_dir ~/WorkspaceLabModels/codellama_34b_v10_2 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_defects4j --strict_defects4j --validate_result_split_defects4j \
  --do_generate \
  --do_validate


CUDA_VISIBLE_DEVICES="1" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_34b \
  --output_dir ~/WorkspaceLabModels/codellama_34b_v11 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_generate \
  --do_validate

CUDA_VISIBLE_DEVICES="1" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_34b \
  --output_dir ~/WorkspaceLabModels/codellama_34b_v11 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_quixbugs \
  --do_generate \
  --do_validate

CUDA_VISIBLE_DEVICES="1" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_34b \
  --output_dir ~/WorkspaceLabModels/codellama_34b_v11 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_defects4j --strict_defects4j --validate_result_split_defects4j \
  --do_generate \
  --do_validate


CUDA_VISIBLE_DEVICES="1" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_34b \
  --output_dir ~/WorkspaceLabModels/codellama_34b_v12 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_generate \
  --do_validate

CUDA_VISIBLE_DEVICES="1" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_34b \
  --output_dir ~/WorkspaceLabModels/codellama_34b_v12 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_quixbugs \
  --do_generate \
  --do_validate

CUDA_VISIBLE_DEVICES="1" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_34b \
  --output_dir ~/WorkspaceLabModels/codellama_34b_v12 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_defects4j --strict_defects4j --validate_result_split_defects4j \
  --do_generate \
  --do_validate
```



## codellama_34b_v8 humaneval 스탭별 테스트
```shell
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
```