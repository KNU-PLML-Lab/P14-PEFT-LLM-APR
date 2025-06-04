## old
```shell
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

CUDA_VISIBLE_DEVICES="1" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_7b \
  --output_dir ~/WorkspaceLabModels/codellama_7b_v5 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_defects4j --validate_result_split_defects4j

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
```



## generation only
```shell
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

CUDA_VISIBLE_DEVICES="2" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_7b \
  --output_dir ~/WorkspaceLabModels/codellama_7b_v11 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_quixbugs \
  --do_defects4j --strict_defects4j \
  --do_generate

CUDA_VISIBLE_DEVICES="3" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_7b \
  --output_dir ~/WorkspaceLabModels/codellama_7b_v12 \
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


CUDA_VISIBLE_DEVICES="1" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_7b \
  --output_dir ~/WorkspaceLabModels/codellama_7b_v11 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_validate

CUDA_VISIBLE_DEVICES="2" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_7b \
  --output_dir ~/WorkspaceLabModels/codellama_7b_v11 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_quixbugs \
  --do_generate \
  --do_validate

CUDA_VISIBLE_DEVICES="3" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_7b \
  --output_dir ~/WorkspaceLabModels/codellama_7b_v11 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_defects4j --strict_defects4j --validate_result_split_defects4j \
  --do_validate


CUDA_VISIBLE_DEVICES="1" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_7b \
  --output_dir ~/WorkspaceLabModels/codellama_7b_v12 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_validate

CUDA_VISIBLE_DEVICES="2" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_7b \
  --output_dir ~/WorkspaceLabModels/codellama_7b_v12 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_quixbugs \
  --do_generate \
  --do_validate

CUDA_VISIBLE_DEVICES="3" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_7b \
  --output_dir ~/WorkspaceLabModels/codellama_7b_v12 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_defects4j --strict_defects4j --validate_result_split_defects4j \
  --do_validate
```



## codellama_7b_v9 humaneval (v5이 중간값이지만, 시드가 없으며, checkpoint가 누락되어서 v9 대신 사용) 스탭별 테스트
```shell
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
```