## generation only
```shell
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
```



## validation only
```shell
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


CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/deepseek_coder_v2_lite_base \
  --output_dir ~/WorkspaceLabModels/deepseek_coder_v2_lite_base_v11 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_humaneval \
  --do_generate \
  --do_validate

CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/deepseek_coder_v2_lite_base \
  --output_dir ~/WorkspaceLabModels/deepseek_coder_v2_lite_base_v11 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_quixbugs \
  --do_generate \
  --do_validate

CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/deepseek_coder_v2_lite_base \
  --output_dir ~/WorkspaceLabModels/deepseek_coder_v2_lite_base_v11 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_defects4j --strict_defects4j --validate_result_split_defects4j \
  --do_generate \
  --do_validate


CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/deepseek_coder_v2_lite_base \
  --output_dir ~/WorkspaceLabModels/deepseek_coder_v2_lite_base_v12 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_humaneval \
  --do_generate \
  --do_validate

CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/deepseek_coder_v2_lite_base \
  --output_dir ~/WorkspaceLabModels/deepseek_coder_v2_lite_base_v12 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_quixbugs \
  --do_generate \
  --do_validate

CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/deepseek_coder_v2_lite_base \
  --output_dir ~/WorkspaceLabModels/deepseek_coder_v2_lite_base_v12 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_defects4j --strict_defects4j --validate_result_split_defects4j \
  --do_generate \
  --do_validate
```



## deepseek_coder_v2_lite_base_v8 humaneval 스탭별 테스트
```shell
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
# 여기까지
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
```



## deepseek_coder_v2_lite_base_v10_paramtest
```shell
CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/deepseek_coder_v2_lite_base \
  --output_dir ~/WorkspaceLabModels/deepseek_coder_v2_lite_base_v10_paramtest \
  --bits 16 \
  --fp16 \
  --bf16 False\
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_humaneval \
  --do_quixbugs \
  --do_defects4j --strict_defects4j \
  --do_generate
```
