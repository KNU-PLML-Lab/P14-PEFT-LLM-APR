## Benchmarks

### DEBUG QuixBugs
```bash
javac -cp .:java_programs:junit4-4.12.jar:hamcrest-all-1.3.jar java_testcases/junit/GCD_TEST.java
java -cp .:java_programs:junit4-4.12.jar:hamcrest-all-1.3.jar org.junit.runner.JUnitCore java_testcases.junit.GCD_TEST
```

### DEBUG defects4j
```bash
defects4j checkout -p Chart -v 4b -w /home/yglee/wl/p14/nosync/defects4j_tmp853/tmp
```

```bash
# --do_humaneval
# --do_quixbugs
# --do_defects4j --strict_defects4j --validate_result_split_defects4j

# --do_generate
# --do_validate

# codellama_13b_v9
CUDA_VISIBLE_DEVICES="1" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_13b \
  --output_dir ~/WorkspaceLabModels/codellama_13b_v9 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_quixbugs \
  --do_defects4j --strict_defects4j \
  --do_generate
#여기까지
CUDA_VISIBLE_DEVICES="3" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_13b \
  --output_dir ~/WorkspaceLabModels/codellama_13b_v9 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_validate &&\
CUDA_VISIBLE_DEVICES="3" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_13b \
  --output_dir ~/WorkspaceLabModels/codellama_13b_v9 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_quixbugs \
  --do_validate

CUDA_VISIBLE_DEVICES="3" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_13b \
  --output_dir ~/WorkspaceLabModels/codellama_13b_v9 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_defects4j --strict_defects4j --validate_result_split_defects4j \
  --do_validate

# codellama_13b_v10
CUDA_VISIBLE_DEVICES="2" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_13b \
  --output_dir ~/WorkspaceLabModels/codellama_13b_v10 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_quixbugs \
  --do_defects4j --strict_defects4j \
  --do_generate
#여기까지
CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_13b \
  --output_dir ~/WorkspaceLabModels/codellama_13b_v10 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_validate

CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_13b \
  --output_dir ~/WorkspaceLabModels/codellama_13b_v10 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_quixbugs \
  --do_validate

CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_13b \
  --output_dir ~/WorkspaceLabModels/codellama_13b_v10 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_defects4j --strict_defects4j --validate_result_split_defects4j \
  --do_validate

# codellama_34b_v9
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
#여기까지
# codellama_34b_v10
CUDA_VISIBLE_DEVICES="1" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_34b \
  --output_dir ~/WorkspaceLabModels/codellama_34b_v10 \
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
  --output_dir ~/WorkspaceLabModels/codellama_34b_v10 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_validate

CUDA_VISIBLE_DEVICES="1" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_34b \
  --output_dir ~/WorkspaceLabModels/codellama_34b_v10 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_quixbugs \
  --do_validate

CUDA_VISIBLE_DEVICES="1" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_34b \
  --output_dir ~/WorkspaceLabModels/codellama_34b_v10 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_defects4j --strict_defects4j --validate_result_split_defects4j \
  --do_validate

# deepseek_coder_v2_lite_base_v10_paramtest
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
