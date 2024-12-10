## generation only
```shell
CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/incoder-6B-finetune \
  --output_dir ~/WorkspaceLabModels/incoder-6B-finetune-out \
  --full_finetune \
  --bits 16 \
  --fp16 \
  --bf16 False\
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_quixbugs \
  --do_defects4j --strict_defects4j \
  --do_generate

CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/incoder-6B-finetune \
  --output_dir ~/WorkspaceLabModels/incoder-6B-finetune-gen-a6000 \
  --full_finetune \
  --bits 16 \
  --fp16 \
  --bf16 False\
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
CUDA_VISIBLE_DEVICES="3" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/incoder-6B-finetune \
  --output_dir ~/WorkspaceLabModels/incoder-6B-finetune-out \
  --full_finetune \
  --bits 16 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_validate

CUDA_VISIBLE_DEVICES="3" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/incoder-6B-finetune \
  --output_dir ~/WorkspaceLabModels/incoder-6B-finetune-out \
  --full_finetune \
  --bits 16 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_quixbugs \
  --do_validate

CUDA_VISIBLE_DEVICES="3" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/incoder-6B-finetune \
  --output_dir ~/WorkspaceLabModels/incoder-6B-finetune-out \
  --full_finetune \
  --bits 16 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_defects4j --strict_defects4j --validate_result_split_defects4j \
  --do_validate


CUDA_VISIBLE_DEVICES="3" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/incoder-6B-finetune \
  --output_dir ~/WorkspaceLabModels/incoder-6B-finetune-gen-a6000 \
  --full_finetune \
  --bits 16 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_validate

CUDA_VISIBLE_DEVICES="3" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/incoder-6B-finetune \
  --output_dir ~/WorkspaceLabModels/incoder-6B-finetune-gen-a6000 \
  --full_finetune \
  --bits 16 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_quixbugs \
  --do_validate

CUDA_VISIBLE_DEVICES="3" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/incoder-6B-finetune \
  --output_dir ~/WorkspaceLabModels/incoder-6B-finetune-gen-a6000 \
  --full_finetune \
  --bits 16 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_defects4j --strict_defects4j --validate_result_split_defects4j \
  --do_validate
```
