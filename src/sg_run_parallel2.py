import asyncio
from concurrent.futures import ThreadPoolExecutor
import subprocess
# dotenv 패키지가 있으면.env 파일의 환경 변수를 로드합니다.
from dotenv import load_dotenv
load_dotenv()

def last_three_line_log(log: str) -> str:
  return '\n'.join(log.split('\n')[-3:])

async def execute_command(command):
  """Execute a single command and return its output"""
  process = await asyncio.create_subprocess_shell(
    command,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
  )
  stdout, stderr = await process.communicate()
  return {
    'command': command,
    'stdout': stdout.decode(),
    'stderr': stderr.decode(),
    'return_code': process.returncode
  }

async def execute_parallel_commands(command_group):
  """Execute a group of commands in parallel"""
  tasks = [execute_command(cmd) for cmd in command_group]
  return await asyncio.gather(*tasks)

async def main(nested_commands):
  """Execute nested command groups sequentially, with parallel execution within each group"""
  results = []
  for command_group in nested_commands:
    group_result = await execute_parallel_commands(command_group)
    results.append(group_result)
  return results

if __name__ == "__main__":
  commands = [
    ['python -m src.sg_tg_report 테스트 시작'],

    ["""
CUDA_VISIBLE_DEVICES="3" python ./src/sg_finetune.py \
    --run_name qwen2.5_coder_7b_v15 \
    --model_name_or_path ~/wlm/qwen2.5_coder_7b \
    --output_dir ~/wlm/qwen2.5_coder_7b_v15 \
    --dataset ./data/finetune_training.jsonl \
    --validation_dataset ./data/finetune_validation.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 0.00001 \
    --lora_r=256 \
    --lora_alpha=512 \
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
    """],
    ['python -m src.sg_tg_report qwen2.5_coder_7b_v15 학습 끝'],

    ["""
CUDA_VISIBLE_DEVICES="3" python src/sg_bench.py \
  --model_name_or_path ~/wlm/qwen2.5_coder_7b \
  --output_dir ~/wlm/qwen2.5_coder_7b_v15 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_humaneval \
  --do_quixbugs \
  --do_defects4j --strict_defects4j \
  --do_generate
    """],
    ['python -m src.sg_tg_report qwen2.5_coder_7b_v15 생성 끝'],

    ["""
CUDA_VISIBLE_DEVICES="1" python src/sg_bench.py \
  --model_name_or_path ~/wlm/qwen2.5_coder_7b \
  --output_dir ~/wlm/qwen2.5_coder_7b_v15 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_humaneval \
  --do_validate
    ""","""
CUDA_VISIBLE_DEVICES="2" python src/sg_bench.py \
  --model_name_or_path ~/wlm/qwen2.5_coder_7b \
  --output_dir ~/wlm/qwen2.5_coder_7b_v15 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_quixbugs \
  --do_generate \
  --do_validate
    """],
    ["killall java"],
    ['python -m src.sg_tg_report qwen2.5_coder_7b_v15 HE QB 검증 끝'],

    ["""
CUDA_VISIBLE_DEVICES="3" python src/sg_bench.py \
  --model_name_or_path ~/wlm/qwen2.5_coder_7b \
  --output_dir ~/wlm/qwen2.5_coder_7b_v15 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --trust_remote_code \
  --do_defects4j --strict_defects4j --validate_result_split_defects4j \
  --do_validate
    """],
    ["killall java"],
    ['python -m src.sg_tg_report qwen2.5_coder_7b_v15 D4J 검증 끝'],

    # ["""

    # ""","""

    # ""","""

    # ""","""

    # """],
    # ["killall java"],
    # ['{ curl -s -m 10 "https://api.telegram.org/bot293512843%3AAAG1t-wJZ6pk_Kxf17XidWcQwFFqVW2XSfM/sendMessage?chat_id=-1002326438945&text=checkpoint" 2> /dev/null > /dev/null & } 2>/dev/null;disown &>/dev/null'],
  ]
  
  results = asyncio.run(main(commands))
  
  # Print results
  for group_idx, group_results in enumerate(results):
    print(f"\nGroup {group_idx + 1} Results:")
    for result in group_results:
      print(f"Command: {result['command']}")
      print(f"Output: {last_three_line_log(result['stdout'].strip())}")
      if result['stderr']:
        print(f"Error: {last_three_line_log(result['stderr'].strip())}")
      print(f"Return Code: {result['return_code']}\n")