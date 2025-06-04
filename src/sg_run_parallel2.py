import asyncio
from concurrent.futures import ThreadPoolExecutor
import subprocess

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
    ['{ curl -s -m 10 "https://api.telegram.org/bot293512843%3AAAG1t-wJZ6pk_Kxf17XidWcQwFFqVW2XSfM/sendMessage?chat_id=-1002326438945&text=a0_start" 2> /dev/null > /dev/null & } 2>/dev/null;disown &>/dev/null'],

    ["""
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
    ""","""
CUDA_VISIBLE_DEVICES="1" python src/sg_bench.py \
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
    """],
    ["killall java"],
    ['{ curl -s -m 10 "https://api.telegram.org/bot293512843%3AAAG1t-wJZ6pk_Kxf17XidWcQwFFqVW2XSfM/sendMessage?chat_id=-1002326438945&text=a1_deepseek_coder_v2_lite_base_v11" 2> /dev/null > /dev/null & } 2>/dev/null;disown &>/dev/null'],

    ["""
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
    ""","""
CUDA_VISIBLE_DEVICES="1" python src/sg_bench.py \
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
    """],
    ["killall java"],
    ['{ curl -s -m 10 "https://api.telegram.org/bot293512843%3AAAG1t-wJZ6pk_Kxf17XidWcQwFFqVW2XSfM/sendMessage?chat_id=-1002326438945&text=a2_deepseek_coder_v2_lite_base_v12" 2> /dev/null > /dev/null & } 2>/dev/null;disown &>/dev/null'],

    ["""
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
    ""","""
CUDA_VISIBLE_DEVICES="1" python src/sg_bench.py \
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
    """],
    ["killall java"],
    ['{ curl -s -m 10 "https://api.telegram.org/bot293512843%3AAAG1t-wJZ6pk_Kxf17XidWcQwFFqVW2XSfM/sendMessage?chat_id=-1002326438945&text=adf1" 2> /dev/null > /dev/null & } 2>/dev/null;disown &>/dev/null'],

    ["""
CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_34b \
  --output_dir ~/WorkspaceLabModels/codellama_34b_v10_2 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_generate \
  --do_validate
    ""","""
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
    """],
    ["killall java"],
    ['{ curl -s -m 10 "https://api.telegram.org/bot293512843%3AAAG1t-wJZ6pk_Kxf17XidWcQwFFqVW2XSfM/sendMessage?chat_id=-1002326438945&text=a3_codellama_34b_v10_2" 2> /dev/null > /dev/null & } 2>/dev/null;disown &>/dev/null'],

    ["""
CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_34b \
  --output_dir ~/WorkspaceLabModels/codellama_34b_v11 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_generate \
  --do_validate
    ""","""
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
    """],
    ["killall java"],
    ['{ curl -s -m 10 "https://api.telegram.org/bot293512843%3AAAG1t-wJZ6pk_Kxf17XidWcQwFFqVW2XSfM/sendMessage?chat_id=-1002326438945&text=a4_codellama_34b_v11" 2> /dev/null > /dev/null & } 2>/dev/null;disown &>/dev/null'],

    ["""
CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_34b \
  --output_dir ~/WorkspaceLabModels/codellama_34b_v10_2 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_defects4j --strict_defects4j --validate_result_split_defects4j \
  --do_generate \
  --do_validate
    ""","""
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
    """],
    ["killall java"],
    ['{ curl -s -m 10 "https://api.telegram.org/bot293512843%3AAAG1t-wJZ6pk_Kxf17XidWcQwFFqVW2XSfM/sendMessage?chat_id=-1002326438945&text=adf2" 2> /dev/null > /dev/null & } 2>/dev/null;disown &>/dev/null'],

    ["""
CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_34b \
  --output_dir ~/WorkspaceLabModels/codellama_34b_v12 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_humaneval \
  --do_generate \
  --do_validate
    ""","""
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
    """],
    ["killall java"],
    ['{ curl -s -m 10 "https://api.telegram.org/bot293512843%3AAAG1t-wJZ6pk_Kxf17XidWcQwFFqVW2XSfM/sendMessage?chat_id=-1002326438945&text=a5_codellama_34b_v12" 2> /dev/null > /dev/null & } 2>/dev/null;disown &>/dev/null'],

    ["""
CUDA_VISIBLE_DEVICES="0" python src/sg_bench.py \
  --model_name_or_path ~/WorkspaceLabModels/codellama_34b \
  --output_dir ~/WorkspaceLabModels/codellama_34b_v12 \
  --do_sample \
  --seed 0 \
  --num_beams 10 \
  --max_new_tokens 128 \
  --do_defects4j --strict_defects4j --validate_result_split_defects4j \
  --do_generate \
  --do_validate
    """],
    ["killall java"],
    ['{ curl -s -m 10 "https://api.telegram.org/bot293512843%3AAAG1t-wJZ6pk_Kxf17XidWcQwFFqVW2XSfM/sendMessage?chat_id=-1002326438945&text=adf3" 2> /dev/null > /dev/null & } 2>/dev/null;disown &>/dev/null'],

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