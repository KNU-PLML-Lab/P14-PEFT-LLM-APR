import codecs
import json
import os
import sys

import numpy
import shutil
import torch
import transformers

import sg_args
import sg_model
import sg_tools

# clm의 코드 가져오기
C_DIR = os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('/') + 1]
# sys.path.append(os.path.join(C_DIR, '../clm/clm-apr/humaneval/'))
import humaneval_command
# sys.path.append(os.path.join(C_DIR, '../clm/clm-apr/quixbugs/'))
import quixbugs_command



def generate_input(
  run_type: str,
  bench_type: str,
  bench_path: str,
  loc_file: str,
  java_project_path: str,
  output_file: str,
  config: dict = None
):
  """
    Java 프로젝트를 통해 각 벤치마크별 코드 생성 입력을 생성

    run_type: 'finetune' | 'codegen'
    bench_type: 'humaneval' | 'quixbugs'
    bench_path: 벤치마크용 HumanEval | Quixbugs 프로젝트 경로
    loc_file: 벤치마크 별 라인 기록 loc 파일
    java_project_path: Jasper Java 프로젝트 경로
    humaneval_output_dir: HumanEval 출력 경로
    output_file: 출력 파일 경로
    config: run_type: 'finetune'이 아닐경우 필요한 설정 (CodeGenInputConfig...)
  """
  # 벤치마크 별 라인 기록 loc 파일 가져오기  
  loc_fp = codecs.open(loc_file, 'r', 'utf-8')

  # 입력 파일 준비
  input_dict = {'config': None, 'data': {}}
  if run_type == 'finetune':
    input_dict['config'] = 'finetune'
  elif run_type == 'codegen':
    input_dict['config'] = config
  else:
    raise ValueError(f'❌ unrecognized run_type {run_type}')

  # 생성 루프
  for line in loc_fp.readlines():
    filename = None
    rem_loc = None
    _add_loc = None
    if bench_type == 'humaneval':
      filename, rem_loc = line.strip().split()
    elif bench_type == 'quixbugs':
      filename, rem_loc, _add_loc = line.strip().split()
    else:
      raise ValueError(f'❌ unrecognized bench_type {bench_type}')

    start, end = rem_loc.split('-')
    end = str(int(end) - 1) if end != start else end
    tmp_file = os.path.join(bench_path, 'tmp.json')

    # 각 벤치마크 별 파일 경로 설정
    buggy_file = None
    if bench_type == 'humaneval':
      buggy_file = os.path.join(
        bench_path,
        'src/main/java/humaneval/buggy',
        f'{filename}.java',
      )
    elif bench_type == 'quixbugs':
      buggy_file = os.path.join(
        bench_path,
        'java_programs',
        f'{filename}.java'
      )
    else:
      raise ValueError(f'❌ unrecognized bench_type {bench_type}')

    # Jasper Java 프로젝트를 통해 입력 생성
    sg_tools.run_java_to_generate_input(
      run_type = run_type,
      java_project_path = java_project_path,
      buggy_file = buggy_file,
      rem_start = start,
      rem_end = end,
      tmp_file = tmp_file,
      config = config
    )
    
    if not os.path.exists(tmp_file):
      print(f'❌ {filename} failed. tmp file not generated')
      continue

    print(f'📜 {filename} input generated. read it...')
    result = json.load(open(tmp_file, 'r'))
    # result is None or empty dict throw error
    if not result or not result['buggy function before']:
      raise ValueError(f'❌ {filename} failed. tmp file is empty')

    if run_type == 'finetune':
      input_dict['data'][filename] = {
        'loc': rem_loc,
        'input': result['buggy function before'] +
          '// buggy lines start:\n' + result['buggy line'] +
          '// buggy lines end:\n' + result['buggy function after'] +
          '// fixed lines: \n',
      }
    elif run_type == 'codegen':
      input_dict['data'][filename] = {
        'loc': rem_loc,
        'input': result['input'],
        'function range': result['function range']
      }

    sg_tools.command(['rm', '-rf', tmp_file])
  with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(input_dict, f, indent=2)



def generate_output(
  model_name: str,
  model: transformers.PreTrainedModel,
  tokenizer: transformers.PreTrainedTokenizer,
  input_file: str,
  output_file: str,
  args: sg_args.GenerationArguments,
) -> None:
  """
    모델을 통해 전치리된 각 벤치마크의 입력으로부터 패치 생성
  """
  # 입력파일 가져오기
  input_dict = json.load(open(input_file, 'r'))
  input_dict['model'] = model_name

  # 모델 탐지
  is_incoder = 'incoder' in model_name.lower()
  if is_incoder:
    print('🧂 incoder detected. dedicate EOS token provide.');
  
  # 생성 메트릭 기록 준비
  device = model.device
  print(f'🔌 Model device: {device}')
  starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
  timings = []
  oom = 0
  memory_allocated, memory_reserved = 0, 0

  # 실행 루프
  with torch.no_grad():
    for filename in input_dict['data']:
      print(f'🏭 Generating {filename}...')
      input_text = input_dict['data'][filename]['input']

      # 모델별 토크나이징 전처리
      input_emb = tokenizer(input_text, return_tensors="pt").to('cuda')
      inputs = {}
      eos_id = None
      if is_incoder:
        inputs['input_ids'] = input_emb.input_ids.to(device)
        # incoder의 경우 EOS 토큰을 '<|endofmask|>'로 지정
        eos_id = tokenizer.convert_tokens_to_ids('<|endofmask|>')
      else: 
        inputs = input_emb.to(device)
        eos_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)

      starter.record()
      try:
        # print(input_emb.input_ids.dtype)
        # print(input_emb.attention_mask.dtype)
        generated_ids = model.generate(
          **inputs,
          max_new_tokens = args.max_new_tokens,
          num_beams = args.num_beams,
          num_return_sequences = args.num_beams,
          early_stopping = True, 

          pad_token_id = eos_id,
          eos_token_id = eos_id,

          generation_config=transformers.GenerationConfig(
            do_sample = args.do_sample,
            max_new_tokens = args.max_new_tokens,
            top_p = args.top_p,
            temperature = args.temperature,
          ),
          use_cache=False
        )
      except Exception as e:
        print(f'❌ {filename} generate failed. OOM counted. {str(e)}')
        oom += 1
        continue

      # 생성 메트릭 기록 중지
      ender.record()
      torch.cuda.synchronize()
      curr_time = starter.elapsed_time(ender)
      timings.append(curr_time)

      # 메모리 사용량 기록
      total_allocated, total_reserved = 0, 0
      total_allocated += torch.cuda.memory_allocated(torch.device(device)) / (1024 * 1024)
      total_reserved += torch.cuda.memory_reserved(torch.device(device)) / (1024 * 1024)
      if total_allocated > memory_allocated:
        memory_allocated = total_allocated
      if total_reserved > memory_reserved:
        memory_reserved = total_reserved
      print(f'(curr_time: {curr_time:.2f}, memory_allocated: {memory_allocated:.2f}MB, memory_reserved: {memory_reserved:.2f}MB, oom: {oom})')

      # 출력 저장
      output = []
      for generated_id in generated_ids:
        output.append(tokenizer.decode(generated_id, skip_special_tokens=False))
      input_dict['data'][filename]['output'] = output
      json.dump(input_dict, open(output_file, 'w'), indent=2)

      # 메모리 해제
      del generated_ids
      torch.cuda.empty_cache()

  input_dict['time'] = int(numpy.sum(timings) / 1000)
  json.dump(input_dict, open(output_file, 'w'), indent=2)



def insert_fix(filename, start_line, end_line, patch):
  """
  end_row is included in the buggy lines / buggy function
  """
  with open(filename, 'r') as file:
    data = file.readlines()

  with open(filename, 'w') as file:
    for i in range(start_line - 1):
      file.write(data[i] + '\n')
    file.write(patch.strip())
    for i in range(end_line, len(data)):
      file.write(data[i])



# TODO: 지금 finetune에 대해서만 구현되어 있음!
def validate_quixbugs(
  model_name: str,
  tokenizer: transformers.PreTrainedTokenizer,
  input_file: str,
  output_file: str,
  tmp_dir: str
):
  model_name = model_name.lower()

  EOS_STR = None
  if 'incoder' in model_name:
    print('🧂 incoder model detected. add EOS token (<|endofmask|>)')
    EOS_STR = '<|endofmask|>'
  else:
    EOS_STR = tokenizer.eos_token

  plausible, total = 0, 0

  if not os.path.exists(tmp_dir):
    quixbugs_command.command_with_timeout(['mkdir', tmp_dir])

  # INJECTED: java_programs_bak 폴더가 없으면 java_programs를 복사
  if not os.path.exists(tmp_dir + "/java_programs_bak"):
    print('📂 java_programs_bak not found. Copy java_programs to java_programs_bak. This is only one-time operation...')
    shutil.copytree(tmp_dir + "/java_programs", tmp_dir + "/java_programs_bak")

  model_output = json.load(open(input_file, 'r'))
  validated_result = {'config': model_output['config'], 'data': {}}
  for proj in model_output['data']:
    print('start validating', proj)
    total += 1
    quixbugs_command.command_with_timeout(['rm', '-rf', tmp_dir + '/java_programs/'])
    quixbugs_command.command_with_timeout(['mkdir', tmp_dir + '/java_programs/'])

    shutil.copyfile(tmp_dir + "/java_programs_bak/" + proj + '.java',
                    tmp_dir + "/java_programs/" + proj + '.java')
    shutil.copyfile(tmp_dir + "/java_programs_bak/Node.java", tmp_dir + "/java_programs/Node.java")
    shutil.copyfile(tmp_dir + "/java_programs_bak/WeightedEdge.java", tmp_dir + "/java_programs/WeightedEdge.java")

    validated_result['data'][proj] = {}
    for key, value in model_output['data'][proj].items():
      if key != 'output':
        validated_result['data'][proj][key] = value
    validated_result['data'][proj]['output'] = []
    start_line, end_line = validated_result['data'][proj]['loc'].split('-')
    end_line = str(int(end_line) - 1) if end_line != start_line else end_line
    current_is_correct = False
    for rank, patch in enumerate(model_output['data'][proj]['output']):
      filename = tmp_dir + "/java_programs/" + proj + '.java'

      patch = sg_tools.ft_output_to_patch(patch, EOS_STR)
      insert_fix(filename, int(start_line), int(end_line), patch)
      
      compile = quixbugs_command.compile_fix(filename, tmp_dir + "/java_programs/")
      correctness = 'uncompilable'
      if compile:
        correctness = quixbugs_command.quixbugs_test_suite(proj, quixbugs_dir=tmp_dir)
        if correctness == 'plausible':
          if not current_is_correct:
            plausible += 1
            current_is_correct = True
          print(plausible, total, rank, "Plausible patch:", patch)
        elif correctness == 'wrong':
          print(plausible, total, rank, "Wrong patch:", patch)
        elif correctness == 'timeout':
          print(plausible, total, rank, "Timeout patch:", patch)
      else:
        print(plausible, total, rank, 'Uncompilable patch:', patch)
      validated_result['data'][proj]['output'].append({
        'patch': patch, 'correctness': correctness
      })
      shutil.copyfile(
        tmp_dir + "/java_programs_bak/" + proj + '.java',
        tmp_dir + "/java_programs/" + proj + '.java'
      )
    json.dump(validated_result, open(output_file, 'w'), indent=2)



def main():
  (
    args,
    _,
    _,
    _,
    generation_args,
    _,
  ) = sg_args.parse_args()

  C_DIR = os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('/') + 1]
  JASPER_DIR = os.path.abspath(os.path.join(C_DIR, '../clm/jasper/'))
  HUMANEVAL_DIR = os.path.abspath(os.path.join(C_DIR, '../clm/humaneval-java/'))
  HUMANEVAL_LOC_FILE = os.path.abspath(os.path.join(C_DIR, '../clm/clm-apr/humaneval/humaneval_loc.txt'))
  QUIXBUGS_DIR = os.path.abspath(os.path.join(C_DIR, '../QuixBugs/'))
  QUIXBUGS_LOC_FILE = os.path.abspath(os.path.join(C_DIR, '../clm/clm-apr/quixbugs/quixbugs_loc.txt'))
  # DEFAULT_PAD_TOKEN = "[PAD]"

  # AutoTokenizer가 CodeLlamaTokenizer를 인식하지 못함
  force_model = None
  if ('code_llama' in args.model_name_or_path.lower()) or ('codellama' in args.model_name_or_path.lower()):
    force_model = 'code_llama'
  model, tokenizer = sg_model.get_model_tokenizer(args, force_model)
  model.config.use_cache = False
  model.eval()

  model_name = sg_tools.nomalize_name_or_path_to_name(args.model_name_or_path) if args.model_name_or_path else 'UnknownModel'

  # Humaneval 테스트
  if generation_args.do_humaneval:

    run_type = 'finetune'
    bench_type = 'humaneval'
    input_file = os.path.join(os.path.abspath(args.output_dir), 'humaneval_finetune_input.json')
    output_file = os.path.join(os.path.abspath(args.output_dir), 'humaneval_finetune_output.json')
    validate_file = os.path.join(os.path.abspath(args.output_dir), 'humaneval_finetune_validate.json')

    if generation_args.do_generate:
      if not os.path.exists(input_file):
        print(f"==========Preparing input of ({bench_type}) benchmark to ({run_type}) model==========")
        generate_input(
          run_type = run_type,
          bench_type = bench_type,
          bench_path = HUMANEVAL_DIR,
          loc_file = HUMANEVAL_LOC_FILE,
          java_project_path = JASPER_DIR,
          output_file = input_file
        )
        print(f"==========Input written to {input_file}==========")
      
      print(f"==========Generating output of ({bench_type}) benchmark by ({run_type}) model==========")
      generate_output(
        model_name = model_name,
        model = model,
        tokenizer = tokenizer,
        input_file = input_file,
        output_file = output_file,
        args = generation_args,
      )
      print(f"==========Output written to {output_file}==========")

  if generation_args.do_quixbugs:
    run_type = 'finetune'
    bench_type = 'quixbugs'
    input_file = os.path.join(os.path.abspath(args.output_dir), 'quixbugs_finetune_input.json')
    output_file = os.path.join(os.path.abspath(args.output_dir), 'quixbugs_finetune_output.json')
    validate_file = os.path.join(os.path.abspath(args.output_dir), 'quixbugs_finetune_validate.json')

    if generation_args.do_generate:
      if not os.path.exists(input_file):
        print(f"==========Preparing input of ({bench_type}) benchmark to ({run_type}) model==========")
        generate_input(
          run_type = run_type,
          bench_type = bench_type,
          bench_path = QUIXBUGS_DIR,
          loc_file = QUIXBUGS_LOC_FILE,
          java_project_path = JASPER_DIR,
          output_file = input_file
        )
        print(f"==========Input written to {input_file}==========")
      
      print(f"==========Generating output of ({bench_type}) benchmark by ({run_type}) model==========")
      generate_output(
        model_name = model_name,
        model = model,
        tokenizer = tokenizer,
        input_file = input_file,
        output_file = output_file,
        args = generation_args,
      )
      print(f"==========Output written to {output_file}==========")
    
    if generation_args.do_validate:
      print(f"==========Validating output of ({bench_type}) benchmark by ({run_type}) model==========")
      validate_quixbugs(
        model_name = model_name,
        tokenizer = tokenizer,
        input_file = output_file,
        output_file = validate_file,
        tmp_dir = QUIXBUGS_DIR
      )
      print(f"==========Output validated. Written to {validate_file}==========")

if __name__ == '__main__':
  main()
