import os
import subprocess

from . import d4j1_1_analytics

def __d4j_single_fixed_lines(project_id, fix_id):
  print(f'🩹⛏️ Get fixed lines for {project_id} {fix_id}...')
  # 환경 변수 설정
  os.environ['D4J_HOME'] = f'{os.environ["PROJECT_ABSOLUTE_PATH"]}/defects4j'
  # sloccount 프로그램 필요 'apt install sloccount'
  # os.environ['SLOC_HOME'] = '/usr/bin'

  # Bash 스크립트 실행
  process = subprocess.Popen([
    './fault-localization-data/d4j_integration/get_fixed_lines.sh',
    str(project_id),
    str(fix_id),
    f'{os.environ["PROJECT_ABSOLUTE_PATH"]}/outputs/d4j_fixed_lines',
  ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

  output = []

  # 출력 내용을 실시간으로 읽고 콘솔에 출력
  for line in iter(process.stdout.readline, ''):
    print(line, end='')  # 콘솔에 출력
    output.append(line)  # 결과를 리스트에 저장

  process.stdout.close()
  process.wait()

  # 전체 출력 결과를 하나의 문자열로 결합
  captured_output = ''.join(output)

def __d4j_single_buggy_lines(project_id, fix_id):
  print(f'🪳⛏️ Get buggy lines for {project_id} {fix_id}...')
  # 환경 변수 설정
  os.environ['D4J_HOME'] = f'{os.environ["PROJECT_ABSOLUTE_PATH"]}/defects4j'
  # sloccount 프로그램 필요 'apt install sloccount'
  os.environ['SLOC_HOME'] = '/usr/bin'

  # Bash 스크립트 실행
  process = subprocess.Popen([
    './fault-localization-data/d4j_integration/get_buggy_lines.sh',
    str(project_id),
    str(fix_id),
    f'{os.environ["PROJECT_ABSOLUTE_PATH"]}/outputs/d4j_buggy_lines',
  ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

  output = []

  # 출력 내용을 실시간으로 읽고 콘솔에 출력
  for line in iter(process.stdout.readline, ''):
    print(line, end='')  # 콘솔에 출력
    output.append(line)  # 결과를 리스트에 저장

  process.stdout.close()
  process.wait()

  # 전체 출력 결과를 하나의 문자열로 결합
  captured_output = ''.join(output)
  

def d4j1_2_extract_lines():
  # 프로젝트 정보 가져오기
  pid_problem_count = d4j1_1_analytics()

  # 프로젝트별로 버그 정보를 가져옴
  for pid, problem_count in pid_problem_count.items():
    for fix_id in range(1, problem_count + 1):
      __d4j_single_fixed_lines(pid, fix_id)
      __d4j_single_buggy_lines(pid, fix_id)

  print('✅ Done!')
