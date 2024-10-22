import os
import subprocess

def d4j1_1_analytics():
  # 환경 변수 설정
  os.environ['D4J_HOME'] = f'{os.environ["PROJECT_ABSOLUTE_PATH"]}/defects4j'
  # sloccount 프로그램 필요 'apt install sloccount'
  # os.environ['SLOC_HOME'] = '/usr/bin'

  # Bash 스크립트 실행
  process = subprocess.Popen([
    './fault-localization-data/d4j_integration/get_fault_stats.sh'
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
  
  # Extract pid and problem_count per line (echo "\\def\\nReal${pid}Faults{$num_bugs\\xspace}")
  pid_problem_count = {}
  for line in captured_output.split('\n'):
    if line.startswith('\\def\\nReal'):
      pid = line.split('\\nReal')[1].split('Faults')[0]
      problem_count = line.split('Faults{')[1].split('\\xspace')[0]
      # pid가 없는 경우는 전체값임 이게 나왔다는건 다음에 나오는 건 Subject count이므로 break
      if pid == '': break
      pid_problem_count[pid] = int(problem_count)

  return pid_problem_count

def d4j1_1_analytics_debug():
  pid_problem_count = d4j1_1_analytics()
  print(pid_problem_count)

def d4j1_1_analytics_save():
  pid_problem_count = d4j1_1_analytics()
  with open(f'{os.environ["PROJECT_ABSOLUTE_PATH"]}/outputs/d4j1_1_analytics.txt', 'w') as f:
    for pid, problem_count in pid_problem_count.items():
      f.write(f'{pid} {problem_count}\n')
