import re
import os

def sum_time_values(text):
  """
  문자열에서 'time: [TIME]s\n' 패턴의 모든 [TIME] 값을 찾아 합계를 반환합니다.
  
  Args:
      text (str): 분석할 입력 문자열
  
  Returns:
      float: 찾은 모든 시간 값의 합계
  """
  # 정규표현식 패턴
  pattern = r'time: (\d+\.?\d*)s\n'
  
  # 모든 매치 찾기
  matches = re.finditer(pattern, text)
  
  # 시간 값들을 추출하여 합계 계산
  total_time = sum(float(match.group(1)) for match in matches)
  
  return total_time

# 사용 예시:
if __name__ == "__main__":
  file = '/home/yglee/WorkspaceLabModels/codegen-6B-finetune/log.txt'
  
  with open(file, 'r') as f:
    text = f.read()

  total_time = sum_time_values(text)
  print(total_time)