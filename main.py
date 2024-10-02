
# dotenv 패키지가 있으면.env 파일의 환경 변수를 로드합니다.
try:
  from dotenv import load_dotenv
  load_dotenv()
except ImportError:
  pass

import torch
print("CUDA Available:", torch.cuda.is_available())

import argparse
import src

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--task',
    type = str,
    choices = [
      'train',
      'extract_code_pairs',
      'qlora_test1',
      'd4j1_1_analytics_save',
      'd4j1_2_extract_lines',
      'd4j1_3_report_lines',
      'ccd1_convert_deepfix_to_json',
    ],
    default='train'
  )
  args = parser.parse_args()

  if args.task == 'train':
    src.trainer()
  elif args.task == 'extract_code_pairs':
    src.extract_code_pairs()
  elif args.task == 'qlora_test1':
    src.qlora_test1()
  elif args.task == 'd4j1_1_analytics_save':
    src.d4j1_1_analytics_save()
  elif args.task == 'd4j1_2_extract_lines':
    src.d4j1_2_extract_lines()
  elif args.task == 'd4j1_3_report_lines':
    src.d4j1_3_report_lines()
  elif args.task == 'ccd1_convert_deepfix_to_json':
    src.ccd1_convert_deepfix_to_json([
      'outputs/single_line_r_test.txt',
      'outputs/single_line_r_train.txt',
      'outputs/single_line_r_valid.txt',
    ])

if __name__ == '__main__':
  main()
