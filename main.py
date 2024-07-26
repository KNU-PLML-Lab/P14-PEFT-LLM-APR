import argparse
import torch

import src

print("CUDA Available:", torch.cuda.is_available())

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--task', type=str, choices=['train', 'extract_code_pairs'], default='train')
  args = parser.parse_args()

  if args.task == 'train':
    src.trainer()
  elif args.task == 'extract_code_pairs':
    src.extract_code_pairs()

if __name__ == '__main__':
  main()
