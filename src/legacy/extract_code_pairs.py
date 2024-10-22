import os
import shutil
import chardet

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        rawdata = f.read()
    result = chardet.detect(rawdata)
    return result['encoding']

def read_file(file_path):
    encoding = detect_encoding(file_path)
    with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
        return f.read()

def _extract_code_pairs(project, bug_id):
  buggy_dir = f"outputs/{project}_{bug_id}_buggy"
  fixed_dir = f"outputs/{project}_{bug_id}_fixed"

  buggy_files = []
  fixed_files = []

  for root, dirs, files in os.walk(buggy_dir):
    for file in files:
      if file.endswith(".java"):
        buggy_files.append(os.path.join(root, file))

  for root, dirs, files in os.walk(fixed_dir):
    for file in files:
      if file.endswith(".java"):
        fixed_files.append(os.path.join(root, file))

  code_pairs = []
  for buggy_file in buggy_files:
    fixed_file = buggy_file.replace(buggy_dir, fixed_dir)
    if fixed_file in fixed_files:
      buggy_code = read_file(buggy_file)
      fixed_code = read_file(fixed_file)

      # Skip if buggy code is the same as fixed code
      if buggy_code == fixed_code:
        continue

      code_pairs.append((buggy_code, fixed_code))

  return code_pairs

def extract_code_pairs():
  project = "Lang"
  bug_id = "1"

  code_pairs = _extract_code_pairs(project, bug_id)

  # Save the code pairs to a file
  with open("outputs/code_pairs.txt", "w") as f:
    for buggy_code, fixed_code in code_pairs:
      f.write(f"Buggy:\n{buggy_code}\n")
      f.write(f"Fixed:\n{fixed_code}\n")
      f.write("-----\n")
