import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import chardet

BUGGY_DIR = './outputs/d4j_buggy_lines'
FIXED_DIR = './outputs/d4j_fixed_lines'

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        rawdata = f.read()
    result = chardet.detect(rawdata)
    return result['encoding']

def __d4j1_3_read_fault_lines(dir, endswith):
  line_count_list = []

  # get all files in BUGGY_DIR
  files = os.listdir(dir)
  # filter with *.buggy.lines file
  # Format: {ProjectName}-{BugID}.buggy.lines
  files = [file for file in files if file.endswith(endswith)]
  for file in files:
    file_path = os.path.join(dir, file)
    project_id = file.split('.')[0].split('-')[0]
    bug_id = file.split('.')[0].split('-')[1]
    
    encoding = detect_encoding(file_path)
    with open(file_path, 'r', encoding=encoding) as f:
      # example lines: org/jfree/chart/renderer/category/AbstractCategoryItemRenderer.java#1797#        if (dataset != null) {
      lines = f.readlines()
      for line in lines:
        # find first #
        first_hash = line.find('#')
        class_name = line[:first_hash]
        # find second #
        second_hash = line[first_hash+1:].find('#')
        line_number = line[first_hash+1:first_hash+1+second_hash]
        buggy_line = line[first_hash+1+second_hash+1:]
      line_count_list.append(len(lines))

  return line_count_list

def d4j1_3_report_lines():
  buggy_line_count_list = __d4j1_3_read_fault_lines(BUGGY_DIR, '.buggy.lines')
  fixed_line_count_list = __d4j1_3_read_fault_lines(FIXED_DIR, '.fixed.lines')

  if len(buggy_line_count_list) != len(fixed_line_count_list):
    print("The number of buggy and fixed lines are different.")
    return

  added_line_count_list = []
  for i in range(len(buggy_line_count_list)):
    added_line_count_list.append(fixed_line_count_list[i] + buggy_line_count_list[i])
  
  line_count_list = {}
  for i in range(len(added_line_count_list)):
    if added_line_count_list[i] > 40:
      continue
    if added_line_count_list[i] not in line_count_list:
      line_count_list[added_line_count_list[i]] = 1
    else:
      line_count_list[added_line_count_list[i]] += 1

  # Plot the histogram
  plt.rcParams['font.size'] = 16
  plt.clf() # clear
  plt.bar(line_count_list.keys(), line_count_list.values())
  plt.xlabel('Number of lines')
  plt.ylabel('Frequency')
  plt.title('')
  plt.tight_layout()
  plt.draw()
  plt.savefig('outputs/d4j1_3_report_lines.png', dpi=200)