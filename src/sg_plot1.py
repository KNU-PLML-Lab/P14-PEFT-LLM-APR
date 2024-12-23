import json
import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy
import seaborn
import pandas

from sg_plot_common import ValidationSummary, VAL_HE, VAL_QB, VAL_D4JS, VAL_D4JS12, VAL_D4JS20



def plt_rq1_plausible(
  model_list_dirpath: str, # ex. 'models/'
):
  data = []

  model_list = [
    ('CodeGen 6B\nQLoRA', 'codegen_6b_v8'),
    ('CodeGen 6B\nReproduce\nFull FT', 'codegen-6B-finetune-out'),
    ('InCoder 6B\nQLoRA', 'incoder_6b_v9'),
    ('InCoder 6B\nReproduce\nFull FT', 'incoder-6B-finetune-out'),
  ]

  for model_nickname, model_dirname in model_list:
    humaneval_java_validation_summary = ValidationSummary(
      model_list_dirpath=model_list_dirpath,
      model_dirname=model_dirname,
      model_nickname=model_nickname,
      validation_json=VAL_HE
    )
    quixbugs_validation_summary = ValidationSummary(
      model_list_dirpath=model_list_dirpath,
      model_dirname=model_dirname,
      model_nickname=model_nickname,
      validation_json=VAL_QB
    )
    defects4j_strict_12_validation_summary = ValidationSummary(
      model_list_dirpath=model_list_dirpath,
      model_dirname=model_dirname,
      model_nickname=model_nickname,
      validation_json=VAL_D4JS12
    )
    defects4j_strict_20_validation_summary = ValidationSummary(
      model_list_dirpath=model_list_dirpath,
      model_dirname=model_dirname,
      model_nickname=model_nickname,
      validation_json=VAL_D4JS20
    )
    data.append({
      'model': model_nickname,
      'bench': 'HumanEval-Java',
      'plausible': humaneval_java_validation_summary.plausible
    })
    data.append({
      'model': model_nickname,
      'bench': 'QuixBugs',
      'plausible': quixbugs_validation_summary.plausible
    })
    data.append({
      'model': model_nickname,
      'bench': 'Defects4J 1.2',
      'plausible': defects4j_strict_12_validation_summary.plausible
    })
    data.append({
      'model': model_nickname,
      'bench': 'Defects4J 2.0',
      'plausible': defects4j_strict_20_validation_summary.plausible
    })

  data[0:0] = [
    {'model': 'CodeGen 6B\nN. Jiang et al.\nFull FT', 'bench': 'HumanEval-Java', 'plausible': 52},
    {'model': 'CodeGen 6B\nN. Jiang et al.\nFull FT', 'bench': 'QuixBugs', 'plausible': 18},
    {'model': 'CodeGen 6B\nN. Jiang et al.\nFull FT', 'bench': 'Defects4J 1.2', 'plausible': 38},
    {'model': 'CodeGen 6B\nN. Jiang et al.\nFull FT', 'bench': 'Defects4J 2.0', 'plausible': 23},
  ]
  data[12:12] = [
    {'model': 'InCoder 6B\nN. Jiang et al.\nFull FT', 'bench': 'HumanEval-Java', 'plausible': 70},
    {'model': 'InCoder 6B\nN. Jiang et al.\nFull FT', 'bench': 'QuixBugs', 'plausible': 22},
    {'model': 'InCoder 6B\nN. Jiang et al.\nFull FT', 'bench': 'Defects4J 1.2', 'plausible': 41},
    {'model': 'InCoder 6B\nN. Jiang et al.\nFull FT', 'bench': 'Defects4J 2.0', 'plausible': 28},
  ]

  MAX_HUMAN_EVAL_JAVA = 164.0
  MAX_QUIXBUGS = 40.0
  MAX_DEFECTS4J_12 = 130.0
  MAX_DEFECTS4J_20 = 108.0
  for d in data:
    if d['bench'] == 'HumanEval-Java':
      d['plausible'] = d['plausible'] / MAX_HUMAN_EVAL_JAVA * 100
    elif d['bench'] == 'QuixBugs':
      d['plausible'] = d['plausible'] / MAX_QUIXBUGS * 100
    elif d['bench'] == 'Defects4J 1.2':
      d['plausible'] = d['plausible'] / MAX_DEFECTS4J_12 * 100
    elif d['bench'] == 'Defects4J 2.0':
      d['plausible'] = d['plausible'] / MAX_DEFECTS4J_20 * 100

  # DataFrame 생성
  df = pandas.DataFrame(data)
  df['model'] = pandas.Categorical(df['model'], categories=[
    'CodeGen 6B\nN. Jiang et al.\nFull FT',
    'CodeGen 6B\nReproduce\nFull FT',
    'CodeGen 6B\nQLoRA',
    'InCoder 6B\nN. Jiang et al.\nFull FT',
    'InCoder 6B\nReproduce\nFull FT',
    'InCoder 6B\nQLoRA',
  ], ordered=True)

  fig, ax = plt.subplots(figsize=(8.5, 3))

  seaborn.barplot(
      ax=ax,
      data=df,
      x='model',
      y='plausible',
      hue='bench',
      palette='bright',
  )

  # x축 그리드 추가
  ax.grid(axis='y', linestyle='-', alpha=0.3)  # alpha로 투명도 조절 가능
  # 그리드를 막대 뒤로 보내기
  ax.set_axisbelow(True)

  # 그래프 스타일링
  plt.title('', pad=20)
  plt.xlabel('')
  plt.ylabel('Plausible (%)')

  # 범례 위치 조정
  plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='')
  # 범례
  # ax.get_legend().set_title(None)
  # ax.get_legend().remove()

  # 여백 조정
  plt.tight_layout()

  # 파일로 저장
  plt.savefig('outputs/plt_rq1_plausible.png', 
              dpi=300, 
              bbox_inches='tight')



def plt_rq1_gpu(
  model_list_dirpath: str, # ex. 'models/'
):
  data = []

  model_list = [
    ('CodeGen 6B', 'QLoRA', 'codegen_6b_v8'),
    ('CodeGen 6B', 'Full FT', 'codegen-6B-finetune-out'),
    ('InCoder 6B', 'QLoRA', 'incoder_6b_v9'),
    ('InCoder 6B', 'Full FT', 'incoder-6B-finetune-out'),
    # ('InCoder 6B', 'Full FT', 'incoder-6B-finetune-gen-a6000'),
  ]

  for model_basename, model_size, model_dirname in model_list:
    humaneval_java_validation_summary = ValidationSummary(
      model_list_dirpath=model_list_dirpath,
      model_dirname=model_dirname,
      model_nickname=f'{model_basename} {model_size}',
      validation_json=VAL_HE
    )
    quixbugs_validation_summary = ValidationSummary(
      model_list_dirpath=model_list_dirpath,
      model_dirname=model_dirname,
      model_nickname=f'{model_basename} {model_size}',
      validation_json=VAL_QB
    )
    defects4j_strict_validation_summary = ValidationSummary(
      model_list_dirpath=model_list_dirpath,
      model_dirname=model_dirname,
      model_nickname=f'{model_basename} {model_size}',
      validation_json=VAL_D4JS #'defects4j_finetune_output.json'
    )
    all_max_allocated_list = []
    all_max_allocated_list.extend(humaneval_java_validation_summary.max_allocated_list)
    all_max_allocated_list.extend(quixbugs_validation_summary.max_allocated_list)
    all_max_allocated_list.extend(defects4j_strict_validation_summary.max_allocated_list)
    max_allocated_peak = numpy.max(all_max_allocated_list)
    all_plausible = humaneval_java_validation_summary.plausible \
      + quixbugs_validation_summary.plausible \
      + defects4j_strict_validation_summary.plausible

    data.append({
      'model': model_basename,
      'size_str': model_size,
      'gpu_memory_usage_peak': max_allocated_peak,
      'plausible': all_plausible,
    })
    print(f'{model_basename} {model_size}: {max_allocated_peak/1000}GB')

  seaborn.set_theme(style="whitegrid")
  fig, ax = plt.subplots(figsize=(5, 4))

  df = pandas.DataFrame(data)
  
  seaborn.scatterplot(
    data=df,
    x='gpu_memory_usage_peak',
    y='plausible',
    marker='o',
    s=100,
    hue='model',
    ax=ax,
    palette='bright',
  )

  # 범례
  # ax.get_legend().set_title(None)
  ax.get_legend().remove()

  # 그래프 스타일 설정
  ax.set_title('GPU Memory Efficiency')
  ax.set_xlabel('GPU Memory Usage Peak (GB)')
  ax.set_ylabel('Plausible')
  max_time = df['gpu_memory_usage_peak'].max()
  min_time = df['gpu_memory_usage_peak'].min()
  diff_time = max_time - min_time
  ax.set_xlim(min_time - diff_time * 0.2, max_time + diff_time * 0.2)
  max_plausible = df['plausible'].max()
  min_plausible = df['plausible'].min()
  diff_plausible = max_plausible - min_plausible
  ax.set_ylim(min_plausible - diff_plausible * 0.2, max_plausible + diff_plausible * 0.05)


  def gb_formatter(x, p):
    return f'{x/1000:.1f}'
    
  ax.xaxis.set_major_formatter(FuncFormatter(gb_formatter))

  # 모든 점에 대해 라벨 추가
  for idx, row in df.iterrows():
    ax.annotate(row['model'] + '\n' + str(row['size_str']), (row['gpu_memory_usage_peak'], row['plausible']), textcoords="offset points", xytext=(0,-30), ha='center')
  
  # 그리드 추가 (선택사항)
  ax.grid(True, linestyle='--', alpha=0.7)
  
  # 여백 조정
  plt.tight_layout()
  
  # 저장
  fig.savefig("outputs/plt_rq1_gpu.png", dpi=200, bbox_inches='tight')



def plt_rq1_time(
  model_list_dirpath: str, # ex. 'models/'
):
  data = []

  model_list = [
    ('CodeGen 6B', 'QLoRA', 'codegen_6b_v8'),
    ('CodeGen 6B', 'Full FT', 'codegen-6B-finetune-out'),
    ('InCoder 6B', 'QLoRA', 'incoder_6b_v9'),
    ('InCoder 6B', 'Full FT', 'incoder-6B-finetune-out'),
    # ('InCoder 6B', 'Full FT', 'incoder-6B-finetune-gen-a6000'),
  ]

  for model_basename, model_size, model_dirname in model_list:
    humaneval_java_validation_summary = ValidationSummary(
      model_list_dirpath=model_list_dirpath,
      model_dirname=model_dirname,
      model_nickname=f'{model_basename} {model_size}',
      validation_json=VAL_HE
    )
    quixbugs_validation_summary = ValidationSummary(
      model_list_dirpath=model_list_dirpath,
      model_dirname=model_dirname,
      model_nickname=f'{model_basename} {model_size}',
      validation_json=VAL_QB
    )
    defects4j_strict_validation_summary = ValidationSummary(
      model_list_dirpath=model_list_dirpath,
      model_dirname=model_dirname,
      model_nickname=f'{model_basename} {model_size}',
      validation_json=VAL_D4JS #'defects4j_finetune_output.json'
    )
    all_time_list = []
    all_time_list.extend(humaneval_java_validation_summary.time_list)
    all_time_list.extend(quixbugs_validation_summary.time_list)
    all_time_list.extend(defects4j_strict_validation_summary.time_list)
    time_avg = numpy.mean(all_time_list)
    all_plausible = humaneval_java_validation_summary.plausible \
      + quixbugs_validation_summary.plausible \
      + defects4j_strict_validation_summary.plausible

    data.append({
      'model': model_basename,
      'size_str': model_size,
      'time_avg': time_avg,
      'plausible': all_plausible
    })
    print('plt_rq1_time', model_basename, model_size, time_avg, all_plausible)

  seaborn.set_theme(style="whitegrid")
  fig, ax = plt.subplots(figsize=(5, 4))
  
  df = pandas.DataFrame(data)
  
  seaborn.scatterplot(
    data=df,
    x='time_avg',
    y='plausible',
    marker='o',
    s=100,
    hue='model',
    ax=ax,
    palette='bright',
  )

  # 범례
  # ax.get_legend().set_title(None)
  ax.get_legend().remove()

  # 그래프 스타일 설정
  ax.set_title('Patch Generation Time')
  ax.set_xlabel('Time Avg. (s)')
  ax.set_ylabel('Plausible')
  max_time = df['time_avg'].max()
  min_time = df['time_avg'].min()
  diff_time = max_time - min_time
  ax.set_xlim(min_time - diff_time * 0.2, max_time + diff_time * 0.2)
  max_plausible = df['plausible'].max()
  min_plausible = df['plausible'].min()
  diff_plausible = max_plausible - min_plausible
  ax.set_ylim(min_plausible - diff_plausible * 0.2, max_plausible + diff_plausible * 0.05)

  def gb_formatter(x, p):
    return f'{x/1000:.1f}'
    
  ax.xaxis.set_major_formatter(FuncFormatter(gb_formatter))

  # 모든 점에 대해 라벨 추가
  for idx, row in df.iterrows():
    ax.annotate(row['model'] + '\n' + str(row['size_str']), (row['time_avg'], row['plausible']), textcoords="offset points", xytext=(0,-30), ha='center')
  
  # 그리드 추가 (선택사항)
  ax.grid(True, linestyle='--', alpha=0.7)
  
  # 여백 조정
  plt.tight_layout()
  
  # 저장
  fig.savefig("outputs/plt_rq1_gpu_time.png", dpi=200, bbox_inches='tight')