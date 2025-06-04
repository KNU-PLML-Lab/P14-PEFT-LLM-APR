# 초기 테스트 플롯 코드
import json
import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy
import seaborn
import pandas

from sg_plot_common import ValidationSummary, VAL_HE, VAL_QB, VAL_D4JS, VAL_D4JS12, VAL_D4JS20



def plt_codegen_incoder_per_bench_time(
    model_list_dirpath: str,
):
  data = []
  



def plt_gpu_memory_usage_line(
  model_list_dirpath: str, # ex. 'models/'
):
  data = []

  model_list = [
    ('CodeGen', '6B', 'codegen_6b_v8'),
    ('CodeGen Full FT', '6B Full FT', 'codegen-6B-finetune-out'),
    ('CodeGen', '16B', 'codegen_16b_v8_2'),
    ('InCoder', '6B', 'incoder_6b_v9'),
    ('CodeLlama', '7B', 'codellama_7b_v10'),
    ('CodeLlama', '13B', 'codellama_13b_v8'),
    ('CodeLlama', '34B', 'codellama_34b_v8'),
    ('StarCoder2', '7B', 'starcoder2_7b_v8'),
    ('Qwen2.5 Coder', '7B', 'qwen2.5_coder_7b_v10'),
    # ('DSCoderV2', 'Lite(16B)', 'deepseek_coder_v2_lite_base_v8')
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
      validation_json=VAL_D4JS
    )
    all_max_allocated_list = []
    all_max_allocated_list.extend(humaneval_java_validation_summary.max_allocated_list)
    all_max_allocated_list.extend(quixbugs_validation_summary.max_allocated_list)
    all_max_allocated_list.extend(defects4j_strict_validation_summary.max_allocated_list)
    # max_allocated_avg = numpy.mean(all_max_allocated_list)
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
    if model_basename == 'InCoder':
      data.append({
        'model': 'InCoder Full FT',
        'size_str': '6B Full FT',
        'gpu_memory_usage_peak': 26000,
        'plausible': 161,
      })

  seaborn.set_theme(style="whitegrid")
  fig, ax = plt.subplots(figsize=(7, 4))

  df = pandas.DataFrame(data)
  
  seaborn.lineplot(
    data=df,
    x='gpu_memory_usage_peak',
    y='plausible',
    marker='o',
    markersize=10,
    hue='model',
    ax=ax,
    palette='bright',
  )
  # seaborn.scatterplot(
  #   data=df,
  #   x='gpu_memory_usage_peak',
  #   y='plausible',
  #   marker='o',
  #   s=100,
  #   hue='model',
  #   ax=ax,
  # )

  # 범례
  # ax.get_legend().set_title(None)
  ax.get_legend().remove()

  # 그래프 스타일 설정
  ax.set_title('GPU Memory Efficiency')
  ax.set_xlabel('GPU Memory Usage Peak (GB)')
  ax.set_ylabel('Plausible')


  def gb_formatter(x, p):
    return f'{x/1000:.1f}'
    
  ax.xaxis.set_major_formatter(FuncFormatter(gb_formatter))

  # 모든 점에 대해 라벨 추가
  for idx, row in df.iterrows():
    ax.annotate(row['size_str'], (row['gpu_memory_usage_peak'], row['plausible']), textcoords="offset points", xytext=(0,-15), ha='center')
  
  # X축 레이블 회전 (필요한 경우)
  # plt.xticks(rotation=45)
  
  # 그리드 추가 (선택사항)
  ax.grid(True, linestyle='--', alpha=0.7)
  
  # 여백 조정
  plt.tight_layout()
  
  # 저장
  fig.savefig("outputs/plot_gpu_memory_usage_line.png", dpi=200, bbox_inches='tight')



def plt_time_line(
  model_list_dirpath: str, # ex. 'models/'
):
  data = []

  model_list = [
    ('CodeGen', '6B', 'codegen_6b_v8'),
    ('CodeGen Full FT', '6B Full FT', 'codegen-6B-finetune-out'),
    ('CodeGen', '16B', 'codegen_16b_v8_2'),
    ('InCoder', '6B', 'incoder_6b_v9'),
    ('CodeLlama', '7B', 'codellama_7b_v10'),
    ('CodeLlama', '13B', 'codellama_13b_v8'),
    # ('CodeLlama', '34B', 'codellama_34b_v8'),
    ('StarCoder2', '7B', 'starcoder2_7b_v8'),
    ('Qwen2.5 Coder', '7B', 'qwen2.5_coder_7b_v10'),
    # ('DSCoderV2', 'Lite(16B)', 'deepseek_coder_v2_lite_base_v8')
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
      validation_json=VAL_D4JS
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
    if model_basename == 'InCoder':
      data.append({
        'model': 'InCoder Full FT',
        'size_str': '6B Full FT',
        'time_avg': 5000,
        'plausible': 161
      })

  

  seaborn.set_theme(style="whitegrid")
  fig, ax = plt.subplots(figsize=(7, 4))
  
  df = pandas.DataFrame(data)
  
  seaborn.lineplot(
    data=df,
    x='time_avg',
    y='plausible',
    marker='o',
    markersize=10,
    hue='model',
    ax=ax,
    palette='bright',
  )
  # seaborn.scatterplot(
  #   data=df,
  #   x='time_avg',
  #   y='plausible',
  #   marker='o',
  #   s=100,
  #   hue='model',
  #   ax=ax
  # )

  # 범례
  ax.get_legend().set_title('')
  # ax.get_legend().remove()

  # 그래프 스타일 설정
  ax.set_title('Patch Generation Time')
  ax.set_xlabel('Time Avg. (s)')
  ax.set_ylabel('Plausible')
  ax.set_xlim(0, 40000)

  def gb_formatter(x, p):
    return f'{x/1000:.1f}'
    
  ax.xaxis.set_major_formatter(FuncFormatter(gb_formatter))

  # 모든 점에 대해 라벨 추가
  for idx, row in df.iterrows():
    ax.annotate(row['size_str'], (row['time_avg'], row['plausible']), textcoords="offset points", xytext=(0,-15), ha='center')
  
  # X축 레이블 회전 (필요한 경우)
  # plt.xticks(rotation=45)
  
  # 그리드 추가 (선택사항)
  ax.grid(True, linestyle='--', alpha=0.7)
  
  # 여백 조정
  plt.tight_layout()
  
  # 저장
  fig.savefig("outputs/plot_time_line.png", dpi=200, bbox_inches='tight')

def plt_steps2():
  data = {
      'model': [
          'CodeGen 6B', 'InCoder 6B', 'CodeLlama 7B', 'StarCoder2 7B',
          'Qwen2.5 Coder 7B', 'CodeGen 16B', 'CodeLlama 13B', 
          'CodeLlama 34B', 'DeepSeek Coder V2'
      ],
      '100': [53, 0, 26, 76, 81, 64, 61, 77, 97],
      '1000': [59, 77, 68, 86, 112, 73, 85, 80, 97],
      '10000': [79, 70, 82, 69, 105, 84, 94, 105, 108],
      '50000': [76, 75, 86, 86, 115, 79, 100, 96, 109],
      '129000': [76, 74, 86, 86, 117, 77, 89, 101, 111]
  }

  # DataFrame 생성
  df = pandas.DataFrame(data)

  # 데이터 재구조화 (melting)
  df_melted = df.melt(
      id_vars=['model'],
      var_name='data_size',
      value_name='performance'
  )

  # 데이터 크기를 숫자로 변환
  df_melted['data_size'] = df_melted['data_size'].astype(int)

  # 그래프 스타일 설정
  # plt.style.use('seaborn')
  plt.figure(figsize=(8, 6))

  # Line plot 생성
  seaborn.lineplot(
      data=df_melted,
      x='data_size',
      y='performance',
      hue='model',
      marker='o',
      palette='bright',
  )

  # x축 로그 스케일로 변경
  plt.xscale('log')

  # 그래프 스타일링
  plt.title('', pad=20)
  plt.xlabel('Training Data Size')
  plt.ylabel('HumanEval-Java')

  # x축 레이블 수정
  plt.xticks(
      [100, 1000, 10000, 50000, 129000],
      ['100', '1,000', '10K', '50K', '129K']
  )

  # 범례 위치 조정
  plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='')

  # 여백 조정
  plt.tight_layout()

  # 파일로 저장
  plt.savefig('outputs/step.png', 
              dpi=300, 
              bbox_inches='tight')
