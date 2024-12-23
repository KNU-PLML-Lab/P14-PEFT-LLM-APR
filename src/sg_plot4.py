import json
import os
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy
import seaborn
import pandas

import sg_tools
from sg_plot_common import ValidationSummary, VAL_HE, VAL_QB, VAL_D4JS, VAL_D4JS12, VAL_D4JS20



VAL_HE = 'humaneval_finetune_validate.json'
VAL_QB = 'quixbugs_finetune_validate.json'
VAL_D4J = 'defects4j_finetune_validate.json'
VAL_D4JS = 'defects4j_finetune_strict_validate.json'
VAL_D4JS12 = 'defects4j_finetune_strict_validate_v12.json'
VAL_D4JS20 = 'defects4j_finetune_strict_validate_v20.json'

class ValidationSummary:
  def __init__(
    self,
    model_nickname: Optional[str] = None, # ex. 'CodeLLaMa 13B'
    model_list_dirpath: Optional[str] = None, # ex. 'models/'
    model_dirname: Optional[str] = None, # ex. 'codellama_13b_v8'
    checkpoint_num: Optional[int] = None, # ex. 1000
    validation_json: Optional[str] = None, # ex. 'humaneval_finetune_validate.json'

    force_json_path: Optional[str] = None
  ):
    self.json_path = ''
    self.model_name = ''

    self.plausible = 0
    self.correct = 0
    self.total = 0
    self.allocated_list = []
    self.allocated_avg = 0
    self.allocated_peak = 0
    self.reserved_list = []
    self.reserved_avg = 0
    self.reserved_peak = 0
    self.max_allocated_list = []
    self.max_allocated_avg = 0
    self.max_allocated_peak = 0
    self.time_list = []
    self.time_avg = 0
    self.time_peak = 0

    if force_json_path:
      self.json_path = force_json_path
    else:
      if checkpoint_num:
        self.json_path = os.path.join(str(model_list_dirpath), str(model_dirname), f'{sg_tools.PREFIX_CHECKPOINT_DIR}{checkpoint_num}', 'adapter_model', str(validation_json))
      else:
        self.json_path = os.path.join(str(model_list_dirpath), str(model_dirname), str(validation_json))
    
    self.model_name = model_nickname or 'UnknownModel'

    if not os.path.exists(self.json_path):
      raise FileNotFoundError(f'💥 Error: json file not found({self.json_path})')

    self.__read_json()

  def __read_json(self):
    with open(self.json_path, 'r') as f:
      validation_data = json.load(f)
      result = validation_data.get('result')
      if result is None:
        raise KeyError(f'💥 Error: result key not found in json({self.json_path})')
      self.plausible = result.get('plausible')
      self.correct = result.get('correct')
      self.total = result.get('total')

      for key, data in validation_data.get('data').items():
        _output = data.get('output')
        meta = data.get('meta')
        if meta is None:
          print(f'💥 Error: meta key not found in json({self.json_path})({key})')
          continue
        allocated = meta.get('allocated')
        reserved = meta.get('reserved')
        max_allocated = meta.get('max_allocated')
        _time = meta.get('time')

        if allocated:
          self.allocated_list.append(allocated)
        else:
          print(f'💥 Error: "allocated" key not found in json({self.json_path})({key})')
        if reserved:
          self.reserved_list.append(reserved)
        else:
          print(f'💥 Error: "reserved" key not found in json({self.json_path})({key})')
        if max_allocated:
          self.max_allocated_list.append(max_allocated)
        else:
          print(f'💥 Error: "max_allocated" key not found in json({self.json_path})({key})')
        if not _time:
          print(f'💥 Error: "time" key not found in json({self.json_path})({key})')
        elif not _output:
          print(f'💥 Error: "output" key not found in json({self.json_path})({key})')
        else:
          # plausible인 경우만 시간을 기록
          for _o in _output:
            if _o.get('correctness') == 'plausible':
              self.time_list.append(_time)
              break

      self.allocated_avg = numpy.mean(self.allocated_list)
      self.allocated_peak = numpy.max(self.allocated_list)
      self.reserved_avg = numpy.mean(self.reserved_list)
      self.reserved_peak = numpy.max(self.reserved_list)
      self.max_allocated_avg = numpy.mean(self.max_allocated_list)
      self.max_allocated_peak = numpy.max(self.max_allocated_list)
      self.time_avg = numpy.mean(self.time_list)
      self.time_peak = numpy.max(self.time_list)

  def __str__(self):
    return f'{self.model_name}({self.json_path}): plausible({self.plausible}), correct({self.correct}), total({self.total})'
  
  def __repr__(self):
    return self.__str__()



def read_validation_summary():
  pass


# def test_plot():
#   seaborn.set_theme(style="darkgrid")

# def read_plausible_from_validation_json(json_path):
#   plausible = 0
#   with open(json_path, 'r') as f:
#     data = json.load(f)
#     result = data.get('result')
#     if result is None:
#       print(f'💥 Error: result key not found in json({json_path})')
#       exit(1)
#     plausible = result.get('plausible')
#     if plausible is None:
#       print(f'💥 Error: plausible key not found in json({json_path})')
#       exit(1)
#     plausible = int(plausible)
#   return plausible

# def plt_steps():
#   step_data_paths = {
#     'CodeGen 6B': {
#       '100': '/home/yglee/WorkspaceLabModels/codegen_6b_v8/checkpoint-13/humaneval_finetune_validate.json',
#       '1000': '/home/yglee/WorkspaceLabModels/codegen_6b_v8/checkpoint-125/humaneval_finetune_validate.json',
#       '10K': '/home/yglee/WorkspaceLabModels/codegen_6b_v8/checkpoint-1250/humaneval_finetune_validate.json',
#       '50K': '/home/yglee/WorkspaceLabModels/codegen_6b_v8/checkpoint-6250/humaneval_finetune_validate.json',
#       '129K': '/home/yglee/WorkspaceLabModels/codegen_6b_v8/humaneval_finetune_validate.json',
#     },
#     'Incoder 6B': {
#       '100': '/home/yglee/WorkspaceLabModels/incoder_6b_v9/checkpoint-13/humaneval_finetune_validate.json',
#       '1000': '/home/yglee/WorkspaceLabModels/incoder_6b_v9/checkpoint-125/humaneval_finetune_validate.json',
#       '10K': '/home/yglee/WorkspaceLabModels/incoder_6b_v9/checkpoint-1250/humaneval_finetune_validate.json',
#       '50K': '/home/yglee/WorkspaceLabModels/incoder_6b_v9/checkpoint-6250/humaneval_finetune_validate.json',
#       '129K': '/home/yglee/WorkspaceLabModels/incoder_6b_v9/humaneval_finetune_validate.json',
#     },
#     'CodeLLaMa 7B': {
#       '100': '/home/yglee/WorkspaceLabModels/codellama_7b_v9/checkpoint-13/humaneval_finetune_validate.json',
#       '1000': '/home/yglee/WorkspaceLabModels/codellama_7b_v9/checkpoint-125/humaneval_finetune_validate.json',
#       '10K': '/home/yglee/WorkspaceLabModels/codellama_7b_v9/checkpoint-1250/humaneval_finetune_validate.json',
#       '50K': '/home/yglee/WorkspaceLabModels/codellama_7b_v9/checkpoint-6250/humaneval_finetune_validate.json',
#       '129K': '/home/yglee/WorkspaceLabModels/codellama_7b_v9/humaneval_finetune_validate.json',
#     },
#     'CodeLLaMa 13B': {
#       '100': '/home/yglee/WorkspaceLabModels/codellama_13b_v8/checkpoint-13/humaneval_finetune_validate.json',
#       '1000': '/home/yglee/WorkspaceLabModels/codellama_13b_v8/checkpoint-125/humaneval_finetune_validate.json',
#       '10K': '/home/yglee/WorkspaceLabModels/codellama_13b_v8/checkpoint-1250/humaneval_finetune_validate.json',
#       '50K': '/home/yglee/WorkspaceLabModels/codellama_13b_v8/checkpoint-6250/humaneval_finetune_validate.json',
#       '129K': '/home/yglee/WorkspaceLabModels/codellama_13b_v8/humaneval_finetune_validate.json',
#     },
#     'CodeLLaMa 34B': {
#       '100': '/home/yglee/WorkspaceLabModels/codellama_34b_v8/checkpoint-25/humaneval_finetune_validate.json',
#       '1000': '/home/yglee/WorkspaceLabModels/codellama_34b_v8/checkpoint-250/humaneval_finetune_validate.json',
#       '10K': '/home/yglee/WorkspaceLabModels/codellama_34b_v8/checkpoint-2500/humaneval_finetune_validate.json',
#       '50K': '/home/yglee/WorkspaceLabModels/codellama_34b_v8/checkpoint-12500/humaneval_finetune_validate.json',
#       '129K': '/home/yglee/WorkspaceLabModels/codellama_34b_v8/humaneval_finetune_validate.json',
#     },
#     'DeepSeek Coder V2 Lite Base': {
#       '100': '/home/yglee/WorkspaceLabModels/deepseek_coder_v2_lite_base_v8/checkpoint-13/humaneval_finetune_validate.json',
#       '1000': '/home/yglee/WorkspaceLabModels/deepseek_coder_v2_lite_base_v8/checkpoint-125/humaneval_finetune_validate.json',
#       '10K': '/home/yglee/WorkspaceLabModels/deepseek_coder_v2_lite_base_v8/checkpoint-1250/humaneval_finetune_validate.json',
#       '50K': '/home/yglee/WorkspaceLabModels/deepseek_coder_v2_lite_base_v8/checkpoint-6250/humaneval_finetune_validate.json',
#       '129K': '/home/yglee/WorkspaceLabModels/deepseek_coder_v2_lite_base_v8/humaneval_finetune_validate.json',
#     }
#   }

#   data = []
#   for model_name, steps in step_data_paths.items():
#     for step, json_path in steps.items():
#       plausible = read_plausible_from_validation_json(os.path.abspath(json_path))
#       data.append({
#         'Model': model_name,
#         'Step': step,
#         'Plausible': plausible
#       })

#   fig, ax = plt.subplots(figsize=(10, 6))
  
#   df = pandas.DataFrame(data)
  
#   # x축 순서를 지정
#   x_order = ['100', '1000', '10K', '50K', '129K']
#   df['Step'] = pandas.Categorical(df['Step'], categories=x_order, ordered=True)
  
#   # 선 그래프 그리기
#   seaborn.lineplot(data=df, x='Step', y='Plausible', hue='Model', ax=ax) # marker='o'
  
#   # 그래프 스타일 설정
#   ax.set_title('Plausible vs Step')
#   ax.set_xlabel('Step')
#   ax.set_ylabel('Plausible')
  
#   # Y축 범위 설정 (데이터의 최소값과 최대값 사용)
#   y_min = df['Plausible'].min()
#   y_max = df['Plausible'].max()
#   ax.set_ylim(y_min, y_max)
  
#   # X축 레이블 회전 (필요한 경우)
#   plt.xticks(rotation=0)
  
#   # 그리드 추가 (선택사항)
#   ax.grid(True, linestyle='--', alpha=0.7)
  
#   # 여백 조정
#   plt.tight_layout()
  
#   # 저장
#   fig.savefig("outputs/plot_steps.png", dpi=200, bbox_inches='tight')



# def plt_humaneval_java_gpu_memory_usage(
#   model_list_dirpath: str, # ex. 'models/'
# ):
#   data = [] # {'model', 'gpu_memory_usage_avg', 'gpu_memory_usage_peak'}

#   model_list = [
#     ('CodeGen 6B', 'codegen_6b_v8'),
#     ('Incoder 6B', 'incoder_6b_v9'),
#     ('CodeLLaMa 7B', 'codellama_7b_v9'),
#     ('CodeLLaMa 13B', 'codellama_13b_v8'),
#     ('CodeLLaMa 34B', 'codellama_34b_v8'),
#     ('DeepSeek Coder V2 Lite Base', 'deepseek_coder_v2_lite_base_v8')
#   ]

#   for model_nickname, model_dirname in model_list:
#     validation_summary = ValidationSummary(
#       model_list_dirpath=model_list_dirpath,
#       model_dirname=model_dirname,
#       model_nickname=model_nickname,
#       validation_json=VAL_HE
#     )
#     data.append({
#       'model': model_nickname,
#       'gpu_memory_usage_avg': validation_summary.max_allocated_avg,
#       'gpu_memory_usage_peak': validation_summary.max_allocated_peak
#     })

#   fig, ax = plt.subplots(figsize=(10, 6))
  
#   df = pandas.DataFrame(data)

#   seaborn.set_theme(style="whitegrid")
  
#   # 박스 그래프 그리기
#   seaborn.set_color_codes("pastel")
#   seaborn.barplot(x="model", y="gpu_memory_usage_peak", data=df,
#               label="GPU Memory Usage Peak", color="b")

#   # 박스 그래프 그리기
#   seaborn.set_color_codes("muted")
#   seaborn.barplot(x="model", y="gpu_memory_usage_avg", data=df,
#             label="GPU Memory Usage Avg.", color="b")

#   # 그래프 스타일 설정
#   ax.set_title('GPU Memory Usage Comparison')
#   ax.set_xlabel('Model')
#   ax.set_ylabel('GPU Memory Usage')

#   seaborn.despine(left=True, bottom=True)

#   # Y축 범위 설정 (데이터의 최소값과 최대값 사용)
#   y_min = df['gpu_memory_usage_avg'].min()
#   y_max = df['gpu_memory_usage_peak'].max()
#   y_diff = y_max - y_min
#   y_min -= y_diff * 0.1
#   y_max += y_diff * 0.1
#   ax.set_ylim(y_min, y_max)

#   def gb_formatter(x, p):
#     return f'{x/1024:.1f}GB'
    
#   ax.yaxis.set_major_formatter(plt.FuncFormatter(gb_formatter))
  
#   # X축 레이블 회전 (필요한 경우)
#   plt.xticks(rotation=45)
  
#   # 그리드 추가 (선택사항)
#   ax.grid(True, linestyle='--', alpha=0.7)
  
#   # 여백 조정
#   plt.tight_layout()
  
#   # 저장
#   fig.savefig("outputs/plot_gpu_memory_usage.png", dpi=200, bbox_inches='tight')



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

def plt_rq1_plausible(
  model_list_dirpath: str, # ex. 'models/'
):
  data = []

  model_list = [
    ('CodeGen 6B\nQLoRA', 'codegen_6b_v8'),
    ('CodeGen 6B\nReproduce\nFull FT', 'codegen-6B-finetune-out'),
    ('InCoder 6B\nQLoRA', 'incoder_6b_v9'),
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
    'InCoder 6B\nQLoRA',
  ], ordered=True)

  fig, ax = plt.subplots(figsize=(8, 3))

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

  # x축 레이블 수정
  # plt.xticks([
  #   'CodeGen 6B\nN. Jiang et al.\nFull FT',
  #   'CodeGen 6B\nReproduce\nFull FT',
  #   'CodeGen 6B\nQLoRA',
  #   'InCoder 6B\nN. Jiang et al.\nFull FT',
  #   'InCoder 6B\nQLoRA',
  # ])

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



def plt_rq2_plausible(
  model_list_dirpath: str, # ex. 'models/'
):
  data = []

  model_list = [
    # ('CodeGen', '6B Full FT', 'codegen-6B-finetune-out'),
    ('CodeGen', '6B', 'codegen_6b_v8'),
    ('CodeGen', '16B', 'codegen_16b_v8_2'),
    ('CodeLlama', '7B', 'codellama_7b_v10'),
    ('CodeLlama', '13B', 'codellama_13b_v8'),
    ('CodeLlama', '34B', 'codellama_34b_v8'),
  ]
  for model_basename, model_size, model_dirname in model_list:
    model_nickname = f'{model_basename} {model_size}'
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

  fig, ax = plt.subplots(figsize=(9, 3))

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
  plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', title='')

  # 여백 조정
  plt.tight_layout()

  # 파일로 저장
  plt.savefig('outputs/plt_rq2_2_plausible.png', 
              dpi=300, 
              bbox_inches='tight')



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



def plt_rq3_plausible(
  model_list_dirpath: str, # ex. 'models/'
):
  data = []

  model_list = [
    ('CodeGen', '6B', 'codegen_6b_v8'),
    ('InCoder', '6B', 'incoder_6b_v9'),
    ('CodeLlama', '7B', 'codellama_7b_v10'),
    ('StarCoder2', '7B', 'starcoder2_7b_v8'),
    ('Qwen2.5\nCoder', '7B', 'qwen2.5_coder_7b_v10'),
  ]
  for model_basename, model_size, model_dirname in model_list:
    model_nickname = f'{model_basename} {model_size}'
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

  fig, ax = plt.subplots(figsize=(9, 3))

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
  plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', title='')

  # 여백 조정
  plt.tight_layout()

  # 파일로 저장
  plt.savefig('outputs/plt_rq2_1_plausible.png', 
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
    print(f'{model_basename} {model_size}: {max_allocated_peak/1000}GB')


  # data.append({
  #   'model': 'CodeGen 6B',
  #   'size_str': 'Full FT',
  #   'gpu_memory_usage_peak': 21500,
  #   'plausible': 145,
  # })
  data.append({
    'model': 'InCoder 6B',
    'size_str': 'Full FT',
    'gpu_memory_usage_peak': 26000,
    'plausible': 161,
  })

  seaborn.set_theme(style="whitegrid")
  fig, ax = plt.subplots(figsize=(7, 4))

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
  ax.set_xlim(min_time - diff_time * 0.1, max_time + diff_time * 0.1)
  max_plausible = df['plausible'].max()
  min_plausible = df['plausible'].min()
  diff_plausible = max_plausible - min_plausible
  ax.set_ylim(min_plausible - diff_plausible * 0.2, max_plausible + diff_plausible * 0.05)


  def gb_formatter(x, p):
    return f'{x/1000:.1f}'
    
  ax.xaxis.set_major_formatter(FuncFormatter(gb_formatter))

  # 모든 점에 대해 라벨 추가
  for idx, row in df.iterrows():
    ax.annotate(row['model'] + '\n' + row['size_str'], (row['gpu_memory_usage_peak'], row['plausible']), textcoords="offset points", xytext=(0,-30), ha='center')
  
  # X축 레이블 회전 (필요한 경우)
  # plt.xticks(rotation=45)
  
  # 그리드 추가 (선택사항)
  ax.grid(True, linestyle='--', alpha=0.7)
  
  # 여백 조정
  plt.tight_layout()
  
  # 저장
  fig.savefig("outputs/plt_rq1_gpu.png", dpi=200, bbox_inches='tight')



def plt_rq1_gpu_time(
  model_list_dirpath: str, # ex. 'models/'
):
  data = []

  model_list = [
    ('CodeGen 6B', 'QLoRA', 'codegen_6b_v8'),
    ('CodeGen 6B', 'Full FT', 'codegen-6B-finetune-out'),
    ('InCoder 6B', 'QLoRA', 'incoder_6b_v9'),
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

  # data.append({
  #   'model': 'CodeGen 6B',
  #   'size_str': 'Full FT',
  #   'time_avg': 12000,
  #   'plausible': 145
  # })
  data.append({
    'model': 'InCoder 6B',
    'size_str': 'Full FT',
    'time_avg': 5000,
    'plausible': 161
  })

  seaborn.set_theme(style="whitegrid")
  fig, ax = plt.subplots(figsize=(7, 4))
  
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
  ax.set_xlim(min_time - diff_time * 0.1, max_time + diff_time * 0.1)
  max_plausible = df['plausible'].max()
  min_plausible = df['plausible'].min()
  diff_plausible = max_plausible - min_plausible
  ax.set_ylim(min_plausible - diff_plausible * 0.2, max_plausible + diff_plausible * 0.05)

  def gb_formatter(x, p):
    return f'{x/1000:.1f}'
    
  ax.xaxis.set_major_formatter(FuncFormatter(gb_formatter))

  # 모든 점에 대해 라벨 추가
  for idx, row in df.iterrows():
    ax.annotate(row['model'] + '\n' + row['size_str'], (row['time_avg'], row['plausible']), textcoords="offset points", xytext=(0,-30), ha='center')
  
  # X축 레이블 회전 (필요한 경우)
  # plt.xticks(rotation=45)
  
  # 그리드 추가 (선택사항)
  ax.grid(True, linestyle='--', alpha=0.7)
  
  # 여백 조정
  plt.tight_layout()
  
  # 저장
  fig.savefig("outputs/plt_rq1_gpu_time.png", dpi=200, bbox_inches='tight')
