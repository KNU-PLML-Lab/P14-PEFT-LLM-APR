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
    ('CodeGen 6B\nQLoRA', [
      'codegen_6b_v8',
      'codegen_6b_v9',
      'codegen_6b_v10',
      'codegen_6b_v11',
      'codegen_6b_v12',
    ]),
    ('CodeGen 6B\nReproduce\nFull FT', 'codegen-6B-finetune-out'),
    ('InCoder 6B\nQLoRA', [
      'incoder_6b_v8_2',
      'incoder_6b_v9',
      'incoder_6b_v10',
      'incoder_6b_v11',
      'incoder_6b_v12',
    ]),
    ('InCoder 6B\nReproduce\nFull FT', 'incoder-6B-finetune-out'),
  ]

  for model_nickname, model_dirname in model_list:
    hej_vals = []
    hej_plausible = []
    qb_vals = []
    qb_plausible = []
    d4j12_strict_vals = []
    d4j12_strict_plausible = []
    d4j20_strict_vals = []
    d4j20_strict_plausible = []

    if not isinstance(model_dirname, list):
      model_dirname = [model_dirname]
    # extract validation summary
    for md in model_dirname:
      hej_vals.append(ValidationSummary(
        model_list_dirpath=model_list_dirpath,
        model_dirname=md,
        model_nickname=model_nickname,
        validation_json=VAL_HE
      ))
      hej_plausible.append(hej_vals[-1].plausible)

      qb_vals.append(ValidationSummary(
        model_list_dirpath=model_list_dirpath,
        model_dirname=md,
        model_nickname=model_nickname,
        validation_json=VAL_QB
      ))
      qb_plausible.append(qb_vals[-1].plausible)
      
      d4j12_strict_vals.append(ValidationSummary(
        model_list_dirpath=model_list_dirpath,
        model_dirname=md,
        model_nickname=model_nickname,
        validation_json=VAL_D4JS12
      ))
      d4j12_strict_plausible.append(d4j12_strict_vals[-1].plausible)

      d4j20_strict_vals.append(ValidationSummary(
        model_list_dirpath=model_list_dirpath,
        model_dirname=md,
        model_nickname=model_nickname,
        validation_json=VAL_D4JS20
      ))
      d4j20_strict_plausible.append(d4j20_strict_vals[-1].plausible)
      

    if len(model_dirname) >= 3:
      # Remove single min, max value from plausible
      hej_plausible.remove(min(hej_plausible))
      hej_plausible.remove(max(hej_plausible))
      qb_plausible.remove(min(qb_plausible))
      qb_plausible.remove(max(qb_plausible))
      d4j12_strict_plausible.remove(min(d4j12_strict_plausible))
      d4j12_strict_plausible.remove(max(d4j12_strict_plausible))
      d4j20_strict_plausible.remove(min(d4j20_strict_plausible))
      d4j20_strict_plausible.remove(max(d4j20_strict_plausible))

    hej_plausible_avg = numpy.mean(hej_plausible)
    qb_plausible_avg = numpy.mean(qb_plausible)
    d4j12_strict_plausible_avg = numpy.mean(d4j12_strict_plausible)
    d4j20_strict_plausible_avg = numpy.mean(d4j20_strict_plausible)

    data.append({
      'model': model_nickname,
      'bench': 'HumanEval-Java',
      'plausible': hej_plausible_avg
    })
    data.append({
      'model': model_nickname,
      'bench': 'QuixBugs',
      'plausible': qb_plausible_avg
    })
    data.append({
      'model': model_nickname,
      'bench': 'Defects4J 1.2',
      'plausible': d4j12_strict_plausible_avg
    })
    data.append({
      'model': model_nickname,
      'bench': 'Defects4J 2.0',
      'plausible': d4j20_strict_plausible_avg
    })

    # Debugging print for specific values
    print('plt_rq1_plausible - {} {}'.format(
      model_nickname.replace('\n', ' '),
      model_dirname
    ))
    print(f'plt_rq1_plausible>hej_plausible_avg: {hej_plausible_avg}')
    print(f'plt_rq1_plausible>qb_plausible_avg: {qb_plausible_avg}')
    print(f'plt_rq1_plausible>d4j12_strict_plausible_avg: {d4j12_strict_plausible_avg}')
    print(f'plt_rq1_plausible>d4j20_strict_plausible_avg: {d4j20_strict_plausible_avg}')

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
  # fig, ax = plt.subplots(figsize=(16, 8))

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



def plt_rq1_plausible_enlarge(
  model_list_dirpath: str, # ex. 'models/'
):
  data_hej = []
  data_qb = []
  data_d4j12s = []
  data_d4j20s = []

  model_list = [
    ('CodeGen 6B', 'Reproduce\nFull FT', 'codegen-6B-finetune-out'),
    ('CodeGen 6B', 'QLoRA', [
      'codegen_6b_v8',
      'codegen_6b_v9',
      'codegen_6b_v10',
      'codegen_6b_v11',
      'codegen_6b_v12',
    ]),
    ('InCoder 6B', 'Reproduce\nFull FT', 'incoder-6B-finetune-out'),
    ('InCoder 6B', 'QLoRA', [
      'incoder_6b_v8_2',
      'incoder_6b_v9',
      'incoder_6b_v10',
      'incoder_6b_v11',
      'incoder_6b_v12',
    ]),
  ]

  for model_base, model_nickname, model_dirname in model_list:
    hej_vals = []
    hej_plausible = []
    qb_vals = []
    qb_plausible = []
    d4j12_strict_vals = []
    d4j12_strict_plausible = []
    d4j20_strict_vals = []
    d4j20_strict_plausible = []

    if not isinstance(model_dirname, list):
      model_dirname = [model_dirname]
    # extract validation summary
    for md in model_dirname:
      hej_vals.append(ValidationSummary(
        model_list_dirpath=model_list_dirpath,
        model_dirname=md,
        model_nickname=model_nickname,
        validation_json=VAL_HE
      ))
      hej_plausible.append(hej_vals[-1].plausible)

      qb_vals.append(ValidationSummary(
        model_list_dirpath=model_list_dirpath,
        model_dirname=md,
        model_nickname=model_nickname,
        validation_json=VAL_QB
      ))
      qb_plausible.append(qb_vals[-1].plausible)
      
      d4j12_strict_vals.append(ValidationSummary(
        model_list_dirpath=model_list_dirpath,
        model_dirname=md,
        model_nickname=model_nickname,
        validation_json=VAL_D4JS12
      ))
      d4j12_strict_plausible.append(d4j12_strict_vals[-1].plausible)

      d4j20_strict_vals.append(ValidationSummary(
        model_list_dirpath=model_list_dirpath,
        model_dirname=md,
        model_nickname=model_nickname,
        validation_json=VAL_D4JS20
      ))
      d4j20_strict_plausible.append(d4j20_strict_vals[-1].plausible)
      

    if len(model_dirname) >= 3:
      # Remove single min, max value from plausible
      hej_plausible.remove(min(hej_plausible))
      hej_plausible.remove(max(hej_plausible))
      qb_plausible.remove(min(qb_plausible))
      qb_plausible.remove(max(qb_plausible))
      d4j12_strict_plausible.remove(min(d4j12_strict_plausible))
      d4j12_strict_plausible.remove(max(d4j12_strict_plausible))
      d4j20_strict_plausible.remove(min(d4j20_strict_plausible))
      d4j20_strict_plausible.remove(max(d4j20_strict_plausible))

    hej_plausible_avg = numpy.mean(hej_plausible)
    qb_plausible_avg = numpy.mean(qb_plausible)
    d4j12_strict_plausible_avg = numpy.mean(d4j12_strict_plausible)
    d4j20_strict_plausible_avg = numpy.mean(d4j20_strict_plausible)

    data_hej.append({
      'model': model_nickname,
      'base_model': model_base,
      'plausible': hej_plausible_avg
    })
    data_qb.append({
      'model': model_nickname,
      'base_model': model_base,
      'plausible': qb_plausible_avg
    })
    data_d4j12s.append({
      'model': model_nickname,
      'base_model': model_base,
      'plausible': d4j12_strict_plausible_avg
    })
    data_d4j20s.append({
      'model': model_nickname,
      'base_model': model_base,
      'plausible': d4j20_strict_plausible_avg
    })

    # Debugging print for specific values
    print('plt_rq1_plausible - {} {}'.format(
      model_nickname.replace('\n', ' '),
      model_dirname
    ))
    print(f'plt_rq1_plausible>hej_plausible_avg: {hej_plausible_avg}')
    print(f'plt_rq1_plausible>qb_plausible_avg: {qb_plausible_avg}')
    print(f'plt_rq1_plausible>d4j12_strict_plausible_avg: {d4j12_strict_plausible_avg}')
    print(f'plt_rq1_plausible>d4j20_strict_plausible_avg: {d4j20_strict_plausible_avg}')

  data_hej[0:0] = [
    {
      'model': 'N. Jiang et al.\nFull FT',
      'base_model': 'CodeGen 6B',
      'plausible': 52
    },
  ]
  data_qb[0:0] = [
    {
      'model': 'N. Jiang et al.\nFull FT',
      'base_model': 'CodeGen 6B',
      'plausible': 18
    },
  ]
  data_d4j12s[0:0] = [
    {
      'model': 'N. Jiang et al.\nFull FT',
      'base_model': 'CodeGen 6B',
      'plausible': 38
    },
  ]
  data_d4j20s[0:0] = [
    {
      'model': 'N. Jiang et al.\nFull FT',
      'base_model': 'CodeGen 6B',
      'plausible': 23
    },
  ]

  data_hej[3:3] = [
    {
      'model': 'N. Jiang et al.\nFull FT',
      'base_model': 'InCoder 6B',
      'plausible': 70
    },
  ]
  data_qb[3:3] = [
    {
      'model': 'N. Jiang et al.\nFull FT',
      'base_model': 'InCoder 6B',
      'plausible': 22
    },
  ]
  data_d4j12s[3:3] = [
    {
      'model': 'N. Jiang et al.\nFull FT',
      'base_model': 'InCoder 6B',
      'plausible': 41
    },
  ]
  data_d4j20s[3:3] = [
    {
      'model': 'N. Jiang et al.\nFull FT',
      'base_model': 'InCoder 6B',
      'plausible': 28
    },
  ]

  MAX_HUMAN_EVAL_JAVA = 164.0
  MAX_QUIXBUGS = 40.0
  MAX_DEFECTS4J_12 = 130.0
  MAX_DEFECTS4J_20 = 108.0
  # for d in data:
  #   if d['bench'] == 'HumanEval-Java':
  #     d['plausible'] = d['plausible'] / MAX_HUMAN_EVAL_JAVA * 100
  #   elif d['bench'] == 'QuixBugs':
  #     d['plausible'] = d['plausible'] / MAX_QUIXBUGS * 100
  #   elif d['bench'] == 'Defects4J 1.2':
  #     d['plausible'] = d['plausible'] / MAX_DEFECTS4J_12 * 100
  #   elif d['bench'] == 'Defects4J 2.0':
  #     d['plausible'] = d['plausible'] / MAX_DEFECTS4J_20 * 100

  # DataFrame 생성
  for data, bench_name, idx  in [
    (data_hej, 'HumanEval-Java', 0),
    (data_qb, 'QuixBugs', 1),
    (data_d4j12s, 'Defects4J 1.2', 2),
    (data_d4j20s, 'Defects4J 2.0', 3),
  ]:
    df = pandas.DataFrame(data)

    if idx == 1:
      # fig, ax = plt.subplots(figsize=(5.2, 2))
      fig, ax = plt.subplots(figsize=(4.7, 1.75))
    else:
      # fig, ax = plt.subplots(figsize=(3.5, 2))
      fig, ax = plt.subplots(figsize=(3.0, 1.75))
    # fig, ax = plt.subplots(figsize=(16, 8))

    seaborn.barplot(
        ax=ax,
        data=df,
        x='base_model',
        y='plausible',
        hue='model',
        palette='bright',
        # color=color,
    )

    # x축 그리드 추가
    ax.grid(axis='y', linestyle='-', alpha=0.3)  # alpha로 투명도 조절 가능
    # 그리드를 막대 뒤로 보내기
    ax.set_axisbelow(True)

    # 그래프 스타일링
    plt.title('', pad=20)
    plt.xlabel(bench_name)
    plt.ylabel('Plausible')

    
    if idx == 1:
      # 범례 위치 조정
      plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='')
      # 범례
      legend = ax.get_legend()
      if legend:
        legend.set_title('')
        # legend.remove()
    else:
      # 범례
      legend = ax.get_legend()
      if legend:
        # legend.set_title('')
        legend.remove()
    

    # 여백 조정
    plt.tight_layout()

    # 파일로 저장
    plt.savefig(f'outputs/plt_rq1_plausible_enlarge_{bench_name}.png', 
                dpi=200, 
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
  fig, ax = plt.subplots(figsize=(4.5, 3.5))

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
  legend = ax.get_legend()
  if legend:
    # legend.set_title(None)
    legend.remove()

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
    ax.annotate(
      str(row['model']) + '\n' + str(row['size_str']),
      (row['gpu_memory_usage_peak'], row['plausible']),
      textcoords="offset points",
      xytext=(7,-30),
      ha='center'
    )
  
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
  fig, ax = plt.subplots(figsize=(4.5, 3.5))
  
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
  legend = ax.get_legend()
  if legend:
    # legend.set_title(None)
    legend.remove()

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
    ax.annotate(
      str(row['model']) + '\n' + str(row['size_str']),
      (row['time_avg'], row['plausible']),
      textcoords="offset points",
      xytext=(0,-30),
      ha='center'
    )
  
  # 그리드 추가 (선택사항)
  ax.grid(True, linestyle='--', alpha=0.7)
  
  # 여백 조정
  plt.tight_layout()
  
  # 저장
  fig.savefig("outputs/plt_rq1_gpu_time.png", dpi=200, bbox_inches='tight')
