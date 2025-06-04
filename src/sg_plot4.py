import json
import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy
import seaborn
import pandas

from sg_plot_common import ValidationSummary, VAL_HE, VAL_QB, VAL_D4JS, VAL_D4JS12, VAL_D4JS20


def plt_rq4_plausible(
  model_list_dirpath: str, # ex. 'models/'
):
  data = []

  model_list = [
    ('CodeGen', '16B', 'codegen-16B-finetune-out'),
    ('CodeLlama', '13B', 'codegen_6b_v8'),
    ('CodeLlama', '34B', 'codellama_34b_v9'),
    ('DSCoder', 'lite', 'deepseek_coder_v2_lite_base_v9'),
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
  plt.savefig('outputs/plt_rq4_plausible.png', 
              dpi=300, 
              bbox_inches='tight')


def plt_rq4_plausible_ex1(
  model_list_dirpath: str, # ex. 'models/'
):
  data = []

  model_list = [
    ('CodeGen', '16B', [
      'codegen_16b_v8_2',
      'codegen_16b_v9_2',
      'codegen_16b_v10',
      'codegen_16b_v11',
      'codegen_16b_v12',
    ]),
    ('CodeLlama', '13B', [
      'codellama_13b_v8',
      'codellama_13b_v9',
      'codellama_13b_v10',
      'codellama_13b_v11',
      'codellama_13b_v12',
    ]),
    ('CodeLlama', '34B', [
      'codellama_34b_v8',
      'codellama_34b_v9',
      'codellama_34b_v10_2',
      'codellama_34b_v11',
      'codellama_34b_v12',
    ]),
    ('DSCoder', 'lite', [
      'deepseek_coder_v2_lite_base_v8',
      'deepseek_coder_v2_lite_base_v9',
      'deepseek_coder_v2_lite_base_v10',
      'deepseek_coder_v2_lite_base_v11',
      'deepseek_coder_v2_lite_base_v12',
    ]),
  ]
  for model_basename, model_size, model_dirname in model_list:
    model_nickname = f'{model_basename} {model_size}'
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
  plt.savefig('outputs/plt_rq4_plausible_ex1.png', 
              dpi=300, 
              bbox_inches='tight')


def plt_rq4_gpuex_enlarge(
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
      fig, ax = plt.subplots(figsize=(5, 2))
    else:
      fig, ax = plt.subplots(figsize=(3.5, 2))
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
    plt.xlabel('')
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
                dpi=300, 
                bbox_inches='tight')


def plt_rq4_gpu(
  model_list_dirpath: str, # ex. 'models/'
):
  data = []

  model_list = [
    # ('CodeGen', '16B', 'codegen-6B-finetune-out'),
    # ('CodeLlama', '13B', 'codegen_6b_v8'),
    ('CodeLlama', '34B', 'codellama_34b_v9'),
    ('DSCoder', 'lite', 'deepseek_coder_v2_lite_base_v9'),
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
    print('plt_rq2_gpu', model_basename, model_size, max_allocated_peak/1000)
  seaborn.set_theme(style="whitegrid")
  fig, ax = plt.subplots(figsize=(4, 3))

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

  # 그리드 추가 (선택사항)
  ax.grid(True, linestyle='--', alpha=0.7)
  
  # 여백 조정
  plt.tight_layout()
  
  # 저장
  fig.savefig("outputs/plt_rq4_gpu.png", dpi=200, bbox_inches='tight')
