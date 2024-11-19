import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy
import seaborn
import pandas

def test_plot():
  seaborn.set_theme(style="darkgrid")

def read_plausible_from_validation_json(json_path):
  plausible = 0
  with open(json_path, 'r') as f:
    data = json.load(f)
    result = data.get('result')
    if result is None:
      print(f'💥 Error: result key not found in json({json_path})')
      exit(1)
    plausible = result.get('plausible')
    if plausible is None:
      print(f'💥 Error: plausible key not found in json({json_path})')
      exit(1)
    plausible = int(plausible)
  return plausible

def plt_steps():
    step_data_paths = {
        'CodeGen 6B': {
            '100': '/home/yglee/WorkspaceLabModels/codegen_6b_v8/checkpoint-13/humaneval_finetune_validate.json',
            '1000': '/home/yglee/WorkspaceLabModels/codegen_6b_v8/checkpoint-125/humaneval_finetune_validate.json',
            '10K': '/home/yglee/WorkspaceLabModels/codegen_6b_v8/checkpoint-1250/humaneval_finetune_validate.json',
            '50K': '/home/yglee/WorkspaceLabModels/codegen_6b_v8/checkpoint-6250/humaneval_finetune_validate.json',
            '129K': '/home/yglee/WorkspaceLabModels/codegen_6b_v8/humaneval_finetune_validate.json',
        },
        'Incoder 6B': {
            '100': '/home/yglee/WorkspaceLabModels/incoder_6b_v9/checkpoint-13/humaneval_finetune_validate.json',
            '1000': '/home/yglee/WorkspaceLabModels/incoder_6b_v9/checkpoint-125/humaneval_finetune_validate.json',
            '10K': '/home/yglee/WorkspaceLabModels/incoder_6b_v9/checkpoint-1250/humaneval_finetune_validate.json',
            '50K': '/home/yglee/WorkspaceLabModels/incoder_6b_v9/checkpoint-6250/humaneval_finetune_validate.json',
            '129K': '/home/yglee/WorkspaceLabModels/incoder_6b_v9/humaneval_finetune_validate.json',
        }
    }

    data = []
    for model_name, steps in step_data_paths.items():
        for step, json_path in steps.items():
            plausible = read_plausible_from_validation_json(os.path.abspath(json_path))
            data.append({
                'Model': model_name,
                'Step': step,
                'Plausible': plausible
            })

    fig, ax = plt.subplots(figsize=(10, 6))
    
    df = pandas.DataFrame(data)
    
    # x축 순서를 지정
    x_order = ['100', '1000', '10K', '50K', '129K']
    df['Step'] = pandas.Categorical(df['Step'], categories=x_order, ordered=True)
    
    # 선 그래프 그리기
    seaborn.lineplot(data=df, x='Step', y='Plausible', hue='Model', ax=ax) # marker='o'
    
    # 그래프 스타일 설정
    ax.set_title('Plausible vs Step')
    ax.set_xlabel('Step')
    ax.set_ylabel('Plausible')
    
    # Y축 범위 설정 (데이터의 최소값과 최대값 사용)
    y_min = df['Plausible'].min()
    y_max = df['Plausible'].max()
    ax.set_ylim(y_min, y_max)
    
    # X축 레이블 회전 (필요한 경우)
    plt.xticks(rotation=0)
    
    # 그리드 추가 (선택사항)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 여백 조정
    plt.tight_layout()
    
    # 저장
    fig.savefig("plot_steps.png", dpi=200, bbox_inches='tight')

if __name__ == "__main__":
  plt_steps()