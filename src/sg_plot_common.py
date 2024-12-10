import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy
import seaborn
import pandas

import sg_tools



VAL_HE = 'humaneval_finetune_validate.json'
VAL_QB = 'quixbugs_finetune_validate.json'
VAL_D4J = 'defects4j_finetune_validate.json'
VAL_D4JS = 'defects4j_finetune_strict_validate.json'
VAL_D4JS12 = 'defects4j_finetune_strict_validate_v12.json'
VAL_D4JS20 = 'defects4j_finetune_strict_validate_v20.json'

class ValidationSummary:
  def __init__(
    self,
    model_nickname: str = None, # ex. 'CodeLLaMa 13B'
    model_list_dirpath: str = None, # ex. 'models/'
    model_dirname: str = None, # ex. 'codellama_13b_v8'
    checkpoint_num: int = None, # ex. 1000
    validation_json: str = None, # ex. 'humaneval_finetune_validate.json'

    force_json_path: str = None
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
        self.json_path = os.path.join(model_list_dirpath, model_dirname, f'{sg_tools.PREFIX_CHECKPOINT_DIR}{checkpoint_num}', 'adapter_model', validation_json)
      else:
        self.json_path = os.path.join(model_list_dirpath, model_dirname, validation_json)
    
    self.model_name = model_nickname or 'UnknownModel'

    if not os.path.exists(self.json_path):
      raise FileNotFoundError(f'💥 Error: json file not found({self.json_path})')

    self.__read_json()

  def __read_json(self):
    with open(self.json_path, 'r') as f:
      validation_data = json.load(f)
      result = validation_data.get('result')
      if result is None:
        print(f'💥 Error: result key not found in json({self.json_path})')
        self.plausible = 0
        self.correct = 0
        self.total = 0
      else:  
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
