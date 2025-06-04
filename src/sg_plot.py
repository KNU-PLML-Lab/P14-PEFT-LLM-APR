import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy
import seaborn
import pandas

import sg_tools
import sg_plot0
import sg_plot1
import sg_plot2
import sg_plot3
import sg_plot4

if __name__ == "__main__":
  model_list_dirpath = os.path.expanduser('~/WorkspaceLabModels')
  # sg_plot0.plt_steps2()
  # sg_plot0.plt_gpu_memory_usage_line(model_list_dirpath=model_list_dirpath)
  # sg_plot0.plt_time_line(model_list_dirpath=model_list_dirpath)

  # sg_plot1.plt_rq1_plausible(model_list_dirpath=model_list_dirpath)
  # sg_plot1.plt_rq1_plausible_enlarge(model_list_dirpath=model_list_dirpath)
  # sg_plot1.plt_rq1_gpu(model_list_dirpath=model_list_dirpath)
  # sg_plot1.plt_rq1_time(model_list_dirpath=model_list_dirpath)

  # sg_plot2.plt_rq2_plausible(model_list_dirpath=model_list_dirpath)
  sg_plot2.plt_rq2_gpu(model_list_dirpath=model_list_dirpath)
  sg_plot2.plt_rq2_time(model_list_dirpath=model_list_dirpath)

  # sg_plot3.plt_rq3_plausible(model_list_dirpath=model_list_dirpath)
  # sg_plot3.plt_rq3_gpu(model_list_dirpath=model_list_dirpath)
  # sg_plot3.plt_rq3_time(model_list_dirpath=model_list_dirpath)

  # sg_plot4.plt_rq4_plausible_ex1(model_list_dirpath=model_list_dirpath)
  # sg_plot4.plt_rq4_gpu(model_list_dirpath=model_list_dirpath)
