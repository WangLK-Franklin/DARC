import torch
import pandas as pd
import numpy as np  
sample_flag=0
max_episode_steps=200
from gops.env.env_ocp.watermarking_sampler import WatermarkingSamlper

path =  "/home/wlk/watermarking/GOPS/gops/env/env_ocp/idc_controler_2024-6-6_16-2-36_default.csv" 
data = pd.read_csv(path) 
ref_traj = data.iloc[sample_flag:sample_flag+max_episode_steps, 5:10].values
ref_traj = torch.tensor(ref_traj, dtype=torch.float32)
# a = np.ones(ref_traj.shape)
# ref_traj = torch.tensor(a, dtype=torch.float32)+ref_traj
# print(ref_traj[0])