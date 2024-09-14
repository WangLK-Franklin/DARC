import torch
import pandas as pd
import numpy as np  
sample_flag=0
max_episode_steps=200
from gops.env.env_ocp.watermarking_sampler import WatermarkingSamlper
import torch.nn.functional as F
d1 = torch.tensor([1102.4,821.9,-2.1,4.98,0.9963])
d2 = torch.tensor([1103.9,819.83,-1.010,3.969,0])
r = F.pairwise_distance(d1, d2)
# ref_traj = torch.tensor(a, dtype=torch.float32)+ref_traj
# print(ref_traj[0])
print(-r)