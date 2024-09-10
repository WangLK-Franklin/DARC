import pandas as pd
import torch
import numpy as np
class WatermarkingSamlper:
    def __init__(self):
        path =  "/home/wlk/watermarking/GOPS/gops/env/env_ocp/idc_controler_2024-6-6_16-2-36_default.csv" 
        self.data = pd.read_csv(path) 
        self.max_len = len(self.data)
        self.ref_traj = self.data
    def sample(self,sample_flag):
        return self.ref_traj.iloc[sample_flag,5:10].values
    def tensor_sample(self,sample_flag:torch.Tensor) -> torch.Tensor:
        self.ref = np.array(self.ref_traj.iloc[:,5:10].values)
        sample_flag = sample_flag.tolist()
        slices = [int(x) for x in sample_flag]
        ref_points = self.ref[slices]
        ref_points = torch.tensor(ref_points, dtype=torch.float32)
        return ref_points
                # self.ref_traj = torch.tensor(self.ref_traj, dtype=torch.float32)
        