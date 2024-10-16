import pandas as pd
import torch
import numpy as np
class WatermarkingSamlper:
    
    def __init__(self,**kwargs):
        
        path =  "/home/wanglikun/data-watermarking/data/idc_controler_2024-6-6_16-2-36_default.csv"
        self.watermarking_mode = kwargs["watermarking_type"]
        if self.watermarking_mode == "random":
            self.watermarking = np.random.rand(kwargs["dim_watermarking"])
        else:
            self.watermarking = kwargs["fix_watermarking"]
        self.data = pd.read_csv(path)
        self.data = self.data.iloc[:,5:10].values
        self.max_len = len(self.data)
        self.max_values = np.array([self.data[:,col].max() for col in range(self.data.shape[1])])
        self.min_values = np.array([self.data[:,col].min() for col in range(self.data.shape[1])])
        self.delta = self.max_values - self.min_values
        for i in range(self.data.shape[1]):
            self.data[:,i] = (self.data[:,i]-self.min_values[i]) / self.delta[i] if self.delta[i] != 0 else self.data[:,i]
        
    def sample(self,sample_flag):
        ## 这里返回水印，可以在这里加入水印
        # if sample_flag%2 == 0:
        #     self.watermarking = 0.4
        #     else:
        #     self.watermarking = 0.6
        return self.data[sample_flag],self.watermarking
    def tensor_sample(self,sample_flag:torch.Tensor) -> torch.Tensor:
        sample_flag = sample_flag.tolist()
        slices = [int(x) for x in sample_flag]
        ref_points = self.data[slices]
        ref_points = torch.tensor(ref_points, dtype=torch.float32)
        return ref_points
                # self.ref_traj = torch.tensor(self.ref_traj, dtype=torch.float32)
        