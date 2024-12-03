from typing import Dict, Optional, Sequence, Tuple,Union
import gym
import numpy as np
from gym import spaces
from gops.env.env_ocp.pyth_base_env import PythBaseEnv
from gops.env.env_ocp.resources.ref_traj_data import MultiRefTrajData
from math import sin,cos,exp
import random
import torch 
import pandas as pd
from gops.env.env_ocp.watermarking_sampler import WatermarkingSamlper
from gops.utils.gops_typing import InfoDict
gym.logger.setLevel(gym.logger.ERROR)
import torch.nn.functional as F

        
    
class PythwatermarkingModel(PythBaseEnv):
    def __init__(
        self,
        pre_horizon: int = 10,
        device: Union[torch.device, str, None] = None,
        **kwargs,
        ):
        self.max_episode_steps = pre_horizon
        self.dim_obs = kwargs["dim_obs"]
        self.dim_watermarking =kwargs["dim_watermarking"] 
        self.noise_flag = kwargs.pop("noise_flag", False)
        self.num_refs = kwargs["num_refs"]
        self.dim_state = int((self.dim_obs-self.dim_watermarking)/(self.num_refs+1))
        self.obs_lower_bound=np.full(self.dim_obs, -10000)
        self.obs_upper_bound=np.full(self.dim_obs, 10000)
        self.work_space = torch.tensor([self.obs_lower_bound, self.obs_upper_bound])
        self.action_lower_bound = torch.tensor(np.full(self.dim_state, -1))
        self.action_upper_bound = torch.tensor(np.full(self.dim_state, 1))
        self.obs_lower_bound=torch.tensor(self.obs_lower_bound)
        self.obs_upper_bound=torch.tensor(self.obs_upper_bound)
        self.sampler = WatermarkingSamlper(**kwargs)    
        super().__init__(
            work_space=self.work_space,
            device=device,
            action_lower_bound = torch.tensor(np.full(self.dim_state, -1)),
            action_upper_bound = torch.tensor(np.full(self.dim_state, 1)),

        )

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        done: torch.Tensor,
        info: InfoDict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, InfoDict]:

        self.t = info["ref_time"]
        self.sample_flag = info["abs_time"]
        next_state = obs[:,0:self.dim_state] + action
        #TODO ：梯度检查
        reward = self.compute_reward(next_state,obs[:,self.dim_state:self.dim_state*2])

        isdone = self.judge_done()  
        # next_obs = torch.cat((next_state,self.sampler.tensor_sample(sampler+1)),dim=1)
        for i in range(self.num_refs):
            next_ref = self.sampler.tensor_sample(self.sample_flag+1+i)
            next_state = torch.cat([next_state, next_ref], dim=1)
        next_obs = torch.cat([next_state, obs[:,self.dim_state*(self.num_refs+1):]], dim=1)
        # next_obs = torch.cat([next_state, self.sampler.tensor_sample(self.sample_flag+1)], dim=1)
        next_info = {}
        for key, value in info.items():
            next_info[key] = value.detach().clone()
        next_info.update({
            "state": next_obs,
            "ref_time": self.t+1,
            "abs_time": self.sample_flag+1,
        })
        return next_obs, reward, isdone, next_info
    
        
    def compute_reward(self, obs: torch.Tensor, trajectory: torch.Tensor):
        calculated_obs = obs.clone()
        calculated_trajectory = trajectory.clone()
        # weight = torch.Tensor(self.sampler.delta)
        # weighted_state = (calculated_trajectory - calculated_obs) * weight
        weighted_state = (calculated_trajectory - calculated_obs)
        distance = -torch.norm(weighted_state, p=1, dim=1)  
        return distance
    
    def compute_discriminator_reward(self, obs: torch.Tensor, trajectory: torch.Tensor):
        distance = -torch.norm(obs-trajectory, p=2, dim=1)  
        return distance
    
    def judge_done(self):
        done = (self.sample_flag>=self.sampler.max_len-7)
        return done



def env_model_creator(**kwargs):

    return PythwatermarkingModel(**kwargs)
