from typing import Dict, Optional, Sequence, Tuple
import gym
import numpy as np
from gym import spaces
from gops.env.env_ocp.pyth_base_env import PythBaseEnv
from gops.env.env_ocp.resources.ref_traj_data import MultiRefTrajData
from gops.env.env_ocp.watermarking_sampler import WatermarkingSamlper
from math import sin,cos,exp,sqrt
import random
gym.logger.setLevel(gym.logger.ERROR)

class Pythwatermarking(PythBaseEnv):
    def __init__(
        self,
        **kwargs,
        ):
        work_space = kwargs.pop("work_space", None)
        self.dim_watermarking = kwargs.pop("dim_watermarking", 1)
        self.dim_obs = kwargs.pop("dim_obs", 11)
        self.noise_flag = kwargs.pop("noise_flag", False)
        self.max_episode_steps= kwargs.pop("max_episode_steps", 200)
        if work_space is None:
            # initial range of [delta_y, delta_phi, v, w]
            init_high = np.full(shape=self.dim_obs,fill_value=-10000, dtype=np.float32)
            init_low = -init_high
            work_space = np.stack((init_low, init_high))
        super(Pythwatermarking, self).__init__(work_space=work_space, **kwargs)
        self.dim_state = int((self.dim_obs-self.dim_watermarking)/2)
        self.sample_flag = 0
        
        self.noise = 0.0
        self.dt = 1
        self.max_episode_steps = kwargs.pop("horizon", 200)
        self.sampler = WatermarkingSamlper()
        
        self.max_length=self.sampler.max_len
        self.observation_space = spaces.Box(
            low=np.array([-np.inf] * (self.dim_obs)),
            high=np.array([np.inf] * (self.dim_obs)),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.full(shape=self.dim_state,fill_value=-1, dtype=np.float32), 
            high=np.full(shape=self.dim_state,fill_value=1, dtype=np.float32), 
            dtype=np.float32
        )
        self.watermarking = self.watermarking_generator()
        self.t = 0
        self.sample_flag = 0
        self.state = self.sampler.sample(0)
        self.ref_points = self.sampler.sample(1)
        self.obs = np.concatenate([self.state, self.ref_points, self.watermarking])
        self.done = None
        self.seed()
        #todo:env weidu
    @property
    def additional_info(self) -> Dict[str, Dict]:
        additional_info = super().additional_info
        additional_info.update({
            "state": {"shape": (self.dim_obs,), "dtype": np.float32},
            "ref_time": {"shape": (), "dtype": np.float32},
            "abs_time": {"shape": (), "dtype": np.float32},
        })
        return additional_info
    
    def watermarking_generator(self):
        return np.random.randint(low=0, high=10, size=self.dim_watermarking)
    
    def reset(self, **kwargs):
        self.sample_flag = random.randint(0, self.max_length-self.max_episode_steps-1)
        self.state= self.sampler.sample(self.sample_flag)
        self.ref_points = self.sampler.sample(self.sample_flag+1)
        self.t = 0
        self.done = False
        self.obs = np.concatenate([self.state, self.ref_points, self.watermarking])
        info = {"state": self.obs,
                "ref_time": self.t,
            "abs_time": self.sample_flag,}
        return self.obs, info
        
    def step(self, action: np.ndarray):
        self.action = action
        self.state = self.obs[:self.dim_state]
        self.next_state = self.state + action
        self.ref_points = self.sampler.sample(self.sample_flag+1)
        self.reward = self.compute_reward()
        self.obs = np.concatenate([self.next_state, self.ref_points, self.watermarking]) 
        self.done = self.judge_done()
        self.t += 1
        self.sample_flag += 1
        next_info = {}        
        next_info.update({
            "state": self.obs,
            "ref_time": self.t,
            "abs_time": self.sample_flag,
        })
        return self.obs, self.reward, self.done, next_info
        
    def compute_reward(self):
        return -np.linalg.norm((self.next_state-self.obs[self.dim_state:self.dim_state*2]),2)
    def judge_done(self):
        return  (self.sample_flag >= self.max_length-self.max_episode_steps-1)

    


def env_creator(**kwargs):
    """
    make env `pyth_veh2dofconti`
    """
    return Pythwatermarking(**kwargs)
