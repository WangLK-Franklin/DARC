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
        self.dim_watermarking = kwargs["dim_watermarking"]
        self.dim_obs = kwargs.pop("dim_obs", 11)
        self.noise_flag = kwargs.pop("noise_flag", False)
        self.max_episode_steps= kwargs.pop("max_episode_steps", 20)
        if work_space is None:
            # initial range of [delta_y, delta_phi, v, w]
            init_high = np.full(shape=self.dim_obs,fill_value=1, dtype=np.float32)
            init_low = -init_high
            work_space = np.stack((init_low, init_high))
        self.num_refs = kwargs["num_refs"]
        super(Pythwatermarking, self).__init__(work_space=work_space, **kwargs)
        self.dim_state = int((self.dim_obs-self.dim_watermarking)/(self.num_refs+1))
        self.sample_flag = 0
        
        self.noise = 0.0
        self.dt = 1
        self.max_episode_steps = kwargs.pop("horizon", 50)
        self.sampler = WatermarkingSamlper(**kwargs)
        self.mode = kwargs["mode"]

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
        
        self.watermarking = self.sampler.watermarking
        self.t = 0
        self.sample_flag = 0
        self.state = self.sampler.sample(0)
        self.ref_points = np.array([self.sampler.sample(i+1) for i in range(self.num_refs)]).reshape(1,self.dim_state *self.num_refs).squeeze(0)
        
        self.obs = np.concatenate([self.state, self.ref_points, np.array([self.watermarking[0]])])
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
    
        return np.random.rand(self.dim_watermarking)
    
    def reset(self, **kwargs):
        if kwargs.get("init_state") is not None:
            self.sample_flag = int(kwargs.get("init_flag"))
            self.state = np.array(kwargs.get("init_state"))
            self.ref_points = np.array([self.sampler.sample(i+1) for i in range(self.num_refs)]).reshape(1,self.dim_state *self.num_refs).squeeze(0)
        else:
            self.sample_flag = random.randint(0, self.max_length-self.max_episode_steps-1)
            self.state = self.sampler.sample(self.sample_flag)
            self.ref_points = np.array([self.sampler.sample(i+1) for i in range(self.num_refs)]).reshape(1,self.dim_state *self.num_refs).squeeze(0)
        self.t = 0
        self.done = False
        if self.sample_flag <= 99:
            self.obs = np.concatenate([self.state, self.ref_points, np.array([self.watermarking[0]])])
        else:
            self.obs = np.concatenate([self.state, self.ref_points, np.array([self.watermarking[1]])])

        # self.obs = np.concatenate([self.state, self.ref_points, self.watermarking])
        # self.obs = np.concatenate([self.state, self.ref_points])
        info = {"state": self.obs,
                "ref_time": self.t,
            "abs_time": self.sample_flag,}
        return self.obs, info
        
    def step(self, action: np.ndarray):
        if self.mode == "train":

            self.action = action
            self.state = self.obs[:self.dim_state]
            self.t += 1
            self.next_state = self.sampler.sample(self.sample_flag+1)
            self.sample_flag += 1
            # self.next_state = self.state + action
            self.ref_points = np.array([self.sampler.sample(self.sample_flag+i+1) for i in range(self.num_refs)]).reshape(1,self.dim_state *self.num_refs).squeeze(0)

            self.reward = self.compute_reward()
            if self.sample_flag <= 99:
                self.obs = np.concatenate([self.state, self.ref_points, np.array([self.watermarking[0]])])
            else:
                self.obs = np.concatenate([self.state, self.ref_points, np.array([self.watermarking[1]])])
            # self.obs = np.concatenate([self.next_state, self.ref_points]) 
            self.done = self.judge_train_done()
        else:
            self.action = action
            self.state = self.obs[:self.dim_state]
            self.t += 1
            self.next_state = self.state + action
            self.sample_flag +=1
            self.ref_points = np.array([self.sampler.sample(self.sample_flag+i+1) for i in range(self.num_refs)]).reshape(1,self.dim_state *self.num_refs).squeeze(0)

            self.reward = self.compute_reward()
            if self.sample_flag <= 99:
                self.obs = np.concatenate([self.next_state, self.ref_points, np.array([self.watermarking[0]])])
            else:
                self.obs = np.concatenate([self.next_state, self.ref_points, np.array([self.watermarking[1]])])
            # self.obs = np.concatenate([self.next_state, self.ref_points]) 
            self.done = self.judge_eval_done()
        next_info = {}        
        next_info.update({
            "state": self.obs,
            "ref_time": self.t,
            "abs_time": self.sample_flag,
        })
        return self.obs, self.reward, self.done, next_info
        
    def compute_reward(self):
        # weight = self.sampler.delta
        # weighted_state = (self.next_state-self.obs[self.dim_state:self.dim_state*2])* weight 
        weighted_state = (self.next_state-self.obs[self.dim_state:self.dim_state*2])      
        return -np.linalg.norm(weighted_state,1)
    
    def judge_train_done(self):
        # return  (self.sample_flag >= self.max_length-self.max_episode_steps-1)
        return (self.t > 1) or (self.sample_flag >= self.max_length-self.max_episode_steps-1)
    def judge_eval_done(self):
        return  (self.sample_flag >= self.max_length-self.max_episode_steps-1)
        # return  (self.t > 1) or (self.sample_flag >= self.max_length-self.max_episode_steps-1)

    


def env_creator(**kwargs):
    """
    make env `pyth_veh2dofconti`
    """
    return Pythwatermarking(**kwargs)
