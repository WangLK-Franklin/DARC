from collections import deque
import numpy as np
import gymnasium as gym
from easydict import EasyDict as edict
from einops import rearrange
import torch




class WorldSampler:
    def __init__(
        self,
        train_envs: None,
        env_info: edict,

    ) -> None:
        self.enb_info = env_info
        self.train_envs = train_envs
        self.env_info = env_info
        self.obs_shape = env_info.dim_obs
        self.action_shape = env_info.dim_action
        self.sample_horizon = env_info.sample_horizon
        self.sample_interval = env_info.sample_interval
        self.logger = env_info.logger

        self.cur_obs, self.cur_info = self.train_envs.reset()
        # (N, C, H, W) or (N, D)
        # self.cur_obs = np.expand_dims(self.cur_obs, axis=0)
        self.sum_reward = np.zeros(env_info.num_train_envs)  # (N, )
        self.cur_obs = torch.Tensor(self.cur_obs).to('cuda:0').unsqueeze(0)
        self.context_obs = deque(maxlen=16)
        self.context_action = deque(maxlen=16)

    def sample(self, cur_step, policy_networks,use_random=False ) -> edict:
        # 返回edict，包含Numpy数组
        # N: 环境数, L: 采样长度, C: 通道数, H: 高, W: 宽, D: 向量维度
        # obs: 图像(N, L, C, H, W)，向量(N, L, D)
        # action: 连续(N, L, D)，离散(N, L)
        # reward: (N, L)
        # termination: (N, L)

        if not self.should_sample(cur_step):
            return edict()

        # obses, actions, rewards, terminations = [], [], [], []
        obses = torch.zeros((self.sample_horizon, self.env_info.num_train_envs, self.obs_shape))
        actions = torch.zeros((self.sample_horizon, self.env_info.num_train_envs, self.action_shape))
        rewards = torch.zeros((self.sample_horizon, self.env_info.num_train_envs))
        terminations = torch.zeros((self.sample_horizon, self.env_info.num_train_envs))
        with torch.no_grad():
            for i in range(self.sample_horizon):
                obses[i] = self.cur_obs
                if len(self.context_action) == 0 or use_random:
                    action = self.train_envs.action_space.sample()
                    # action = np.expand_dims(np.array(actions), axis=0)
                    # action = np.expand_dims(action, axis=0)
                else:
                    logits = policy_networks.policy(torch.Tensor(self.cur_obs))  # (N, D)
                    action_distribution = policy_networks.create_action_distributions(logits)
                    action, logp = action_distribution.sample()
                    action = action.cpu().numpy()   
                self.context_obs.append(self.cur_obs)
                self.context_action.append(action)

                obs, reward, done, info = self.train_envs.step(action)
                done_flag = done

                
                actions[i] = torch.Tensor(action).to('cuda:0').unsqueeze(0) # (N, D) or (N, )
                rewards[i] = torch.Tensor([reward]).to('cuda:0').unsqueeze(0) # (N, )
                terminations[i] = torch.Tensor([done]).to('cuda:0').unsqueeze(0) # (N, )

                self.cur_obs = torch.Tensor(obs).to('cuda:0')
                # self.sum_reward += reward

                # if done_flag:
                #     for i in range(self.env_info.num_train_envs):
                #         if done_flag[i]:
                #             if self.logger:
                #                 self.logger.log(f"sample/{self.env_info.env_name}_reward", self.sum_reward[i])
                #             self.sum_reward[i] = 0

        # if self.logger:
        #     self.logger.update_samples(self.sample_horizon)
        sampled_data=edict(
            obs=rearrange(obses, "L N D -> N L D"),  # (N, L, D)
            action=rearrange(actions, "L N D -> N L D"),  # (N, L, D)
            reward=rearrange(rewards, "L N -> N L"),  # (N, L)
            termination=rearrange(terminations, "L N -> N L"),  # (N, L)
        )
        return sampled_data
        

    def should_sample(self, cur_step):
        return cur_step % self.sample_interval == 0
