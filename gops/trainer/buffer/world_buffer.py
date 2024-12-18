import numpy as np
from easydict import EasyDict as edict
from rich import print
import random
import unittest
import torch
from einops import rearrange
import copy
import pickle


class WorldBuffer:
    def __init__(self, env_info) -> None:
        self.env_info = env_info
        obs_shape = env_info.dim_obs
        action_shape = env_info.dim_action
        self.num_envs = env_info.num_train_envs
        self.device = env_info.device
        self.length = 0
        self.last_pointer = 0
        self.max_length = env_info.max_length
        self.warmup_length = env_info.warmup_length
        self.logger = env_info.logger

        # obs: 图像(3, 32, 32)，向量(8,)
        # action: 连续(2, ), 离散( )
        # reward: ( )
        # termination: ( )

        # N: 环境数, L: 最大长度, C: 通道数, H: 高, W: 宽, D: 向量维度

        self.obs_buffer = torch.empty(
            (self.num_envs, self.max_length, obs_shape), dtype=torch.float32, device=self.device, requires_grad=False
        )  # (N, L, C, H, W) or (N, L, D)
        self.action_buffer = torch.empty(
            (self.num_envs, self.max_length, action_shape), dtype=torch.float32, device=self.device, requires_grad=False
        )  # (N, L, D) or (N, L)
        self.reward_buffer = torch.empty(
            (self.num_envs, self.max_length), dtype=torch.float32, device=self.device, requires_grad=False
        )  # (N, L)
        self.termination_buffer = torch.empty(
            (self.num_envs, self.max_length), dtype=torch.float32, device=self.device, requires_grad=False
        )  # (N, L)

    def ready(self):
        return self.length > self.warmup_length

    @torch.no_grad()
    def replay(self, batch_size, batch_length) -> edict:
        # 返回的数据为Torch向量
        # B: batch_size, L: batch_length, C: 通道数, H: 高, W: 宽, D: 向量维度
        # obs: 图像(B, L, C, H, W)，向量(B, L, D)
        # action: 连续(B, L, D)，离散(B, L)
        # reward: (B, L)
        # termination: (B, L)

        if self.length < self.warmup_length:
            print(
                f"[bold blue][Info][/bold blue] replay_buffer length = {self.length} < warmup_length = {self.warmup_length}"
            )
            return edict()
        if batch_size <= 0 or batch_length <= 0:
            print("[bold red][Error][/bold red] batch_size and batch_length should be positive")
            return edict()
        if batch_size % self.num_envs != 0:
            print("[bold red][Error][/bold red] batch_size should be divisible by num_envs")
            return edict()

        # 如果一共要采样batch_size个数据，那么每个环境采样batch_size//num_envs个数据
        obs, action, reward, termination = [], [], [], []
        for i in range(self.num_envs):
            indexes = np.random.randint(0, self.length + 1 - batch_length, size=batch_size // self.num_envs)  # (B//N, )
            obs.append(
                torch.stack([self.obs_buffer[i, idx : idx + batch_length] for idx in indexes])
            )  # (B//N, L, C, H, W) or (B//N, L, D)
            action.append(
                torch.stack([self.action_buffer[i, idx : idx + batch_length] for idx in indexes])
            )  # (B//N, L, D) or (B//N, L)
            reward.append(
                torch.stack([self.reward_buffer[i, idx : idx + batch_length] for idx in indexes])
            )  # (B//N, L)
            termination.append(
                torch.stack([self.termination_buffer[i, idx : idx + batch_length] for idx in indexes])
            )  # (B//N, L)

        replayed_data = edict(
            obs=rearrange(obs, "N S L ... -> (N S) L ...").float(),  # (B, L, C, H, W) or (B, L, D), S = B//N
            action=rearrange(action, "N S L ... -> (N S) L ..."),  # (B, L, D) or (B, L), S = B//N
            reward=rearrange(reward, "N S L -> (N S) L"),  # (B, L), S = B//N
            termination=rearrange(termination, "N S L -> (N S) L"),  # (B, L), S = B//N
        )

        # if self.logger:
        #     self.logger.update_replays(batch_size)

        return replayed_data

    def append(self, sampled_data: edict) -> None:
        # sampled_data:
        # N: 环境数, L: 采样长度, C: 通道数, H: 高, W: 宽, D: 向量维度
        # obs: (N, L, C, H, W) or (N, L, D)
        # action: (N, L, D) or (N, L)
        # reward: (N, L)
        # termination: (N, L)

        L = sampled_data.obs.shape[1]  # L
        start = self.last_pointer
        if start < self.max_length - L:  # 不需要分段存储
            self.obs_buffer[:, start : start + L, ...] = sampled_data.obs
            self.action_buffer[:, start : start + L, ...] = sampled_data.action
            self.reward_buffer[:, start : start + L, ...] = sampled_data.reward
            self.termination_buffer[:, start : start + L, ...] = sampled_data.termination
        else:  # 需要分段存储
            part = self.max_length - start
            self.obs_buffer[:, start:, ...] = sampled_data.obs[:, :part, ...]
            self.action_buffer[:, start:, ...] = sampled_data.action[:, :part, ...]
            self.reward_buffer[:, start:, ...] = sampled_data.reward[:, :part, ...]
            self.termination_buffer[:, start:, ...] = sampled_data.termination[:, :part, ...]

            self.obs_buffer[:, : L - part, ...] = sampled_data.obs[:, part:, ...]
            self.action_buffer[:, : L - part, ...] = sampled_data.action[:, part:, ...]
            self.reward_buffer[:, : L - part, ...] = sampled_data.reward[:, part:, ...]
            self.termination_buffer[:, : L - part, ...] = sampled_data.termination[:, part:, ...]

        self.last_pointer = (self.last_pointer + L) % self.max_length
        self.length = min(self.length + L, self.max_length)

    def __len__(self):
        return self.length
