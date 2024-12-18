#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Serial trainer for off-policy RL algorithms
#  Update Date: 2021-05-21, Shengbo LI: Format Revise
#  Update Date: 2022-04-14, Jiaxin Gao: decrease parameters copy times
#  Update: 2022-12-05, Wenhan Cao: add annotation

__all__ = ["OffSerialTrainer"]

from cmath import inf
import os
import time
import numpy as np
import gin
import ray
import torch
from torch.utils.tensorboard import SummaryWriter
from easydict import EasyDict as edict
from gops.utils.common_utils import ModuleOnDevice
from gops.utils.parallel_task_manager import TaskPool
from gops.utils.tensorboard_setup import add_scalars, tb_tags
from gops.utils.log_data import LogData
from gops.trainer.buffer.replay_buffer import ReplayBuffer
from world_model.diffusion.elucidated_diffusion import REDQTrainer
from world_model.diffusion.train_diffuser import SimpleDiffusionGenerator
from world_model.diffusion.utils import construct_diffusion_model
from world_model.weser import world_models

from gops.trainer.buffer.world_buffer import WorldBuffer
from gops.trainer.sampler.world_sampler import WorldSampler
def combine_two_tensors(tensor1, tensor2)->torch.Tensor:
    return torch.concatenate([tensor1, tensor2], axis=0)


@gin.configurable
class OffSerialDiffTrainer:
    def __init__(self, alg, sampler, buffer:ReplayBuffer, evaluator,disable_diffusion=gin.REQUIRED,trainer_type=gin.REQUIRED, **kwargs):
        self.alg = alg
        self.sampler = sampler
        self.buffer = buffer
        self.per_flag = kwargs["buffer_name"] == "prioritized_replay_buffer"
        self.device = torch.device(kwargs["device"] if torch.cuda.is_available() else "cpu")
        print(f'{self.device} is used.')
        self.evaluator = evaluator
        self.kwargs = kwargs    
        # create center network
        self.networks = self.alg.networks
        self.sampler.networks = self.networks
        
    
        # initialize center network
        if kwargs["ini_network_dir"] is not None:
            self.networks.load_state_dict(torch.load(kwargs["ini_network_dir"]))

        self.replay_batch_size = kwargs["replay_batch_size"]
        self.max_iteration = kwargs["max_iteration"]
        self.sample_interval = kwargs.get("sample_interval", 1)
        self.log_save_interval = kwargs["log_save_interval"]
        self.apprfunc_save_interval = kwargs["apprfunc_save_interval"]
        self.eval_interval = kwargs["eval_interval"]
        self.best_tar = -inf
        self.save_folder = kwargs["save_folder"]
        self.iteration = 0
        
        
        self.disable_diffusion = disable_diffusion
        self.replay_start = kwargs.get("replay_start", 200000)
        self.replay_interval = kwargs.get("replay_interval", 200000)
        self.dim_obs = kwargs.get("obsv_dim", False)
        self.dim_act = kwargs.get("action_dim", False)
        self.diffusion_buffer = None
        self.num_samples = kwargs.get("num_samples", 200_000)
        # self.num_samples = kwargs.get("num_samples", 100000)
        self.diffusion_sample_ratio = kwargs.get("diffusion_ratio", 0.5)
        self.writer = SummaryWriter(log_dir=self.save_folder, flush_secs=20)
        self.diffusion_buffer = ReplayBuffer(**self.kwargs)
        # flush tensorboard at the beginning
        add_scalars(
            {tb_tags["alg_time"]: 0, tb_tags["sampler_time"]: 0}, self.writer, 0
        )
        self.writer.flush()

        self.trainer_type = trainer_type
        print(f'Training with {self.trainer_type}')
        
        #create world replayer
        self.replayer_dict = edict(
            {
                
                "dim_obs": kwargs.get("obsv_dim", False),
                "dim_act": kwargs.get("action_dim", False),
                "is_action_discrete": False,
                "num_envs": kwargs.get("replay_batch_size", 256),
                "max_length": kwargs.get("buffer_size", 1000000),
                "device": self.device,
                "logger": self.writer,
                "warmup_length": kwargs.get("buffer_warm_size", 1000),
                "sample_horizon":8,
                "sample_interval":1 
                
            }
        )
        self.world_replayer = WorldBuffer(**self.replayer_dict)
        self.world_sampler = WorldSampler(self.sampler.env,self.replayer_dict)
    
        # pre sampling
        while self.buffer.size < kwargs["buffer_warm_size"] or self.world_replayer.length < kwargs["buffer_warm_size"]:
            kwargs["mode"] = "train"
            samples, _ = self.sampler.sample()
            world_samples = self.world_sampler.sample(self.iteration,use_random=True)
            self.world_replayer.append(world_samples)
            self.buffer.add_batch(samples)
        self.sampler_tb_dict = LogData()

        # create evaluation tasks
        self.evluate_tasks = TaskPool()
        self.last_eval_iteration = 0

        self.use_gpu = kwargs["use_gpu"]
        if self.use_gpu:
            self.networks.cuda()
        self.start_time = time.time()
        
        diff_dims = self.dim_obs + self.dim_act + 1 + self.dim_obs
        self.inputs = torch.zeros((128, diff_dims)).float()
        self.skip_dims = [self.dim_obs + self.dim_act]
    
    def sample(self):
        diffusion_batch_size = int(self.replay_batch_size * self.diffusion_sample_ratio)
        online_batch_size = int(self.replay_batch_size - diffusion_batch_size)
        batch = {}
        if self.diffusion_buffer.size < diffusion_batch_size:
            return self.buffer.sample_batch(self.replay_batch_size)
        else:
            diffusion_batch = self.diffusion_buffer.sample_batch(batch_size=diffusion_batch_size)
            online_batch = self.buffer.sample_batch(batch_size=online_batch_size)
            obs_tensor = combine_two_tensors(online_batch['obs'], diffusion_batch['obs'])
            next_obs_tensor = combine_two_tensors(online_batch['obs2'], diffusion_batch['obs2'])
            act_tensor = combine_two_tensors(online_batch['act'], diffusion_batch['act'])
            rew_tensor = combine_two_tensors(online_batch['rew'], diffusion_batch['rew']).unsqueeze(1)
            done_tensor = combine_two_tensors(online_batch['done'], diffusion_batch['done']).unsqueeze(1)
            batch['obs'] = obs_tensor
            batch['obs2'] = next_obs_tensor
            batch['act'] = act_tensor
            batch['rew'] = rew_tensor
            batch['done'] = done_tensor
            # print(f'Using Diffusion batch : {diffusion_batch_size}')
            return batch
    @gin.configurable    
    def step(self):
        # sampling
        if self.iteration % self.sample_interval == 0:
            with ModuleOnDevice(self.networks, device="cpu"):
                sampler_samples, sampler_tb_dict = self.sampler.sample()
            self.buffer.add_batch(sampler_samples)
            self.sampler_tb_dict.add_average(sampler_tb_dict)

        # replay
        replay_samples = self.sample()

        # learning
        if self.use_gpu:
            for k, v in replay_samples.items():
                replay_samples[k] = v.cuda()

        self.networks.train()
        if self.per_flag:
            alg_tb_dict, idx, new_priority = self.alg.local_update(
                replay_samples, self.iteration
            )
            self.buffer.update_batch(idx, new_priority)
        else:
            alg_tb_dict = self.alg.local_update(replay_samples, self.iteration)
            # print('iteration update complete!')
        self.networks.eval()

        # log
        if self.iteration % self.log_save_interval == 0:
            print("Iter = ", self.iteration)
            add_scalars(alg_tb_dict, self.writer, step=self.iteration)
            add_scalars(self.sampler_tb_dict.pop(), self.writer, step=self.iteration)

        # save
        if self.iteration % self.apprfunc_save_interval == 0:
            self.save_apprfunc()
        # print(not self.disable_diffusion and (self.iteration + 1) % self.diffusion_interval == 0 and (self.iteration + 1) >= self.diffusion_start)
        
        # diffusion_training
        if self.trainer_type == 'diffusion':
            
            if not self.disable_diffusion and (self.iteration + 1) % self.replay_interval == 0 and (self.iteration + 1) >= self.replay_start:
                print(f'Retraining diffusion model at step {self.iteration + 1}')
                diffusion_trainer = REDQTrainer(
                construct_diffusion_model(
                    inputs=self.inputs,
                    skip_dims=self.skip_dims,
                    disable_terminal_norm=False,
                ),
                model_terminals=False,
                )
                diffusion_trainer.update_normalizer(self.buffer, device=self.device)
                diffusion_trainer.train_from_redq_buffer(self.buffer)
                self.diffusion_buffer = ReplayBuffer(**self.kwargs)
                generator = SimpleDiffusionGenerator(env=self.kwargs, ema_model=diffusion_trainer.ema.ema_model)
                observations, actions, rewards, next_observations, terminals = generator.sample(num_samples=self.num_samples)
        
                # diffusion_sampling  
                print(f'Adding {self.num_samples} samples to replay buffer.')
                for o, a, r, o2, term in zip(observations, actions, rewards, next_observations, terminals):
                    self.diffusion_buffer.store(obs=o, act=a, rew=r, done=term, info={}, next_obs=o2, next_info={}, logp=None)
                # self.buffer.add_batch(self.diffusion_buffer.buf)
                if True:
                    ptr_location = self.buffer.ptr
                    real_observations = self.buffer.buf["obs"][:ptr_location]
                    real_actions = self.buffer.buf['act'][:ptr_location]
                    real_next_observations = self.buffer.buf['obs2'][:ptr_location]
                    real_rewards = self.buffer.buf['rew'][:ptr_location]
                    # Print min, max, mean, std of each dimension in the obs, rew and action
                    print('Buffer stats:')
                    for i in range(observations.shape[1]):
                        print(f'Diffusion Obs {i}: {np.mean(observations[:, i]):.2f} {np.std(observations[:, i]):.2f}')
                        print(
                        f'     Real Obs {i}: {np.mean(real_observations[:, i]):.2f} {np.std(real_observations[:, i]):.2f}')
                    for i in range(actions.shape[1]):
                        print(f'Diffusion Action {i}: {np.mean(actions[:, i]):.2f} {np.std(actions[:, i]):.2f}')
                        print(f'     Real Action {i}: {np.mean(real_actions[:, i]):.2f} {np.std(real_actions[:, i]):.2f}')
                        print(f'Diffusion Reward: {np.mean(rewards):.2f} {np.std(rewards):.2f}')
                        print(f'     Real Reward: {np.mean(real_rewards):.2f} {np.std(real_rewards):.2f}')
                        print(f'Replay buffer size: {ptr_location}')
                        print(f'Diffusion buffer size: {self.diffusion_buffer.ptr}')
        else:
            
            self.world_model = world_models.WorldModel(
                in_channels=self.kwargs.get("in_channels", 1),
                action_dim=self.dim_act,
                transformer_max_length=self.kwargs.get("transformer_max_length", 100),
                transformer_hidden_dim=self.kwargs.get("transformer_hidden_dim", 1024),
                transformer_num_layers=self.kwargs.get("transformer_num_layers", 6),
                transformer_num_heads=self.kwargs.get("transformer_num_heads", 8)
            ).to('cuda:0')
            if (self.iteration + 1) % self.replay_interval == 0 and (self.iteration + 1) >= self.replay_start:
                print(f'Retraining replayer at step {self.iteration + 1}')
                
                
        # evaluate
        if self.iteration - self.last_eval_iteration >= self.eval_interval:
            if self.evluate_tasks.count == 0:
                # There is no evaluation task, add one.
                self._add_eval_task()
            elif self.evluate_tasks.completed_num == 1:
                # Evaluation tasks is completed, log data and add another one.
                objID = next(self.evluate_tasks.completed())[1]
                total_avg_return = ray.get(objID)
                self._add_eval_task()

                if (
                    total_avg_return >= self.best_tar
                    and self.iteration >= self.max_iteration / 5
                ):
                    self.best_tar = total_avg_return
                    print("Best return = {}!".format(str(self.best_tar)))

                    for filename in os.listdir(self.save_folder + "/apprfunc/"):
                        if filename.endswith("_opt.pkl"):
                            os.remove(self.save_folder + "/apprfunc/" + filename)

                    torch.save(
                        self.networks.state_dict(),
                        self.save_folder
                        + "/apprfunc/apprfunc_{}_opt.pkl".format(self.iteration),
                    )

                self.writer.add_scalar(
                    tb_tags["Buffer RAM of RL iteration"],
                    self.buffer.__get_RAM__(),
                    self.iteration,
                )
                self.writer.add_scalar(
                    tb_tags["TAR of RL iteration"], total_avg_return, self.iteration
                )
                self.writer.add_scalar(
                    tb_tags["TAR of replay samples"],
                    total_avg_return,
                    self.iteration * self.replay_batch_size,
                )
                self.writer.add_scalar(
                    tb_tags["TAR of total time"],
                    total_avg_return,
                    int(time.time() - self.start_time),
                )
                self.writer.add_scalar(
                    tb_tags["TAR of collected samples"],
                    total_avg_return,
                    self.sampler.get_total_sample_number(),
                )

    def train(self):
        while self.iteration < self.max_iteration:
            self.step()
            self.iteration += 1

        self.save_apprfunc()
        self.writer.flush()

    def save_apprfunc(self):
        torch.save(
            self.networks.state_dict(),
            self.save_folder + "/apprfunc/apprfunc_{}.pkl".format(self.iteration),
        )

    def _add_eval_task(self):
        with ModuleOnDevice(self.networks, "cpu"):
            self.evaluator.load_state_dict.remote(self.networks.state_dict())
        self.evluate_tasks.add(
            self.evaluator,
            self.evaluator.run_evaluation.remote(self.iteration)
        )
        self.last_eval_iteration = self.iteration
