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

from world_model.weser import world_models_gops
from gops.trainer.buffer.world_buffer import WorldBuffer
from gops.trainer.sampler.world_sampler import WorldSampler
def combine_two_tensors(tensor1, tensor2)->torch.Tensor:
    return torch.concatenate([tensor1, tensor2], axis=0)


def sequence_to_dict(states, actions, rewards, dones, batch_size):
   
    data_dict = {
        'obs': states[:, 0].to(dtype=torch.float32).detach(),  # (batch_size, state_dim)
        'obs2': states[:, 1].to(dtype=torch.float32).detach(),  # (batch_size, state_dim)
        'act': actions[:, 0].to(dtype=torch.float32).detach(),      # (batch_size, action_dim)
        'rew': rewards[:, 0].to(dtype=torch.float32).detach(),  # (batch_size, 1)
        'done': dones[:, 0].to(dtype=torch.float32).detach()      # (batch_size, 1)
    }
    
    return data_dict

@gin.configurable
class OffSerialWorldTrainer:
    def __init__(self, alg, sampler, buffer:ReplayBuffer, evaluator,trainer_type=gin.REQUIRED, **kwargs):
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
        self.batch_length = 16
        self.max_iteration = kwargs["max_iteration"]
        self.sample_interval = kwargs.get("sample_interval", 1)
        self.log_save_interval = kwargs["log_save_interval"]
        self.apprfunc_save_interval = kwargs["apprfunc_save_interval"]
        self.eval_interval = kwargs["eval_interval"]
        self.best_tar = -inf
        self.save_folder = kwargs["save_folder"]
        self.iteration = 0
        self.sample_ratio = 0.5
        self.replay_dict={}
        
        self.replay_start = kwargs.get("replay_start", 10000)
        self.replay_interval = kwargs.get("replay_interval", 1)
        self.replay_itertation = 0
        self.replay_warm_iteration = 1
        # self.num_samples = kwargs.get("num_samples", 100000)
        self.writer = SummaryWriter(log_dir=self.save_folder, flush_secs=20)

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
                "dim_action": kwargs.get("action_dim", False),
                "is_action_discrete": False,
                "num_train_envs": 1,
                "max_length": kwargs.get("buffer_size", 10000),
                "device": self.device,
                "logger": self.writer,
                "warmup_length": kwargs.get("buffer_warm_size", 1000),
                "sample_horizon":16,
                "sample_interval":1 

            }
        )
        self.world_replayer = WorldBuffer(self.replayer_dict)
        self.world_sampler = WorldSampler(self.sampler.env,self.replayer_dict)

        self.world_model = world_models_gops.WorldModel(
                state_dim=kwargs.get("obsv_dim", False),
                action_dim=kwargs.get("action_dim", False),
                transformer_max_length=self.kwargs.get("transformer_max_length", 64),
                transformer_hidden_dim=self.kwargs.get("transformer_hidden_dim", 512),
                transformer_num_layers=self.kwargs.get("transformer_num_layers", 2),
                transformer_num_heads=self.kwargs.get("transformer_num_heads", 8)
            ).to('cuda:0')
        
        # pre sampling and training
        while self.buffer.size < kwargs["buffer_warm_size"] or self.world_replayer.length < kwargs["buffer_warm_size"]:
            kwargs["mode"] = "train"
            samples, _ = self.sampler.sample()
            world_samples = self.world_sampler.sample(self.iteration,self.networks,use_random=True)
            self.world_replayer.append(world_samples)
            self.buffer.add_batch(samples)
        self.sampler_tb_dict = LogData()

        # create evaluation tasks
        self.evluate_tasks = TaskPool()
        self.last_eval_iteration = 0

        self.use_gpu = kwargs["use_gpu"]
        if self.use_gpu:
            self.networks.to('cuda:0')
        self.start_time = time.time()
       
    def sample(self):
        if self.replay_dict=={} or self.iteration < self.replay_warm_iteration:
            return self.buffer.sample_batch(self.replay_batch_size)
        else:
            # print(f'Using Diffusion batch : {diffusion_batch_size}')
            # return self.buffer.sample_batch(self.replay_batch_size)
            return self.replay_dict
    # def sample(self):
        
    #     return self.buffer.sample_batch(self.replay_batch_size)
        
    @gin.configurable    
    def step(self):
        # sampling
        torch.cuda.empty_cache()
        world_model_tb_dict = {
            "World_model/reward_loss": 0,
            "World_model/termination_loss": 0,
            "World_model/dynamics_loss": 0,
            "World_model/total_loss": 0,
        }
        
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
                replay_samples[k] = v.to('cuda:0')
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

        
        # print(not self.disable_diffusion and (self.iteration + 1) % self.diffusion_interval == 0 and (self.iteration + 1) >= self.diffusion_start)
        # training
        sampled_data = self.world_sampler.sample(cur_step=self.iteration,
                                                     policy_networks=self.networks,
                                                     use_random = False)
        self.world_replayer.append(sampled_data)
        
        if self.world_replayer.ready():
            
            self.world_model.train()
            with torch.set_grad_enabled(True):
                replayed_data = self.world_replayer.replay(batch_size=self.replay_batch_size , batch_length=self.batch_length)
                world_model_tb_dict = self.world_model.update(obs=replayed_data.obs, 
                                        action=replayed_data.action, 
                                        reward=replayed_data.reward,
                                        termination=replayed_data.termination)
            self.world_model.eval() 

                  
            
        if   (self.iteration + 1) % self.replay_interval == 0 and (self.iteration + 1) >= self.replay_start:
            
            with torch.no_grad():
    
                replayed_data = self.world_replayer.replay(batch_size=self.replay_batch_size , batch_length=self.batch_length)
                # self.imagine_batch_size = int(self.replay_batch_size/self.batch_length)

                state, action, reward_hat, termination_hat = self.world_model.imagine_data(
                agent=self.networks, 
                sample_obs=replayed_data.obs, 
                sample_action=replayed_data.action,
                imagine_batch_size=self.replay_batch_size,
                imagine_batch_length=self.batch_length,
                )
                self.replay_dict = sequence_to_dict(state, action, reward_hat, termination_hat, self.replay_batch_size)
                self.replay_itertation += 1
                 
        # log
        if self.iteration % self.log_save_interval == 0:
            print("Iter = ", self.iteration)
            add_scalars(alg_tb_dict, self.writer, step=self.iteration)
            add_scalars(self.sampler_tb_dict.pop(), self.writer, step=self.iteration)
            add_scalars(world_model_tb_dict, self.writer, step=self.iteration) 
        # save
        if self.iteration % self.apprfunc_save_interval == 0:
            self.save_apprfunc()
            
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
