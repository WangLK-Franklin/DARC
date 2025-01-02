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

from world_model.weser import world_models_diffusion
from gops.trainer.buffer.world_buffer import WorldBuffer
from gops.trainer.sampler.world_sampler import WorldSampler
from gops.trainer.world_model.weser.value_diffusion import DiffusionModel
def combine_two_tensors(tensor1, tensor2)->torch.Tensor:
    return torch.concatenate([tensor1, tensor2], axis=0)


def sequence_to_dict(states, actions, rewards, dones, buffer:ReplayBuffer):
    # 1. 先进行批量数据类型转换
    states = states.to(torch.float32)
    actions = actions.to(torch.float32)
    rewards = rewards.to(torch.float32)
    dones = dones.to(torch.float32)
    
    # 2. 一次性转换为numpy
    states_np = states.cpu().numpy()
    actions_np = actions.cpu().numpy()
    rewards_np = rewards.cpu().numpy()
    dones_np = dones.cpu().numpy()
    
    # 3. 预分配列表大小
    batch_size, seq_len = states.shape[:2]
    imagine_samples = []
    imagine_samples.extend(
        (
            states_np[j, i],#obs
            actions_np[j, i],#act
            rewards_np[j, i],#rew
            dones_np[j, i],#done
            {"mode": "imagine"},#info
            states_np[j, i+1],#next_obs
            {"mode": "imagine"},#next_info
            np.zeros(1, dtype=np.float32),#logp
        )
        for j in range(batch_size)
        for i in range(seq_len-1)  # -1 因为需要next_obs
    )
    
    # 4. 一次性添加所有样本
    buffer.add_batch(imagine_samples)
    
    

@gin.configurable
class OffSerialGenTrainer:
    def __init__(self, alg, sampler, buffer:ReplayBuffer, evaluator, **kwargs):
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
        
        self.default_sample = True
    
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
        
        if kwargs["trainer_mode"] == "debug":
            self.replay_start = 1
            self.replay_interval = 1
        else:
            self.replay_start = kwargs.get("replay_start", 400000)
            self.replay_interval = kwargs.get("replay_interval", 10000)
        self.replay_itertation = 0
        self.replay_warm_iteration = 1
        # self.num_samples = kwargs.get("num_samples", 100000)
        self.writer = SummaryWriter(log_dir=self.save_folder, flush_secs=20)

        # flush tensorboard at the beginning
        add_scalars(
            {tb_tags["alg_time"]: 0, tb_tags["sampler_time"]: 0}, self.writer, 0
        )
        self.writer.flush()

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

        self.world_model = world_models_diffusion.WorldModel(
                state_dim=kwargs.get("obsv_dim", None),
                action_dim=kwargs.get("action_dim", None),
                latent_dim=self.kwargs.get("latent_dim", 256),
                transformer_max_length=self.kwargs.get("transformer_max_length", 64),
                transformer_hidden_dim=self.kwargs.get("transformer_hidden_dim", 512),
                transformer_num_layers=self.kwargs.get("transformer_num_layers", 2),
                transformer_num_heads=self.kwargs.get("transformer_num_heads", 8)
            ).to('cuda:0')
        
        self.generator_model = DiffusionModel(
            state_dim=kwargs.get("obsv_dim", None)).to('cuda:0')
        self.optimizer = torch.optim.Adam(self.generator_model.parameters(), lr=1e-5)

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
       
    # def sample(self):
    #     if self.replay_dict=={} or self.iteration < self.replay_warm_iteration:
    #         return self.buffer.sample_batch(self.replay_batch_size)
    #     else:
    #         # print(f'Using Diffusion batch : {diffusion_batch_size}')
    #         return self.buffer.sample_batch(self.replay_batch_size)
    #         # return self.replay_dict
    
    def mixed_sample(self, imagine_ratio=0.5):
        policy_flag = False
        total_batch_size = self.replay_batch_size
        imagine_batch_size = int(total_batch_size * imagine_ratio)
        real_batch_size = total_batch_size - imagine_batch_size
        
        real_samples = self.buffer.sample_batch(real_batch_size)
        
        if (self.iteration + 1) % self.replay_interval == 0 and (self.iteration + 1) >= self.replay_start:
            policy_flag = False
            self.generator_model.mode = "td"
            gen_state = self.generator_model.guided_sample_batch(
                batch_size=imagine_batch_size, 
                world_model=self.world_model, 
                policy_net=self.networks
            )
            
            logits = self.networks.policy(gen_state)
            action_distribution = self.networks.create_action_distributions(logits)
            gen_action, _ = action_distribution.sample()
            
            state, action, reward_hat, termination_hat = self.world_model.imagine_data(
                agent=self.networks,
                sample_obs=gen_state,
                sample_action=gen_action,
                imagine_batch_size=imagine_batch_size,
                imagine_batch_length=1 
            )
            
            imagine_samples = {
                "obs": state[:, :1].squeeze(1).to('cpu'),
                "act": action.squeeze(1).to('cpu'),
                "rew": reward_hat.squeeze(1).to('cpu'),
                "done": termination_hat.squeeze(1).to('cpu'),
                "obs_next": state[:, 1:].squeeze(1).to('cpu') if state.shape[1] > 1 else state.squeeze(1).to('cpu')
            }
            
            # 3. 合并两种数据
            with torch.no_grad():
                mixed_samples = {}
                for key in real_samples.keys():
                    if key in imagine_samples:
                        mixed_samples[key] = torch.cat([
                            real_samples[key],
                            imagine_samples[key]
                        ], dim=0)
                    else:
                        # 对于imagine_samples中没有的键，用real_samples的值填充
                        mixed_samples[key] = torch.cat([
                            real_samples[key],
                            real_samples[key][:imagine_batch_size]
                        ], dim=0)
            
            self.generator_model.mode = "entropy"
            gen_state_for_policy = self.generator_model.guided_sample_batch(
            batch_size=self.replay_batch_size, 
            world_model=self.world_model, 
            policy_net=self.networks
            )
            policy_samples = {
                "obs": gen_state_for_policy.detach().to('cpu'),
                "act": action.squeeze(1).to('cpu'),
                "rew": reward_hat.squeeze(1).to('cpu'),
                "done": termination_hat.squeeze(1).to('cpu'),
                "obs_next": gen_state_for_policy.detach().to('cpu')
            }
            return self.buffer.sample_batch(total_batch_size),mixed_samples,policy_samples,policy_flag
        
        # 如果world_replayer还没准备好，返回全部真实数据
        return self.buffer.sample_batch(total_batch_size),self.buffer.sample_batch(total_batch_size),self.buffer.sample_batch(total_batch_size),policy_flag
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
            if self.default_sample:
                self.buffer.add_batch(sampler_samples)
            self.sampler_tb_dict.add_average(sampler_tb_dict)

        # replay
        replay_samples,imagine_samples,policy_samples,policy_flag = self.mixed_sample()
        
        
        # learning
        if self.use_gpu:
                for k, v in replay_samples.items():
                    replay_samples[k] = v.to('cuda:0')
                for k, v in imagine_samples.items():
                    imagine_samples[k] = v.to('cuda:0')
                for k, v in policy_samples.items():
                    policy_samples[k] = v.to('cuda:0')
        self.generator_model.to('cuda:0')
        self.generator_model.train()
        state=replay_samples["obs"]
        # action=replay_samples["act"]
        gen_info = self.generator_model.train_step(self.optimizer, state)
        
        self.networks.train()
        if self.per_flag:
            alg_tb_dict, idx, new_priority = self.alg.local_update(
                imagine_samples, policy_samples, self.iteration,policy_flag
            )
            self.buffer.update_batch(idx, new_priority)
        else:
            alg_tb_dict = self.alg.local_update(imagine_samples, policy_samples, self.iteration,policy_flag)
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

                  
            
        # if   (self.iteration + 1) % self.replay_interval == 0 and (self.iteration + 1) >= self.replay_start:
        #     self.default_sample = False
            
        #     # replayed_data = self.world_replayer.replay(batch_size=int(self.replay_batch_size/self.batch_length) , batch_length=self.batch_length)
        #     # self.imagine_batch_size = int(self.replay_batch_size/self.batch_length)
        #     gen_state = self.generator_model.guided_sample_batch(batch_size=int(self.replay_batch_size/self.batch_length), world_model=self.world_model, policy_net=self.networks)
        #     logits = self.networks.policy(gen_state)
        #     action_distribution = self.networks.create_action_distributions(logits)
        #     gen_action, logp = action_distribution.sample()
        #     state, action, reward_hat, termination_hat = self.world_model.imagine_data(
        #     agent=self.networks, 
        #     sample_obs=gen_state, 
        #     sample_action=gen_action,
        #     imagine_batch_size=self.replay_batch_size,
        #     imagine_batch_length=self.batch_length,
        #     )
        #     sequence_to_dict(state, action, reward_hat, termination_hat, self.buffer)
        #     self.replay_itertation += 1
                 
        # log
        if self.iteration % self.log_save_interval == 0:
            print("Iter = ", self.iteration)
            add_scalars(alg_tb_dict, self.writer, step=self.iteration)
            add_scalars(self.sampler_tb_dict.pop(), self.writer, step=self.iteration)
            add_scalars(world_model_tb_dict, self.writer, step=self.iteration) 
            add_scalars(gen_info, self.writer, step=self.iteration)
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
