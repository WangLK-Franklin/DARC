#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Approximate Dynamic Program Algorithm for Finity Horizon (FHADP)
#  Reference: Li SE (2023) 
#             Reinforcement Learning for Sequential Decision and Optimal Control. Springer, Singapore.
#  Update: 2021-03-05, Fawang Zhang: create FHADP algorithm
#  Update: 2022-12-04, Jiaxin Gao: supplementary comment information
#  Update: 2023-08-28, Guojian Zhan: support lr schedule

__all__ = ["FHADP"]

import time
from copy import deepcopy
from typing import Tuple

import torch
from torch.optim import Adam
from gops.algorithm.base import AlgorithmBase, ApprBase
from gops.create_pkg.create_apprfunc import create_apprfunc
from gops.create_pkg.create_env_model import create_env_model
from gops.utils.common_utils import get_apprfunc_dict
from gops.utils.gops_typing import DataDict, InfoDict
from gops.utils.tensorboard_setup import tb_tags
import numpy as np

class ApproxContainer(ApprBase):
    def __init__(
        self,
        *,
        policy_learning_rate: float,
        **kwargs,
    ):
        """Approximate function container for FHADP."""
        """Contains one policy network."""
        super().__init__(**kwargs)
        policy_args = get_apprfunc_dict("policy", **kwargs)
        # policy_args {"obs_dim"}
        
        self.policy = create_apprfunc(**policy_args)
        self.policy_optimizer = Adam(
            self.policy.parameters(), lr=policy_learning_rate
        )
        policy_args["obs_dim"]=int((kwargs["dim_obs"]-kwargs["dim_watermarking"])/2+kwargs["dim_watermarking"])-1
        policy_args["act_dim"]=kwargs["dim_watermarking"]
        policy_args["act_high_lim"]=np.array([1])
        policy_args["act_low_lim"]=np.array([-1])
        self.policy_d = create_apprfunc(**{**policy_args})
        self.policy_optimizer_d = Adam(
            self.policy_d.parameters(), lr=policy_learning_rate
        )
        self.optimizer_dict = {
            "policy": self.policy_optimizer,
            "policy_d": self.policy_optimizer_d,
        }
        self.init_scheduler(**kwargs)

    def create_action_distributions(self, logits):
        """create action distribution"""
        return self.policy.get_act_dist(logits)


class FHADP(AlgorithmBase):
    """Approximate Dynamic Program Algorithm for Finity Horizon

    Paper: https://link.springer.com/book/10.1007/978-981-19-7784-8

    :param int pre_horizon: envmodel predict horizon.
    :param float gamma: discount factor.
    """

    def __init__(
        self,
        *,
        pre_horizon: int,
        gamma: float = 1.0,
        index: int = 0,
        **kwargs,
    ):
        super().__init__(index, **kwargs)
        self.networks = ApproxContainer(**kwargs)
        self.envmodel = create_env_model(**kwargs, pre_horizon=pre_horizon)
        self.pre_horizon = pre_horizon
        self.gamma = gamma
        self.tb_info = dict()
        self.batch_size = kwargs.get("--replay_batch_size",64)

    @property
    def adjustable_parameters(self) -> Tuple[str]:
        para_tuple = ("pre_horizon", "gamma")
        return para_tuple

    def _local_update(self, data: DataDict, iteration: int) -> InfoDict:
        self._compute_gradient(data)
        self.networks.policy_optimizer.step()
        self.networks.policy_optimizer_d.step()
        return self.tb_info

    def get_remote_update_info(self, data: DataDict, iteration: int) -> Tuple[InfoDict, DataDict]:
        self._compute_gradient(data)
        policy_grad1 = [p._grad for p in self.networks.policy.parameters()]
        policy_grad2 = [p._grad for p in self.networks.policy_d.parameters()]
        update_info = dict()
        update_info["grad1"] = policy_grad1
        update_info["grad2"] = policy_grad2
        return self.tb_info, update_info

    def _remote_update(self, update_info: DataDict):
        for p, grad in zip(self.networks.policy.parameters(), update_info["grad1"]):
            p.grad = grad
        self.networks.policy_optimizer.step()
        for p, grad in zip(self.networks.policy_d.parameters(), update_info["grad2"]):
            p.grad = grad
        self.networks.policy_optimizer_d.step()

    def _compute_gradient(self, data: DataDict):
        start_time = time.time()
        self.networks.policy.zero_grad()
        self.networks.policy_d.zero_grad()
        loss_policy, loss_info, generate_data = self._compute_loss_policy(deepcopy(data))
        loss_d, loss_info_d = self._compute_loss_discriminator(generate_data)
        loss = loss_policy + loss_d
        loss.backward()
        end_time = time.time()
        self.tb_info.update(loss_info)
        self.tb_info.update(loss_info_d)
        self.tb_info[tb_tags["alg_time"]] = (end_time - start_time) * 1000  # ms

    def _compute_loss_policy(self, data: DataDict) -> Tuple[torch.Tensor, InfoDict]:
        o_list = []
        o, d = data["obs"], data["done"]
        # ref = data["ref_points"]
        d = torch.zeros_like(d).bool()
        info = data
        # w = data["watermarking"]
        v_pi = 0
        v_d = 0 
        o_list.append(o)
        shape_tensor = torch.zeros((self.batch_size,self.envmodel.dim_watermarking))
        w_zero = torch.zeros(size=shape_tensor.shape)
        for step in range(self.pre_horizon):
            a = self.networks.policy(o, step + 1)
            o, r, d, info = self.envmodel.forward(o, a, d, info)
            v_pi += r * (self.gamma ** step)
            # TODO: chaifen
            o_list.append(o)
        
        
        # for step in range(int(self.pre_horizon)):
        #     input  = torch.cat((o_list[step][:,0:self.envmodel.dim_state], w_zero),dim=1)
        #     a_d = self.networks.policy_d(input,step + 1)
        #     a_d = torch.tanh(a_d)
        #     u = w_zero + a_d
        #     r_d = self.envmodel.compute_reward(o[:,self.envmodel.dim_state*2: ].detach(),u)
        #     w_zero = u
        #     v_d += r_d * (self.gamma ** step)

        loss_actor = -v_pi.mean()
        # loss_discriminator = -v_d.mean()
        # loss_policy = loss_actor + loss_discriminator
        loss_policy = -v_pi.mean()
        loss_info = {
            tb_tags["loss_actor"]: loss_actor.item(),
            # tb_tags["loss_discriminator"]: loss_discriminator.item()
        }
        return loss_policy, loss_info, o_list

    def _compute_loss_discriminator(self, data: list) -> Tuple[torch.Tensor, InfoDict]:
        v_d=0
        for o in data:
            v_d=0
            input  = o[:,0:self.envmodel.dim_state]
            a_d = self.networks.policy_d(input)
            r_d = self.envmodel.compute_reward(o[:,self.envmodel.dim_state*2: ].detach(),a_d)
            v_d = r_d+v_d
        loss_discriminator = -v_d.mean()
        loss_info = {

            tb_tags["loss_discriminator"]: loss_discriminator.item()
        }
        return loss_discriminator, loss_info