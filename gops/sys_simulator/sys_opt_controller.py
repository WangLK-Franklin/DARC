import argparse
import time
from typing import List, Tuple
from gops.create_pkg.create_env import create_env
from gops.create_pkg.create_env_model import create_env_model
from gops.env.env_ocp.pyth_base_model import PythBaseModel
import matplotlib.pyplot as plt
import torch
import torch.autograd.functional as F
import os
import numpy as np
import scipy.optimize as opt
from cyipopt import minimize_ipopt

class OptController:
    def __init__(
        self, 
        model: PythBaseModel, 
        ctrl_dt: float, 
        num_pred_step: int, 
        minimize_options: dict={},
        verbose=True,
    ):

        self.model = model
        self.ctrl_dt = ctrl_dt
        self.sim_dt = model.dt
        self.obs_dim = model.obs_dim
        self.action_dim = model.action_dim
        self.num_pred_step = num_pred_step
        self.initial_guess = np.zeros(self.action_dim * num_pred_step)
        self.minimize_options = minimize_options
        self.bounds = opt.Bounds(
            np.tile(self.model.action_lower_bound, (num_pred_step,)), 
            np.tile(self.model.action_upper_bound, (num_pred_step,))
        )
        self.verbose = verbose
        self.__reset_statistics()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        res = minimize_ipopt(
            fun=self.__cost_fcn, 
            x0=self.initial_guess,
            args=(x,),
            jac=self.__cost_jac,
            bounds=opt._constraints.new_bounds_to_old(self.bounds.lb, self.bounds.ub, self.num_pred_step * self.action_dim),
            constraints=[{
                "type": "ineq", 
                "fun": self.__constraint_fcn,
                "jac": self.__constraint_jac,
                "args": (x,)
            }],
            options=self.minimize_options
        )
        self.initial_guess = np.concatenate((
            res.x[self.action_dim:], 
            np.zeros(self.action_dim)
        ))
        if self.verbose:
            self.__print_statistics(res)
        return res.x.reshape((self.action_dim, -1))[:, 0]

    def __cost_fcn(self, inputs: np.ndarray, x: np.ndarray) -> float:
        x = torch.tensor(x, dtype=torch.float32)
        inputs = torch.tensor(
            inputs, 
            dtype=torch.float32, 
            requires_grad=True
        )
        cost = self.__compute_cost(inputs, x)
        return cost.detach().item()

    def __cost_jac(self, inputs: np.ndarray, x: np.ndarray) -> np.ndarray:
        x = torch.tensor(x, dtype=torch.float32)
        inputs = torch.tensor(
            inputs, 
            dtype=torch.float32, 
            requires_grad=True
        )
        cost = self.__compute_cost(inputs, x)
        jac = torch.autograd.grad(cost, inputs)[0]
        return jac.numpy().astype("d")

    def __constraint_fcn(self, inputs: np.ndarray, x: np.ndarray) -> torch.Tensor:

        if self.model.get_constraint is None:
            return torch.tensor([0.])
        else:
            self.constraint_evaluations += 1

            x = torch.tensor(x, dtype=torch.float32)
            inputs = torch.tensor(
                inputs, 
                dtype=torch.float32
            )
            states, _ = self.__rollout(inputs, x)
            
            return self.model.get_constraint(states)

    def __constraint_jac(self, inputs: np.ndarray, x: np.ndarray) -> np.ndarray:
        x = torch.tensor(x, dtype=torch.float32)
        inputs = torch.tensor(
            inputs, 
            dtype=torch.float32, 
            requires_grad=True
        )
        jac = F.jacobian(self.__constraint_fcn, (inputs, x))[0]
        return jac.numpy().astype("d")

    def __rollout(self, inputs: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List]:
        self.system_simulations += 1
        inputs_reshaped = inputs.reshape((self.action_dim , -1))
        states = torch.zeros((self.obs_dim, self.num_pred_step + 1))
        rewards = torch.zeros(self.num_pred_step)
        states[:, 0] = x
        done = torch.tensor([False])
        info = {}

        for i in range(self.num_pred_step):
            next_x = states[:, i].clone().unsqueeze(0)
            u = inputs_reshaped[:, i].unsqueeze(0)
            for _ in range(int(self.ctrl_dt / self.sim_dt)):
                next_x, reward, done, info = self.model.forward(
                    next_x, 
                    u,
                    done=done,
                    info=info
                )
            states[:, i + 1] = next_x
            rewards[i] = -reward
        return states, rewards
    
    def __compute_cost(self, inputs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # rollout the states and rewards
        states, rewards = self.__rollout(inputs, x)

        # sum up the intergral costs from timestep 0 to T-1
        cost = torch.sum(rewards)

        # Terminal cost for timestep T
        if self.model.get_terminal_cost is not None:
            cost += self.model.get_terminal_cost(states[:, -1])
        return cost

    def __reset_statistics(self):
        """Reset counters for keeping track of statistics"""
        self.constraint_evaluations = 0
        self.system_simulations = 0

    def __print_statistics(self, res: opt.OptimizeResult, reset=True):
        """Print out summary statistics from last run"""
        print(res.message)
        print("Summary statistics:")
        print("* Number of iterations:", res.nit)
        print("* Cost function calls:", res.nfev)
        if self.constraint_evaluations:
            print("* Constraint calls:", self.constraint_evaluations)
        print("* System simulations:", self.system_simulations)
        print("* Final cost:", res.fun, "\n")
        if reset:
            self.__reset_statistics()


class NNController:
    def __init__(self, args, log_policy_dir):
        print(args)
        alg_name = args["algorithm"]
        alg_file_name = alg_name.lower()
        file = __import__(alg_file_name)
        ApproxContainer = getattr(file, "ApproxContainer")
        self.networks = ApproxContainer(**args)

        for filename in os.listdir(log_policy_dir + "/apprfunc"):
            if filename.endswith("_opt.pkl"):
                log_path = os.path.join(log_policy_dir, "apprfunc", filename)
        self.networks.load_state_dict(torch.load(log_path))

    def __call__(self, obs):
        batch_obs = torch.from_numpy(np.expand_dims(obs, axis=0).astype("float32"))
        logits = self.networks.policy(batch_obs)
        action_distribution = self.networks.create_action_distributions(logits)
        action = action_distribution.mode()
        action = action.detach().numpy()[0]
        return action


if __name__ == "__main__":
    # Parameters Setup
    parser = argparse.ArgumentParser()
    env_id = "pyth_oscillatorconti"
    parser.add_argument("--env_id", type=str, default=env_id)
    parser.add_argument("--lq_config", type=str, default="s4a2")
    parser.add_argument('--clip_action', type=bool, default=True)
    parser.add_argument('--clip_obs', type=bool, default=False)
    parser.add_argument('--mask_at_done', type=bool, default=True)
    parser.add_argument(
        "--is_adversary", type=bool, default=False, help="Adversary training"
    )
    parser.add_argument('--sample_batch_size', type=int, default=64, help='Batch size of sampler for buffer store = 64')

    if env_id == "pyth_aircraftconti":
        parser.add_argument('--max_episode_steps', type=int, default=200)
        parser.add_argument('--gamma_atte', type=float, default=5)
        parser.add_argument('--fixed_initial_state', type=list, default=[1.0, 1.5, 1.0], help='for env_data')
        parser.add_argument('--initial_state_range', type=list, default=[0.1, 0.2, 0.1], help='for env_model')
        parser.add_argument('--state_threshold', type=list, default=[2.0, 2.0, 2.0])
        parser.add_argument('--lower_step', type=int, default=200, help='for env_model')
        parser.add_argument('--upper_step', type=int, default=700, help='for env_model')
    
    if env_id == "pyth_oscillatorconti":
        parser.add_argument('--max_episode_steps', type=int, default=200)
        parser.add_argument('--gamma_atte', type=float, default=2)
        parser.add_argument('--fixed_initial_state', type=list, default=[0.5, -0.5], help='for env_data [0.5, -0.5]')
        parser.add_argument('--initial_state_range', type=list, default=[1.5, 1.5], help='for env_model')
        parser.add_argument('--state_threshold', type=list, default=[5.0, 5.0])
        parser.add_argument('--lower_step', type=int, default=200, help='for env_model')
        parser.add_argument('--upper_step', type=int, default=700, help='for env_model')

    if env_id == "pyth_suspensionconti":
        parser.add_argument('--gamma_atte', type=float, default=30)
        parser.add_argument('--state_weight', type=list, default=[1000.0, 3.0, 100.0, 0.1])
        parser.add_argument('--control_weight', type=list, default=[1.0])
        parser.add_argument('--fixed_initial_state', type=list, default=[0, 0, 0, 0], help='for env_data')
        parser.add_argument('--initial_state_range', type=list, default=[0.05, 0.5, 0.05, 1.0], help='for env_model')
        # State threshold
        parser.add_argument('--state_threshold', type=list, default=[0.08, 0.8, 0.1, 1.6])
        parser.add_argument('--lower_step', type=int, default=200, help='for env_model')
        parser.add_argument('--upper_step', type=int, default=500, help='for env_model')  # shorter, faster but more error
        parser.add_argument('--max_episode_steps', type=int, default=1500, help='for env_data')
        parser.add_argument('--max_newton_iteration', type=int, default=50)
        parser.add_argument('--max_iteration', type=int, default=parser.parse_args().max_newton_iteration)

    args = vars(parser.parse_args())
    env_model = create_env_model(**args)
    obs_dim = env_model.obs_dim
    action_dim = env_model.action_dim

    ctrl_dt = env_model.dt
    if args["env_id"] == "pyth_lq":
        max_state_errs = []
        mean_state_errs = []
        max_action_errs = []
        mean_action_errs = []
        K = env_model.dynamics.compute_control_matrix()
    
    times = []
    # num_pred_steps = range(70, 100, 10)
    num_pred_steps = (70,)
    for num_pred_step in num_pred_steps:
        controller = OptController(
            env_model, 
            ctrl_dt=ctrl_dt, 
            num_pred_step=num_pred_step, 
            minimize_options={
                "max_iter": 200, 
                "tol": 1e-5,
                # "acceptable_tol": 1e-1,
                # "acceptable_iter": 10,
                # "print_level": 5,
            }
        )

        env = create_env(**args)
        env.seed(0)
        x = env.reset()
        sim_num = 2500
        sim_horizon = np.arange(sim_num)
        xs = []
        us= []
        ts = []
        for i in sim_horizon:
            print(f"step: {i + 1}")
            t1 = time.time()
            u = controller(x.astype(np.float32))
            t2 = time.time()
            xs.append(x)
            us.append(u)
            ts.append(t2 - t1)
            x, _, _, _ = env.step(u)
        xs = np.stack(xs)
        us = np.stack(us)
        times.append(ts)

        if args["env_id"] == "pyth_lq":
            env.seed(0)
            x = env.reset()
            xs_lqr = []
            us_lqr= []
            for i in sim_horizon:
                u = -K @ x
                xs_lqr.append(x)
                us_lqr.append(u)
                x, _, _, _ = env.step(u)
            xs_lqr = np.stack(xs_lqr)
            us_lqr = np.stack(us_lqr)

            max_state_err = np.max(np.abs(xs - xs_lqr), axis=0) / (np.max(xs_lqr, axis=0) - np.min(xs_lqr, axis=0)) * 100
            mean_state_err = np.mean(np.abs(xs - xs_lqr), axis=0) / (np.max(xs_lqr, axis=0) - np.min(xs_lqr, axis=0)) * 100
            max_action_err = np.max(np.abs(us - us_lqr), axis=0) / (np.max(us_lqr, axis=0) - np.min(us_lqr, axis=0)) * 100
            mean_action_err = np.mean(np.abs(us - us_lqr), axis=0) / (np.max(us_lqr, axis=0) - np.min(us_lqr, axis=0)) * 100
            max_state_errs.append(max_state_err)
            mean_state_errs.append(mean_state_err)
            max_action_errs.append(max_action_err)
            mean_action_errs.append(mean_action_err)

    if args["env_id"] == "pyth_lq":
        max_state_errs = np.stack(max_state_errs)
        mean_state_errs = np.stack(mean_state_errs)
        max_action_errs = np.stack(max_action_errs)
        mean_action_errs = np.stack(mean_action_errs)

    #=======state-timestep=======#
    plt.figure()
    for i in range(obs_dim):
        plt.subplot(obs_dim, 1, i + 1)
        plt.plot(sim_horizon, xs[:, i], label="mpc")
        if args["env_id"] == "pyth_lq":
            plt.plot(sim_horizon, xs_lqr[:, i], label="lqr")
            print(f"State-{i+1} Max error: {round(max_state_err[i], 3)}%, Mean error: {round(mean_state_err[i], 3)}%")
        plt.ylabel(f"State-{i+1}")
    plt.legend()
    plt.xlabel("Time Step")
    if args["env_id"] == "pyth_lq":
        plt.savefig(f"State-{args['env_id']}-{args['lq_config']}.png")
    else:
        plt.savefig(f"State-{args['env_id']}.png")

    #=======action-timestep=======#
    plt.figure()
    for i in range(action_dim):
        plt.subplot(action_dim, 1, i + 1)
        plt.plot(sim_horizon, us[:, i], label="mpc")
        if args["env_id"] == "pyth_lq":
            plt.plot(sim_horizon, us_lqr[:, i], label="lqr")
            print(f"Action-{i+1} Max error: {round(max_action_err[i], 3)}%, Mean error: {round(mean_action_err[i], 3)}%")
        plt.ylabel(f"Action-{i+1}")
    plt.legend()
    plt.xlabel("Time Step")
    if args["env_id"] == "pyth_lq":
        plt.savefig(f"Action-{args['env_id']}-{args['lq_config']}.png")
    else:
        plt.savefig(f"Action-{args['env_id']}.png")

    #=======MPC solving times=======#
    plt.figure()
    plt.boxplot(times, labels=num_pred_steps, showfliers=False)
    plt.xlabel("num pred step")
    plt.ylabel(f"Time (s)")
    if args["env_id"] == "pyth_lq":
        plt.savefig(f"MPC-solving-time-{args['env_id']}-{args['lq_config']}.png")
    else:
        plt.savefig(f"MPC-solving-time-{args['env_id']}.png")

    #=======error-predstep=======#
    if args["env_id"] == "pyth_lq":
        plt.figure()
        for i in range(obs_dim):
            plt.plot(num_pred_steps, max_state_errs[:, i], label=f"State-{i+1}")
        plt.legend()
        plt.xlabel("num pred step")
        plt.ylabel(f"State Max error (%)")
        plt.yscale ('log')
        plt.savefig(f"State-max-err-{args['env_id']}-{args['lq_config']}.png")

        plt.figure()
        for i in range(action_dim):
            plt.plot(num_pred_steps, max_action_errs[:, i], label=f"Action-{i+1}")
        plt.legend()
        plt.xlabel("num pred step")
        plt.ylabel(f"Action Max error (%)")
        plt.yscale ('log')
        plt.savefig(f"Action-max-err-{args['env_id']}-{args['lq_config']}.png")

        plt.figure()
        for i in range(obs_dim):
            plt.plot(num_pred_steps, mean_state_errs[:, i], label=f"State-{i+1}")
        plt.legend()
        plt.xlabel("num pred step")
        plt.ylabel(f"State Mean error (%)")
        plt.yscale ('log')
        plt.savefig(f"State-mean-err-{args['env_id']}-{args['lq_config']}.png")

        plt.figure()
        for i in range(action_dim):
            plt.plot(num_pred_steps, mean_action_errs[:, i], label=f"Action-{i+1}")
        plt.legend()
        plt.xlabel("num pred step")
        plt.ylabel(f"Action Mean error (%)")
        plt.yscale ('log')
        plt.savefig(f"Action-mean-err-{args['env_id']}-{args['lq_config']}.png")