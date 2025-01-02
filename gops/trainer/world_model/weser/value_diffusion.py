import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from tqdm import tqdm
from torch.distributions import Normal
from torchviz import make_dot
def cosine_beta_schedule(timesteps, s=0.008):
    """
    余弦调度的Beta参数计算
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(min=0.0001, max=0.9999)

def _q_evaluate(obs, act, qnet):
        StochaQ = qnet(obs, act)
        mean, std = StochaQ[..., 0], StochaQ[..., -1]
        normal = Normal(torch.zeros_like(mean), torch.ones_like(std))
        z = normal.sample()
        z = torch.clamp(z, -3, 3)
        q_value = mean + torch.mul(z, std)
        return mean, std, q_value
    
class DiffusionModel(nn.Module):
    """
    直接在状态空间生成的扩散模型
    """
    def __init__(
        self,
        state_dim: int,
        n_timesteps=1000,
        hidden_dims=[256, 256, 256],
        dropout_rate=0.1,
        beta_schedule='cosine',
        predict_epsilon=True,
        guidance_scale=1.0,  # 指导强度
    ):
        super().__init__()
        self.state_dim = state_dim
        self.n_timesteps = n_timesteps
        self.predict_epsilon = predict_epsilon
        self.guidance_scale = guidance_scale
        self.mode = "td"
        # 噪声调度
        if beta_schedule == 'linear':
            betas = torch.linspace(1e-4, 2e-2, n_timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(n_timesteps)
        
        # 计算扩散参数
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        
        # 注册buffer
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        
        # 后验分布参数
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # 时间嵌入网络
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 128),
            nn.GELU(),
            nn.Linear(128, 64),
        )
        
        # 去噪网络
        self.denoise_net = self._build_denoise_net(hidden_dims, dropout_rate)


    
    def _build_denoise_net(self, hidden_dims, dropout_rate):
        """构建去噪网络"""
        layers = []
        input_dim = self.state_dim + 128  # 状态维度 + 时间嵌入维度
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, self.state_dim))
        return nn.Sequential(*layers)

    def _time_embedding(self, t):
        """时间步编码"""
        half_dim = 64
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

    # def normalize(self, x):

    #     if self.data_min is not None and self.data_max is not None:
    #         x_norm = 2 * (x - self.data_min) / (self.data_max - self.data_min) - 1
    #     else:
    #         x_norm = (x - self.data_mean) / self.data_std
    #     return x_norm
        
    # def denormalize(self, x):
    #     if self.data_min is not None and self.data_max is not None:
    #         x_orig = (x + 1) / 2 * (self.data_max - self.data_min) + self.data_min
    #     else:
    #         x_orig = x * self.data_std + self.data_mean
    #     return x_orig
    
    def q_sample(self, x_start, t, noise=None):
        """前向扩散过程"""
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_mean_variance(self, x_t, t):
        """计算逆扩散过程的均值和方差"""
        # 构造网络输入
        time_emb = self._time_embedding(t)
        model_input = torch.cat([x_t, time_emb], dim=-1)
        
        # 预测噪声或直接预测x_0
        pred = self.denoise_net(model_input)
        
        # 计算均值
        if self.predict_epsilon:
            x_recon = extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - \
                     extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * pred
        else:
            x_recon = pred
            
        x_recon.clamp_(-1., 1.)
        
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x_t, t=t
        )
        
        return model_mean, posterior_variance, posterior_log_variance

    def q_posterior(self, x_start, x_t, t):
        """计算后验分布参数"""
        posterior_mean = extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + \
                        extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    @torch.no_grad()
    def p_sample(self, x_t, t):
        """单步采样"""
        model_mean, _, model_log_variance = self.p_mean_variance(x_t, t)
        noise = torch.randn_like(x_t)
        nonzero_mask = (1 - (t == 0).float()).reshape(-1, *([1] * (len(x_t.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def sample(self, batch_size=1):
        """生成状态样本"""
        device = next(self.parameters()).device
        
        # 初始化随机噪声
        x = torch.randn((batch_size, self.state_dim), device=device)
        
        # 逐步去噪
        for t in tqdm(reversed(range(0, self.n_timesteps)), desc='采样进度'):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_batch)
            
            # 数值稳定性检查
            x = torch.nan_to_num(x, nan=0.0)
            x = torch.clamp(x, -1., 1.)
        
        return x

    def compute_loss(self, x_start, t):
        """计算训练损失"""
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        
        time_emb = self._time_embedding(t)
        model_input = torch.cat([x_noisy, time_emb], dim=-1)
        pred = self.denoise_net(model_input)
        
        target = noise if self.predict_epsilon else x_start
        return F.mse_loss(pred, target)

    def train_step(self, optimizer, state:torch.Tensor):
        """
        单步训练流程。
        states_actions: [batch_size, state_dim + action_dim]
        """
        batch_size = state.shape[0]
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=state.device)

        optimizer.zero_grad()
        loss = self.compute_loss(state, t)
        loss.backward()
        optimizer.step()
        tb_info = {
            "Generator/diffusion_loss": loss.item(),
        }
        return tb_info

    @torch.no_grad()
    def sample_batch(self, batch_size, num_samples_per_batch=1):
        """批量采样状态
        Args:
            batch_size: 批次大小
            num_samples_per_batch: 每个批次生成的样本数量
        Returns:
            samples: 生成的状态样本 [batch_size, num_samples_per_batch, state_dim]
        """
        device = next(self.parameters()).device
        total_samples = batch_size * num_samples_per_batch
        
        # 初始化随机噪声
        x = torch.randn((total_samples, self.state_dim), device=device)
        
        # 逐步去噪
        for t in reversed(range(0, self.n_timesteps)):
            # 为所有样本创建相同的时间步
            t_batch = torch.full((total_samples,), t, device=device, dtype=torch.long)
            
            # 单步去噪
            x = self.p_sample(x, t_batch)
            
            # 数值稳定性检查
            x = torch.nan_to_num(x, nan=0.0)
            x = torch.clamp(x, -1., 1.)
        
        # 重塑为 [batch_size, num_samples_per_batch, state_dim]
        samples = x.view(batch_size*num_samples_per_batch, self.state_dim)
        
        return samples

    def entropy_guided_p_mean_variance(self, x_t, t, networks):
        """使用TD error指导的扩散过程"""
        # 1. 计算原始去噪预测
        time_emb = self._time_embedding(t)
        model_input = torch.cat([x_t, time_emb], dim=-1)
        pred = self.denoise_net(model_input)

        # 2. 计算TD error梯度
        with torch.enable_grad():
            x_t_grad = x_t.detach().requires_grad_(True)
            
            # 2.1 当前状态的Q值
            logits = networks.policy(x_t_grad)
            act_dist = networks.create_action_distributions(logits)
            action, log_prob = act_dist.rsample()

            # 修改梯度计算
            grad = torch.autograd.grad(-log_prob.sum(), x_t_grad)[0]

        # 3. 计算指导后的预测
        if self.predict_epsilon:
            alpha_t = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
            sigma_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
            guided_pred = pred + self.guidance_scale * grad * sigma_t
            x_recon = extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - \
                     extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * guided_pred
        else:
            x_recon = pred + self.guidance_scale * grad
            
        x_recon.clamp_(-1., 1.)
        
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x_t, t=t
        )
 
        networks.policy.eval()
        return model_mean, posterior_variance, posterior_log_variance
    
    def td_guided_p_mean_variance(self, x_t, t, world_model, networks):
        """使用TD error指导的扩散过程"""
        # 1. 计算原始去噪预测
        time_emb = self._time_embedding(t)
        model_input = torch.cat([x_t, time_emb], dim=-1)
        pred = self.denoise_net(model_input)

        # 2. 计算TD error梯度
        with torch.enable_grad():
            x_t_grad = x_t.detach().requires_grad_(True)
            
            # 2.1 当前状态的Q值
            logits = networks.policy(x_t_grad)
            act_dist = networks.create_action_distributions(logits)
            action, _ = act_dist.rsample()
            
            q1_current = networks.q1(x_t_grad, action)[..., 0]
            q2_current = networks.q2(x_t_grad, action)[..., 0]
            q_current = (q1_current + q2_current) / 2
            x_t_expand = x_t_grad.unsqueeze(1)
            action_expand = action.unsqueeze(1)
            world_model.storm_transformer.reset_kv_cache_list(x_t_expand.shape[0], dtype=x_t_grad.dtype)
            next_obs, reward, _, _ = world_model.predict_next(x_t_expand, action_expand)
            next_obs = next_obs.squeeze(1).float()

            next_logits = networks.policy(next_obs)
            next_act_dist = networks.create_action_distributions(next_logits)
            next_action, _ = next_act_dist.rsample()
            q1_next = networks.q1(next_obs, next_action)[..., 0]
            q2_next = networks.q2(next_obs, next_action)[..., 0]
            q_next = (q1_next + q2_next) / 2

            reward = reward.squeeze(-1)
            target_q = torch.add(reward, torch.mul(q_next, 0.99))
            td_loss = F.smooth_l1_loss(q_current, target_q)
            # 修改梯度计算
            grad = torch.autograd.grad(td_loss.sum(), x_t_grad)[0]


        
        # 3. 计算指导后的预测
        if self.predict_epsilon:
            alpha_t = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
            sigma_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
            guided_pred = pred + self.guidance_scale * grad * sigma_t
            x_recon = extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - \
                     extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * guided_pred
        else:
            x_recon = pred + self.guidance_scale * grad
            
        # x_recon.clamp_(-1., 1.)
        
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x_t, t=t
        )
        world_model.eval()
        networks.policy.eval()
        networks.q1.eval()
        networks.q2.eval()
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def guided_p_sample(self, x_t, t, world_model, policy_net):
        """带拟合指导的单步采样"""
        if self.mode=="td":
            model_mean, _, model_log_variance = self.td_guided_p_mean_variance(x_t, t, world_model, policy_net)
        else:
            model_mean, _, model_log_variance = self.entropy_guided_p_mean_variance(x_t, t, policy_net)
        noise = torch.randn_like(x_t)
        nonzero_mask = (1 - (t == 0).float()).reshape(-1, *([1] * (len(x_t.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def guided_sample(self, batch_size=1, world_model=None, policy_net=None):
        """带拟合器指导的采样过程"""
        device = next(self.parameters()).device
        
        # 初始化随机噪声
        x = torch.randn((batch_size, self.state_dim), device=device)
        
        # 逐步去噪
        for t in tqdm(reversed(range(0, self.n_timesteps)), desc='Guided sampling'):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            if world_model is not None and policy_net is not None:
                x = self.guided_p_sample(x, t_batch, world_model, policy_net)
            else:
                x = self.p_sample(x, t_batch)
            
            # 数值稳定性检查
            x = torch.nan_to_num(x, nan=0.0)
            x = torch.clamp(x, -1., 1.)
        
        return x

    @torch.no_grad()
    def guided_sample_batch(self, batch_size, num_samples_per_batch=1, world_model=None, policy_net=None):
        """带拟合器指导的批量采样
        Args:
            batch_size: 批次大小
            num_samples_per_batch: 每个批次的样本数
            world_model: 世界模型网络
            policy_net: 策略网络
        Returns:
            samples: 生成的样本 [batch_size * num_samples_per_batch, state_dim]
        """
        device = next(self.parameters()).device
        total_samples = batch_size * num_samples_per_batch
        
        # 初始化随机噪声
        x = torch.randn((total_samples, self.state_dim), device=device)
        
        # 逐步去噪
        for t in reversed(range(0, self.n_timesteps)):
            t_batch = torch.full((total_samples,), t, device=device, dtype=torch.long)
            if world_model is not None and policy_net is not None:
                x = self.guided_p_sample(x, t_batch, world_model, policy_net)
            else:
                x = self.p_sample(x, t_batch)
            
            # 数值稳定性检查
            x = torch.nan_to_num(x, nan=0.0)
            x = torch.clamp(x, -1., 1.)
        
        return x

def extract(a, t, x_shape):
    """辅助函数：提取适当形状的系数"""
    b = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))




