import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from tqdm import tqdm

def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule as proposed in:
    https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = betas.clamp(min=0.0001, max=0.9999)
    return betas

def extract(a, t, x_shape):
    """
    将时刻 t 对应的标量从向量 a 中提取出来，并 reshape 到与 x_shape 相匹配的形状。
    a: [timesteps]
    t: [batch_size]
    x_shape: x 的形状，如 [batch_size, dim]
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t)  # 根据 t 的每个元素索引到 a 中
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

class DiffusionModel(nn.Module):
    """
    一个通用的扩散模型，用于在隐空间生成样本，并结合外部的编码器和解码器进行状态-动作对的建模。
    编码器 (encoder): 将原始 (state, action) 映射到隐向量 latent
    解码器 (decoder): 将隐向量 latent 映射回原始 (state, action)

    Args:
        state_dim (int): 状态的维度
        action_dim (int): 动作的维度
        encoder (nn.Module): 外部提供的编码器模型
        decoder (nn.Module): 外部提供的解码器模型
        n_timesteps (int): 扩散过程的总步数
        hidden_dims (List[int]): 中间网络（denoise_net）的隐藏层维度列表
        embed_dim (int): 隐空间的维度（即编码器输出的维度）
        dropout_rate (float): Dropout 概率
        beta_schedule (str): 噪声调度方案，'linear' 或 'cosine'
        predict_epsilon (bool): 模型是否输出噪声项 epsilon；若为 False 则直接输出重构后的 x
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        encoder: nn.Module,
        decoder: nn.Module,
        n_timesteps=1000,
        hidden_dims=[256, 256, 256],
        embed_dim=256,
        dropout_rate=0.1,
        beta_schedule='cosine',
        predict_epsilon=True,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.output_dim = state_dim + action_dim
        self.embed_dim = embed_dim
        self.n_timesteps = n_timesteps
        self.predict_epsilon = predict_epsilon
        self.dropout_rate = dropout_rate
        self.hidden_dims = hidden_dims

        # --- 噪声 Beta 参数的调度 ---
        if beta_schedule == 'linear':
            betas = torch.linspace(1e-4, 2e-2, n_timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(n_timesteps)
        else:
            raise ValueError(f"Unsupported beta_schedule: {beta_schedule}")

        # --- 计算扩散过程中的 Alpha 参数 ---
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)                    # \prod_{i=1}^t alpha_i
        alphas_cumprod_prev = torch.cat([torch.ones(1, device=betas.device), alphas_cumprod[:-1]])

        # --- 注册到 buffer，便于在训练和推理时直接使用 ---
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # --- 后验分布参数 (q(x_{t-1} | x_t, x_0)) ---
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # --- 外部注入的编解码器 ---
        self.encoder = encoder
        self.decoder = decoder

        # --- 时间嵌入网络 ---
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 128),
            nn.GELU(),
            nn.Linear(128, 128),
        )

        # --- 去噪网络 denoise_net，用于在隐空间中预测噪声或重构向量 ---
        self.denoise_net = self._build_denoise_net()

    def _build_denoise_net(self):
        """
        构建将 [latent + time_emb] -> [latent] 的网络。
        如果 predict_epsilon=True，则输出预测噪声 epsilon；
        否则直接输出对 x_0 的重构。
        """
        layers = []
        input_dim = self.embed_dim + 128  # 隐向量维度 + 时间嵌入维度

        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            ])
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, self.embed_dim))
        return nn.Sequential(*layers)

    def _time_embedding(self, t):
        """
        将离散的扩散步数 t 映射到时间嵌入向量。
        t: [batch_size]
        """
        half_dim = 64
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]  # [batch_size, half_dim]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # [batch_size, 2 * half_dim]
        return emb  # [batch_size, 128]

    # -------------------------------------------------------------------------
    #                               扩散过程关键函数
    # -------------------------------------------------------------------------
    def q_sample(self, x_start, t, noise=None):
        """
        前向扩散过程：q(x_t | x_0)
        给定 x_0 (即初始的干净样本)，在第 t 步加上噪声，得到 x_t。
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def predict_start_from_noise(self, x_t, t, noise):
        """
        根据网络预测的噪声，反推出 x_0 (或者直接预测 x_0)。
        """
        if self.predict_epsilon:
            # x_0 = (1/sqrt(alpha_t)) * x_t - (beta_t/sqrt(1-alpha_bar_t)) * epsilon
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            # 如果网络直接输出 x_0，则不需要再计算
            return noise

    def q_posterior(self, x_start, x_t, t):
        """
        后验分布 q(x_{t-1} | x_t, x_0) 的均值和方差。
        """
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x_t, t):
        """
        p(x_{t-1} | x_t) 的均值和方差，由网络预测 x_0 后得到。
        """
        # 1) 构造网络输入
        time_emb = self._time_embedding(t)              # [batch_size, 128]
        model_input = torch.cat([x_t, time_emb], dim=-1)  # [batch_size, embed_dim + 128]

        # 2) 预测噪声或 x_0
        noise_pred = self.denoise_net(model_input)

        # 3) 得到 x_0
        x_recon = self.predict_start_from_noise(x_t, t, noise_pred)
        x_recon.clamp_(-1., 1.)  # 避免数值爆炸，可根据任务需求进行裁剪

        # 4) 通过后验参数得到 x_{t-1} 的分布
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x_t, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x_t, t):
        """
        从 p(x_{t-1} | x_t) 中采样，得到 x_{t-1}。
        """
        batch_size = x_t.shape[0]
        model_mean, _, model_log_variance = self.p_mean_variance(x_t, t)

        # 当 t>0 时加噪声，当 t=0 时不再加噪声
        noise = torch.randn_like(x_t)
        nonzero_mask = (1 - (t == 0).float()).reshape(batch_size, *((1,) * (len(x_t.shape) - 1)))

        # x_{t-1}
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    # -------------------------------------------------------------------------
    #                               推理 / 采样
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, batch_size=1):
        """
        在隐空间中随机采样 x_T (高斯噪声)，然后逐步反向扩散得到 x_0，
        最后用解码器将 x_0 解码成 (state, action)。
        """
        device = next(self.parameters()).device
        # 1) 在隐空间初始化随机噪声 x_T
        x = torch.randn((batch_size, self.embed_dim), device=device)

        # 2) 逐步去噪，从 T-1 到 0
        for t_ in tqdm(reversed(range(0, self.n_timesteps)), desc='采样进度'):
            t_ = torch.full((batch_size,), t_, device=device, dtype=torch.long)
            x = self.p_sample(x, t_)

        # 3) 用解码器将最终的隐向量 x_0 解码成 (state, action)
        decoded = self.decoder.decode(x)
        state, action = decoded[:, :self.state_dim], decoded[:, self.state_dim:]# [batch_size, state_dim + action_dim]
        return state, action, decoded

    @torch.no_grad()
    def sample_batch(self, batch_size):
        """
        在隐空间中采样多个 x_0，并解码成 (state, action)。
        返回形状为 [batch_size, state_dim + action_dim] 的张量。
        """
        device = next(self.parameters()).device
        total_samples = batch_size 

        # 1) 在隐空间初始化随机噪声
        x = torch.randn((total_samples, self.embed_dim), device=device)

        # 2) 逐步去噪
        for t_ in tqdm(reversed(range(0, self.n_timesteps)), desc='采样进度'):
            t_ = torch.full((total_samples,), t_, device=device, dtype=torch.long)
            x = self.p_sample(x, t_)

        # 3) 解码
        decoded = self.decoder.decode(x)
        state, action = decoded[:, :self.state_dim], decoded[:, self.state_dim:]
        return state, action, decoded

    # -------------------------------------------------------------------------
    #                               训练相关
    # -------------------------------------------------------------------------
    def compute_loss(self, state, action, t):
        """
        扩散模型的训练损失由两部分组成：
        1) 扩散去噪损失: 预测噪声 epsilon 与真实噪声之间的 MSE
           或者预测 x_0 与真实 x_0 之间的 MSE
        2) 重构损失: 将编码后的 latent 用解码器还原到 (state, action)，并和 x_start 做对比
        """
        # 1) 编码到隐空间
        x_latent = self.encoder.encode(state, action)  # [batch_size, embed_dim]

        # 2) 为 x_latent 添加噪声，得到 x_noisy
        noise = torch.randn_like(x_latent)
        x_noisy = self.q_sample(x_latent, t, noise)

        # 3) 用 denoise_net 预测噪声或 x_0
        time_emb = self._time_embedding(t)
        model_input = torch.cat([x_noisy, time_emb], dim=-1)
        pred = self.denoise_net(model_input)  # [batch_size, embed_dim]

        # 4) 扩散损失
        target = noise if self.predict_epsilon else x_latent
        diffusion_loss = F.mse_loss(pred, target)

        # 5) 重构损失
        if self.predict_epsilon:
            # 通过预测的噪声恢复 x_0
            x_recon = self.predict_start_from_noise(x_noisy, t, pred)
        else:
            x_recon = pred
        decoded = self.decoder.decode(x_recon)  # [batch_size, state_dim + action_dim]
        recon_loss = F.mse_loss(decoded, torch.cat([state, action], dim=-1))

        # 6) 最终总损失
        total_loss = diffusion_loss + 0.1*recon_loss
        return total_loss



    def train_step(self, optimizer, state:torch.Tensor, action:torch.Tensor):
        """
        单步训练流程。
        states_actions: [batch_size, state_dim + action_dim]
        """
        batch_size = state.shape[0]
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=state.device)

        optimizer.zero_grad()
        loss = self.compute_loss(state, action, t)
        loss.backward()
        optimizer.step()
        tb_info = {
            "Generator/diffusion_loss": loss.item(),
        }
        return tb_info


# def train(model, train_dataloader, optimizer, n_epochs, device):
#     """
#     训练循环示例。
#     """
#     model.train()
#     for epoch in range(n_epochs):
#         epoch_loss = 0.0
#         for states_actions in train_dataloader:
#             states_actions = states_actions.to(device)
#             loss = train_step(model, optimizer, states_actions, device)
#             epoch_loss += loss

#         avg_loss = epoch_loss / len(train_dataloader)
#         print(f"Epoch {epoch+1}/{n_epochs}, Average Loss: {avg_loss:.4f}")

