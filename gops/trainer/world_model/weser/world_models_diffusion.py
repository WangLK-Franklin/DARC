import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical, Normal
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from torch.cuda.amp import autocast

from gops.trainer.world_model.weser.functions_losses import SymLogTwoHotLoss
from gops.trainer.world_model.weser.attention_blocks import get_subsequent_mask_with_batch_length, get_subsequent_mask
from gops.trainer.world_model.weser.transformer_model import StochasticTransformerKVCache
from gops.utils.tensorboard_setup import tb_tags

class StateActionEmb(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim=256):
        super().__init__()
        self.input_dim = state_dim + action_dim
        self.latent_dim = latent_dim
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128), 
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, self.input_dim)
        )
        
    def encode(self, states, actions):
        
        x = torch.cat([states, actions], dim=-1)
        return self.encoder(x)
        
    def decode(self, z):
        
        return self.decoder(z)
        
    def forward(self, states, actions):
        L = states.shape[1]
        states = rearrange(states, "B L C -> (B L) C")
        actions = rearrange(actions, "B L C -> (B L) C")
        z = self.encode(states, actions)
        recon = self.decode(z)
        output = rearrange(z, "(B L) C -> B L C", L=L)
        recon = rearrange(recon, "(B L) C -> B L C", L=L)
        return output, recon
    
    
class EncoderBN(nn.Module):
    def __init__(self, in_channels, stem_channels, final_feature_width) -> None:
        super().__init__()

        backbone = []
        # stem
        backbone.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=stem_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            )
        )
        feature_width = 64//2
        channels = stem_channels
        backbone.append(nn.BatchNorm2d(stem_channels))
        backbone.append(nn.ReLU(inplace=True))

        # layers
        while True:
            backbone.append(
                nn.Conv2d(
                    in_channels=channels,
                    out_channels=channels*2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False
                )
            )
            channels *= 2
            feature_width //= 2
            backbone.append(nn.BatchNorm2d(channels))
            backbone.append(nn.ReLU(inplace=True))

            if feature_width == final_feature_width:
                break

        self.backbone = nn.Sequential(*backbone)
        self.last_channels = channels

    def forward(self, x):
        batch_size = x.shape[0]
        x = rearrange(x, "B L C H W -> (B L) C H W")
        x = self.backbone(x)
        x = rearrange(x, "(B L) C H W -> B L (C H W)", B=batch_size)
        return x


class DecoderBN(nn.Module):
    def __init__(self, stoch_dim, last_channels, original_in_channels, stem_channels, final_feature_width) -> None:
        super().__init__()

        backbone = []
        # stem
        backbone.append(nn.Linear(stoch_dim, last_channels*final_feature_width*final_feature_width, bias=False))
        backbone.append(Rearrange('B L (C H W) -> (B L) C H W', C=last_channels, H=final_feature_width))
        backbone.append(nn.BatchNorm2d(last_channels))
        backbone.append(nn.ReLU(inplace=True))
        # residual_layer
        # backbone.append(ResidualStack(last_channels, 1, last_channels//4))
        # layers
        channels = last_channels
        feat_width = final_feature_width
        while True:
            if channels == stem_channels:
                break
            backbone.append(
                nn.ConvTranspose2d(
                    in_channels=channels,
                    out_channels=channels//2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False
                )
            )
            channels //= 2
            feat_width *= 2
            backbone.append(nn.BatchNorm2d(channels))
            backbone.append(nn.ReLU(inplace=True))

        backbone.append(
            nn.ConvTranspose2d(
                in_channels=channels,
                out_channels=original_in_channels,
                kernel_size=4,
                stride=2,
                padding=1
            )
        )
        self.backbone = nn.Sequential(*backbone)

    def forward(self, sample):
        batch_size = sample.shape[0]
        obs_hat = self.backbone(sample)
        obs_hat = rearrange(obs_hat, "(B L) C H W -> B L C H W", B=batch_size)
        return obs_hat


class DistHead(nn.Module):
    '''
    Dist: abbreviation of distribution
    '''
    def __init__(self, transformer_hidden_dim, state_dim) -> None:
        super().__init__()
        self.state_dim = state_dim
        # self.post_head = nn.Linear(image_feat_dim, stoch_dim*stoch_dim)
        self.prior_head = nn.Linear(transformer_hidden_dim, state_dim)

    def unimix(self, logits, mixing_ratio=0.01):
        # uniform noise mixing
        probs = F.softmax(logits, dim=-1)
        mixed_probs = mixing_ratio * torch.ones_like(probs) / self.state_dim + (1-mixing_ratio) * probs
        logits = torch.log(mixed_probs)
        return logits

    # def forward_post(self, x):
    #     logits = self.post_head(x)
    #     logits = rearrange(logits, "B L (K C) -> B L K C", K=self.stoch_dim)
    #     logits = self.unimix(logits)
    #     return logits

    def forward_prior(self, x):
        logits = self.prior_head(x)
        # logits = rearrange(logits, "B L (K C) -> B L K C", K=self.stoch_dim)
        # logits = self.unimix(logits)
        return logits

class RewardDecoderContinuous(nn.Module):
    def __init__(self, embedding_size, transformer_hidden_dim) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(transformer_hidden_dim, transformer_hidden_dim, bias=False),
            nn.LayerNorm(transformer_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(transformer_hidden_dim, transformer_hidden_dim, bias=False),
            nn.LayerNorm(transformer_hidden_dim),
            nn.ReLU(inplace=True),
        )
        # 改为输出单个连续值
        self.head = nn.Linear(transformer_hidden_dim, 1)

    def forward(self, feat):
        feat = self.backbone(feat)
        # 移除了分类层，直接输出连续值
        reward = self.head(feat).squeeze(-1)  # 移除最后一维
        return reward

class RewardDecoder(nn.Module):
    def __init__(self, num_classes, embedding_size, transformer_hidden_dim) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(transformer_hidden_dim, transformer_hidden_dim, bias=False),
            nn.LayerNorm(transformer_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(transformer_hidden_dim, transformer_hidden_dim, bias=False),
            nn.LayerNorm(transformer_hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(transformer_hidden_dim, num_classes)

    def forward(self, feat):
        feat = self.backbone(feat)
        reward = self.head(feat)
        return reward


class TerminationDecoder(nn.Module):
    def __init__(self,  embedding_size, transformer_hidden_dim) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(transformer_hidden_dim, transformer_hidden_dim, bias=False),
            nn.LayerNorm(transformer_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(transformer_hidden_dim, transformer_hidden_dim, bias=False),
            nn.LayerNorm(transformer_hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(transformer_hidden_dim, 1),
            # nn.Sigmoid()
        )

    def forward(self, feat):
        feat = self.backbone(feat)
        termination = self.head(feat)
        termination = termination.squeeze(-1)  # remove last 1 dim
        return termination


class MSELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, obs_hat, obs):
        loss = (obs_hat - obs)**2
        # loss = reduce(loss, "B L C H W -> B L", "sum")
        return loss.mean()


class CategoricalKLDivLossWithFreeBits(nn.Module):
    def __init__(self, free_bits) -> None:
        super().__init__()
        self.free_bits = free_bits

    def forward(self, p_logits, q_logits):
        p_dist = OneHotCategorical(logits=p_logits)
        q_dist = OneHotCategorical(logits=q_logits)
        kl_div = torch.distributions.kl.kl_divergence(p_dist, q_dist)
        kl_div = reduce(kl_div, "B L D -> B L", "sum")
        kl_div = kl_div.mean()
        real_kl_div = kl_div
        kl_div = torch.max(torch.ones_like(kl_div)*self.free_bits, kl_div)
        return kl_div, real_kl_div


class WorldModel(nn.Module):
    def __init__(self, state_dim, action_dim,latent_dim,
                 transformer_max_length, transformer_hidden_dim, transformer_num_layers, transformer_num_heads):
        super().__init__()
        
        self.transformer_hidden_dim = transformer_hidden_dim
        self.final_feature_width = 4
        self.stoch_dim = 32
        self.latent_dim = latent_dim    
        self.stoch_flattened_dim = state_dim
        self.use_amp = True
        self.tensor_dtype = torch.bfloat16 if self.use_amp else torch.float32
        self.imagine_batch_size = -1
        self.imagine_batch_length = -1
        self.action_dim = action_dim
        # self.encoder = EncoderBN(
        #     in_channels=in_channels,
        #     stem_channels=32,
        #     final_feature_width=self.final_feature_width
        # )
        self.storm_transformer = StochasticTransformerKVCache(
            latent_dim=self.latent_dim,
            feat_dim=transformer_hidden_dim,
            num_layers=transformer_num_layers,
            num_heads=transformer_num_heads,
            max_length=transformer_max_length,
            dropout=0.1
        )
        self.dist_head = DistHead(
            transformer_hidden_dim=transformer_hidden_dim,
            state_dim=self.stoch_flattened_dim
        )
        self.state_action_emb = StateActionEmb(
            state_dim=state_dim,
            action_dim=action_dim,
            latent_dim=self.latent_dim
        )
        # self.image_decoder = DecoderBN(
        #     stoch_dim=self.stoch_flattened_dim,
        #     last_channels=self.encoder.last_channels,
        #     original_in_channels=in_channels,
        #     stem_channels=32,
        #     final_feature_width=self.final_feature_width
        # )
        self.reward_decoder = RewardDecoderContinuous(

            embedding_size=self.stoch_flattened_dim,
            transformer_hidden_dim=transformer_hidden_dim
        )
        self.termination_decoder = TerminationDecoder(
            embedding_size=self.stoch_flattened_dim,
            transformer_hidden_dim=transformer_hidden_dim
        )

        self.mse_loss_func = MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_with_logits_loss_func = nn.BCEWithLogitsLoss()
        self.symlog_twohot_loss_func = SymLogTwoHotLoss(num_classes=255, lower_bound=-20, upper_bound=20)
        self.categorical_kl_div_loss = CategoricalKLDivLossWithFreeBits(free_bits=1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

    def encode_obs(self, obs):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            embedding = self.encoder(obs)
            post_logits = self.dist_head.forward_post(embedding)
            sample = self.stright_throught_gradient(post_logits, sample_mode="random_sample")
            flattened_sample = self.flatten_sample(sample)
        return flattened_sample

    def calc_last_dist_feat(self, latent, action):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            temporal_mask = get_subsequent_mask(latent)
            dist_feat = self.storm_transformer(latent, action, temporal_mask)
            last_dist_feat = dist_feat[:, -1:]
            prior_logits = self.dist_head.forward_prior(last_dist_feat)
            prior_sample = self.stright_throught_gradient(prior_logits, sample_mode="random_sample")
            prior_flattened_sample = self.flatten_sample(prior_sample)
        return prior_flattened_sample, last_dist_feat

    def predict_next(self, state, action):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            latent,_ = self.state_action_emb(state, action)
            
            dist_feat = self.storm_transformer.forward_with_kv_cache(latent)

            next_obs = self.dist_head.forward_prior(dist_feat)
            
            reward_hat = self.reward_decoder(dist_feat)
            
            termination_hat = self.termination_decoder(dist_feat)
            termination_hat = termination_hat > 0

        return next_obs, reward_hat, termination_hat, dist_feat

    def stright_throught_gradient(self, logits, sample_mode="random_sample"):
        dist = OneHotCategorical(logits=logits)
        if sample_mode == "random_sample":
            sample = dist.sample() + dist.probs - dist.probs.detach()
        elif sample_mode == "mode":
            sample = dist.mode
        elif sample_mode == "probs":
            sample = dist.probs
        return sample

    def flatten_sample(self, sample):
        return rearrange(sample, "B L K C -> B L (K C)")

    def init_imagine_buffer(self, imagine_batch_size, imagine_batch_length, dtype):
        '''
        This can slightly improve the efficiency of imagine_data
        But may vary across different machines
        '''
        if self.imagine_batch_size != imagine_batch_size or self.imagine_batch_length != imagine_batch_length:
            print(f"init_imagine_buffer: {imagine_batch_size}x{imagine_batch_length}@{dtype}")
            self.imagine_batch_size = imagine_batch_size
            self.imagine_batch_length = imagine_batch_length
            latent_size = (imagine_batch_size, imagine_batch_length+1, self.stoch_flattened_dim)
            hidden_size = (imagine_batch_size, imagine_batch_length+1, self.transformer_hidden_dim)
            action_size = (imagine_batch_size, imagine_batch_length, self.action_dim)
            scalar_size = (imagine_batch_size, imagine_batch_length)
            self.state_buffer = torch.zeros(latent_size, dtype=dtype, device="cuda")
            self.hidden_buffer = torch.zeros(hidden_size, dtype=dtype, device="cuda")
            self.action_buffer = torch.zeros(action_size, dtype=dtype, device="cuda")
            self.reward_hat_buffer = torch.zeros(scalar_size, dtype=dtype, device="cuda")
            self.termination_hat_buffer = torch.zeros(scalar_size, dtype=dtype, device="cuda")


    def imagine_data(self, agent, sample_obs, sample_action,
                     imagine_batch_size, imagine_batch_length):
        self.init_imagine_buffer(int(imagine_batch_size/imagine_batch_length), imagine_batch_length, dtype=self.tensor_dtype)
        obs_hat_list = []

        self.storm_transformer.reset_kv_cache_list(int(imagine_batch_size/imagine_batch_length), dtype=self.tensor_dtype)
        # context
        sample_obs = sample_obs.unsqueeze(1)
        sample_action = sample_action.unsqueeze(1)
        context_latent = sample_obs
        for i in range(sample_obs.shape[1]):  # context_length is sample_obs.shape[1]
             last_state, last_reward_hat, last_termination_hat, last_dist_feat = self.predict_next(
                context_latent[:, i:i+1],
                sample_action[:, i:i+1],
            )
        self.state_buffer[:, 0:1] = last_state
        self.hidden_buffer[:, 0:1] = last_dist_feat

        # imagine
        for i in range(imagine_batch_length):
            logits = agent.policy(self.state_buffer[:, i:i+1].float())
            action_distribution = agent.create_action_distributions(logits)
            action, logp = action_distribution.sample()
            action = action.to(device=self.state_buffer.device, dtype=self.state_buffer.dtype)
            self.action_buffer[:, i:i+1] = action

            last_state, last_reward_hat, last_termination_hat, last_dist_feat = self.predict_next(
                self.state_buffer[:, i:i+1], self.action_buffer[:, i:i+1])

         
            self.state_buffer[:, i+1:i+2] = last_state
            self.hidden_buffer[:, i+1:i+2] = last_dist_feat
            self.reward_hat_buffer[:, i:i+1] = last_reward_hat
            self.termination_hat_buffer[:, i:i+1] = last_termination_hat
            
        return self.state_buffer, self.action_buffer, self.reward_hat_buffer, self.termination_hat_buffer
        
    def update(self, obs, action, reward, termination, logger=None):
        self.train()
        batch_size, batch_length = obs.shape[:2]

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            # encoding

            # transformer
            temporal_mask = get_subsequent_mask_with_batch_length(batch_length, obs.device)
            latent,obs_hat = self.state_action_emb(obs, action)
            dist_feat = self.storm_transformer(latent, temporal_mask)
            prior_logits = self.dist_head.forward_prior(dist_feat)
            # decoding reward and termination with dist_feat
            reward_hat = self.reward_decoder(dist_feat)
            termination_hat = self.termination_decoder(dist_feat)

            # env loss
            representation_loss = self.mse_loss_func(obs_hat, torch.cat([obs,action],dim=-1))
            reward_loss = self.mse_loss_func(reward_hat, reward)
            termination_loss = self.bce_with_logits_loss_func(termination_hat, termination)
            # dyn-rep loss
            dynamics_loss = self.mse_loss_func(obs[:, 1:].detach(), prior_logits[:, :-1])

            total_loss = representation_loss + 5.0*reward_loss + termination_loss + dynamics_loss

        # gradient descent
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)  # for clip grad
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1000.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        
        tb_info = {
            "World_model/reward_loss": reward_loss.item(),
            "World_model/termination_loss": termination_loss.item(),
            "World_model/dynamics_loss": dynamics_loss.item(),
            "World_model/total_loss": total_loss.item(),
            "World_model/representation_loss": representation_loss.item(),
        }
        return tb_info