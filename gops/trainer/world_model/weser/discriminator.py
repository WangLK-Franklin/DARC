import torch
import torch.nn as nn

class Dynamics_discriminator(nn.Module):
    def __init__(self, state_dim, action_dim, lr=1e-4):
        # 首先调用父类的初始化
        super(Dynamics_discriminator, self).__init__()
        
        input_dim = state_dim * 2 + action_dim
        
        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        # 动作编码器
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # 动力学关系处理网络
        self.dynamics_processor = nn.Sequential(
            nn.Linear(256 + 128 + 256, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )
        
        # 在所有网络层构建完成后再初始化优化器
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.BCELoss()
        
    def forward(self, state, action, next_state):
        # 分别编码状态和动作
        state_encoded = self.state_encoder(state)
        action_encoded = self.action_encoder(action)
        next_state_encoded = self.state_encoder(next_state)
        
        # 合并编码并处理动力学关系
        combined = torch.cat([state_encoded, action_encoded, next_state_encoded], dim=1)
        return self.dynamics_processor(combined)
    
    def update_step(self, real_batch, fake_batch):
        self.optimizer.zero_grad()
        
        # 解包数据
        real_state, real_action, real_next_state = real_batch
        fake_state, fake_action, fake_next_state = fake_batch
        
        batch_size = real_state.shape[0]
        device = real_state.device
        
        # 混合真实和生成的数据
        mixed_states = torch.cat([real_state, fake_state], dim=0)
        mixed_actions = torch.cat([real_action, fake_action], dim=0)
        mixed_next_states = torch.cat([real_next_state, fake_next_state], dim=0)
        
        # 创建对应的标签
        real_labels = torch.zeros((batch_size, 2), device=device)
        real_labels[:, 0] = 1  # [1,0] for real
        fake_labels = torch.zeros((batch_size, 2), device=device)
        fake_labels[:, 1] = 1  # [0,1] for fake
        mixed_labels = torch.cat([real_labels, fake_labels], dim=0)
        
        # 生成随机排列索引
        perm = torch.randperm(2 * batch_size, device=device)
        
        # 打乱数据和标签
        mixed_states = mixed_states[perm]
        mixed_actions = mixed_actions[perm]
        mixed_next_states = mixed_next_states[perm]
        mixed_labels = mixed_labels[perm]
        
        # 前向传播
        mixed_pred = self.forward(mixed_states, mixed_actions, mixed_next_states)
        loss = self.criterion(mixed_pred, mixed_labels)
        
        # 反向传播和优化
        loss.backward()
        self.optimizer.step()
        
        # 计算准确率 (需要根据标签来区分真实和生成样本)
        real_mask = mixed_labels[:, 0] == 1  # 找出真实样本的位置
        fake_mask = mixed_labels[:, 1] == 1  # 找出生成样本的位置
        
        real_acc = (mixed_pred[real_mask].argmax(dim=1) == 0).float().mean()
        fake_acc = (mixed_pred[fake_mask].argmax(dim=1) == 1).float().mean()
        
        return {
            'Dynamics_discriminator/total_loss': loss.item(),
            'Dynamics_discriminator/real_accuracy': real_acc.item(),
            'Dynamics_discriminator/fake_accuracy': fake_acc.item(),
            'Dynamics_discriminator/avg_accuracy': (real_acc + fake_acc).item() / 2
        }
class State_discriminator(nn.Module):
    def __init__(self, state_dim):
        super(State_discriminator, self).__init__()
        
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        self.discriminator = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, state):
        encoded = self.state_encoder(state)
        return self.discriminator(encoded)
    