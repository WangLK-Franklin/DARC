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
            
            nn.Linear(512, 384),
            nn.LayerNorm(384),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(384, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )
        
        # 在所有网络层构建完成后再初始化优化器
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        
    def forward(self, state, action, next_state):
        # 分别编码状态和动作
        state_encoded = self.state_encoder(state)
        action_encoded = self.action_encoder(action)
        next_state_encoded = self.state_encoder(next_state)
        
        # 合并编码并处理动力学关系
        combined = torch.cat([state_encoded, action_encoded, next_state_encoded], dim=1)
        return self.dynamics_processor(combined)
    
    def update_step(self, real_batch, fake_batch):
        """
        执行一步判别器更新
        
        Args:
            real_batch: 元组 (state, action, next_state) 包含真实轨迹
            fake_batch: 元组 (state, action, next_state) 包含生成轨迹
        
        Returns:
            dict: 包含损失和准确率信息
        """
        self.optimizer.zero_grad()
        
        # 解包数据
        real_state, real_action, real_next_state = real_batch
        fake_state, fake_action, fake_next_state = fake_batch
        
        batch_size = real_state.shape[0]
        device = real_state.device
        
        # 创建标签
        real_labels = torch.ones(batch_size, dtype=torch.long, device=device)
        fake_labels = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # 真实数据的前向传播和损失计算
        real_pred = self.forward(real_state, real_action, real_next_state)
        real_loss = self.criterion(real_pred, real_labels)
        
        # 生成数据的前向传播和损失计算
        fake_pred = self.forward(fake_state, fake_action, fake_next_state)
        fake_loss = self.criterion(fake_pred, fake_labels)
        
        # 总损失
        total_loss = real_loss + fake_loss
        
        # 反向传播和优化
        total_loss.backward()
        self.optimizer.step()
        
        # 计算准确率
        real_acc = (real_pred.argmax(dim=1) == real_labels).float().mean()
        fake_acc = (fake_pred.argmax(dim=1) == fake_labels).float().mean()
        
        return {
            'total_loss': total_loss.item(),
            'real_loss': real_loss.item(),
            'fake_loss': fake_loss.item(),
            'real_accuracy': real_acc.item(),
            'fake_accuracy': fake_acc.item(),
            'avg_accuracy': (real_acc + fake_acc).item() / 2
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
    