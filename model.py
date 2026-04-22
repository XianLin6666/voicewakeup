import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNTransformer(nn.Module):
    def __init__(self, n_mels=40, time_steps=100, d_model=64, nhead=4, num_layers=1, num_classes=2):
        """
        n_mels: Mel 频谱维度（40）
        time_steps: 固定时间帧数（100）
        d_model: Transformer 输入维度
        nhead: 注意力头数
        num_layers: Transformer 编码器层数
        num_classes: 分类数（唤醒/非唤醒）
        """
        super().__init__()
        
        # CNN 前端：提取局部特征并降维
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2)  # 2倍降采样
        
        # 经过 CNN 后的特征尺寸计算：
        # 输入 (batch, 1, 40, 100) -> 一次池化后 (batch, 16, 20, 50) -> 第二次卷积后 (batch, 32, 20, 50)（无池化）
        # 所以我们使用一个池化，将频率维度减半（40→20），时间维度减半（100→50）
        # 最终每个时间步的特征维度 = 32 * 20 = 640
        self.cnn_output_dim = 32 * (n_mels // 2)
        self.seq_len = time_steps // 2  # 50
        
        # 线性投影：将 640 维映射到 d_model（64）
        self.projection = nn.Linear(self.cnn_output_dim, d_model)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            activation='relu',
            batch_first=False  # PyTorch Transformer 默认输入格式为 (seq_len, batch, d_model)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 分类头
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        """
        x: (batch, 1, 40, 100)
        """
        # CNN 部分
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # (batch, 16, 20, 50)
        x = F.relu(self.bn2(self.conv2(x)))               # (batch, 32, 20, 50)
        
        # 重整维度以适应 Transformer
        batch, ch, freq, time = x.shape
        x = x.permute(0, 3, 1, 2)          # (batch, time, ch, freq)
        x = x.reshape(batch, time, ch * freq)  # (batch, time, 640)
        
        # 线性投影到 d_model
        x = self.projection(x)              # (batch, time, d_model)
        
        # Transformer 期望输入 (time, batch, d_model)
        x = x.permute(1, 0, 2)              # (time, batch, d_model)
        x = self.transformer(x)              # (time, batch, d_model)
        
        # 取所有时间步的平均作为序列表示（也可取最后一个时间步，但平均更稳定）
        x = x.mean(dim=0)                    # (batch, d_model)
        
        # 分类
        out = self.classifier(x)             # (batch, num_classes)
        return out

# 快速测试模型（可选）
if __name__ == '__main__':
    model = CNNTransformer()
    dummy_input = torch.randn(2, 1, 40, 100)  # batch=2
    output = model(dummy_input)
    print(f"模型输出形状：{output.shape}")  # 应为 (2, 2)