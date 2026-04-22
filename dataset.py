import torch
from torch.utils.data import Dataset
import librosa
import numpy as np
import os

class SpeechCommandsDataset(Dataset):
    def __init__(self, file_list_path, augment=False, fixed_length=100):
        """
        file_list_path: 包含"音频路径\t标签"的文本文件
        augment: 是否进行数据增强（用于训练集）
        fixed_length: 固定时间帧数（对应约1秒，因为10ms一帧）
        """
        self.file_paths = []
        self.labels = []
        with open(file_list_path, 'r', encoding='utf-8') as f:
            for line in f:
                path, label = line.strip().split('\t')
                # 确保路径存在（相对路径可以转为绝对路径，但我们的文件列表已是绝对路径）
                self.file_paths.append(path)
                self.labels.append(int(label))
        self.augment = augment
        self.fixed_length = fixed_length
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # 加载音频
        audio_path = self.file_paths[idx]
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # --- 数据增强（仅训练集） ---
        if self.augment:
            # 1. 添加高斯白噪声（概率0.5，噪声水平随机）
            if np.random.rand() > 0.5:
                noise_level = np.random.uniform(0, 0.01)
                noise = np.random.randn(len(audio)) * noise_level
                audio = audio + noise
            # 2. 时间平移（概率0.5，最多平移±0.1秒）
            if np.random.rand() > 0.5:
                shift = np.random.randint(-1600, 1601)  # ±0.1秒
                audio = np.roll(audio, shift)
                if shift > 0:
                    audio[:shift] = 0
                else:
                    audio[shift:] = 0
        
        # --- 提取 Mel 频谱 ---
        mel = librosa.feature.melspectrogram(
            y=audio, sr=sr,
            n_mels=40,
            n_fft=400,
            hop_length=160
        )
        log_mel = librosa.power_to_db(mel)  # (40, T)
        
        # --- 固定长度处理（裁剪或填充） ---
        current_len = log_mel.shape[1]
        if current_len > self.fixed_length:
            # 随机裁剪一段
            start = np.random.randint(0, current_len - self.fixed_length + 1)
            log_mel = log_mel[:, start:start+self.fixed_length]
        elif current_len < self.fixed_length:
            # 填充0（右边填充）
            pad_width = self.fixed_length - current_len
            log_mel = np.pad(log_mel, ((0,0), (0,pad_width)), mode='constant')
        
        # --- 归一化（每个样本独立z-score） ---
        mean = log_mel.mean()
        std = log_mel.std()
        if std > 0:
            log_mel = (log_mel - mean) / std
        else:
            log_mel = log_mel - mean  # 避免除以0
        
        # --- 转换为Tensor并添加通道维度 (C=1) ---
        feature = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0)  # (1, 40, fixed_length)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return feature, label