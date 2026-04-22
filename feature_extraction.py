import librosa
import numpy as np
import matplotlib.pyplot as plt

def load_audio(file_path, target_sr=16000):
    """
    加载音频文件，并统一采样率为 target_sr（16kHz）
    """
    # librosa.load 会自动重采样到 target_sr
    audio, sr = librosa.load(file_path, sr=target_sr)
    return audio, sr

def extract_mel_spectrogram(audio, sr=16000, n_mels=40, n_fft=400, hop_length=160):
    """
    从音频波形提取 Mel 频谱（对数刻度）
    参数：
        audio: 音频波形数组
        sr: 采样率
        n_mels: Mel 滤波器个数（特征维度）
        n_fft: FFT 窗口大小（对应 25ms，因为 16000*0.025=400）
        hop_length: 帧移（对应 10ms，因为 16000*0.01=160）
    返回：
        log_mel: 对数 Mel 频谱，形状 (n_mels, time_steps)
    """
    # 计算 Mel 频谱（功率谱）
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )
    # 转换为对数刻度（分贝），更符合人耳感知，也方便模型训练
    log_mel = librosa.power_to_db(mel_spec)
    return log_mel

def plot_mel_spectrogram(mel_spec, sr=16000, hop_length=160, title='Mel Spectrogram'):
    """
    可视化 Mel 频谱
    """
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        mel_spec, sr=sr, hop_length=hop_length,
        x_axis='time', y_axis='mel'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.savefig('mel_spectrogram.png', dpi=150, bbox_inches='tight')
    plt.show()

# 测试代码（当直接运行此文件时执行）
if __name__ == "__main__":
    # 你需要先准备一个测试音频文件，可以从网上下载一个短的 wav 文件
    # 或者我们先用 librosa 自带的示例音频（一个女声说 "example"）
    print("正在加载示例音频...")
    # librosa 提供了一个示例音频文件（路径会根据系统变化，但肯定存在）
    example_file = librosa.example('nutcracker')  # 这是一个音乐片段，但没关系
    audio, sr = load_audio(example_file)
    print(f"音频采样率: {sr}, 时长: {len(audio)/sr:.2f}秒")
    
    # 提取 Mel 频谱
    mel = extract_mel_spectrogram(audio)
    print(f"Mel 频谱形状: {mel.shape}")  # 应该是 (40, T)
    
    # 可视化
    plot_mel_spectrogram(mel, title='Example Audio Mel Spectrogram')