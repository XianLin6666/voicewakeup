import torch
import librosa
import numpy as np
from model import CNNTransformer
import sounddevice as sd
import soundfile as sf

# 1. 加载模型
device = torch.device('cpu')
model = CNNTransformer().to(device)
model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
model.eval()

# 2. 录制自己的声音说 "yes"
print("请说 'yes' (录制3秒)...")
duration = 3  # 秒
fs = 16000
recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
sd.wait()
sf.write('my_yes.wav', recording, fs)

# 3. 预处理
audio, sr = librosa.load('my_yes.wav', sr=16000)
mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=40, n_fft=400, hop_length=160)
log_mel = librosa.power_to_db(mel)
# 固定长度
if log_mel.shape[1] > 100:
    log_mel = log_mel[:, :100]
else:
    pad = 100 - log_mel.shape[1]
    log_mel = np.pad(log_mel, ((0,0),(0,pad)), mode='constant')
# 归一化
log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
feature = torch.tensor(log_mel).unsqueeze(0).unsqueeze(0)  # (1,1,40,100)

# 4. 预测
with torch.no_grad():
    output = model(feature)
    prob = torch.softmax(output, dim=1)
    pred = output.argmax(dim=1).item()
    print(f"预测结果：{'唤醒词 (yes)' if pred==1 else '非唤醒词'}")
    print(f"唤醒概率：{prob[0,1].item():.2%}")