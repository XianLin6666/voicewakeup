import onnxruntime as ort
import numpy as np
import torch
from dataset import SpeechCommandsDataset
from torch.utils.data import DataLoader

# 加载 ONNX 模型
ort_session = ort.InferenceSession('wakeword_model.onnx')

# 从验证集取一个样本测试
val_dataset = SpeechCommandsDataset('val_files.txt', augment=False)
sample_feat, sample_label = val_dataset[0]
# 增加 batch 维度
input_numpy = sample_feat.unsqueeze(0).numpy()  # (1,1,40,100)

# ONNX 推理
outputs = ort_session.run(['output'], {'input': input_numpy})
pred = np.argmax(outputs[0], axis=1)
print(f"真实标签：{sample_label}，ONNX 预测：{pred[0]}")