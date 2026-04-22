import torch
from model import CNNTransformer

device = torch.device('cpu')
model = CNNTransformer().to(device)
# 加载模型，忽略 weights_only 警告（可加 weights_only=False 显式）
model.load_state_dict(torch.load('best_model.pth', map_location='cpu', weights_only=False))
model.eval()

dummy_input = torch.randn(1, 1, 40, 100)

torch.onnx.export(
    model,
    dummy_input,
    'wakeword_model.onnx',
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    opset_version=14  # 改为14
)
print("ONNX模型导出成功：wakeword_model.onnx")