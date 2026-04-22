import torch
from torch.utils.data import DataLoader
from dataset import SpeechCommandsDataset
from model import CNNTransformer

device = torch.device('cpu')
model = CNNTransformer().to(device)
model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
model.eval()

val_dataset = SpeechCommandsDataset('val_files.txt', augment=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

correct = 0
total = 0
with torch.no_grad():
    for features, labels in val_loader:
        outputs = model(features)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

print(f'验证集准确率：{100.*correct/total:.2f}%')