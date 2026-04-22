import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import SpeechCommandsDataset
from model import CNNTransformer
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 超参数设置
batch_size = 32
epochs = 20
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备：{device}")

# 数据加载
train_dataset = SpeechCommandsDataset('train_files.txt', augment=True)
val_dataset = SpeechCommandsDataset('val_files.txt', augment=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# 初始化模型、损失函数、优化器
model = CNNTransformer().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 记录训练过程
train_losses = []
train_accs = []
val_losses = []
val_accs = []
best_val_acc = 0.0

for epoch in range(1, epochs + 1):
    # 训练阶段
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{epochs} [Train]')
    for features, labels in pbar:
        features, labels = features.to(device), labels.to(device)
        
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * features.size(0)
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'Loss': f'{train_loss/train_total:.4f}',
            'Acc': f'{100.*train_correct/train_total:.2f}%'
        })
    
    epoch_train_loss = train_loss / train_total
    epoch_train_acc = 100. * train_correct / train_total
    train_losses.append(epoch_train_loss)
    train_accs.append(epoch_train_acc)
    
    # 验证阶段
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for features, labels in tqdm(val_loader, desc=f'Epoch {epoch}/{epochs} [Val]'):
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * features.size(0)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    
    epoch_val_loss = val_loss / val_total
    epoch_val_acc = 100. * val_correct / val_total
    val_losses.append(epoch_val_loss)
    val_accs.append(epoch_val_acc)
    
    print(f'Epoch {epoch}: Train Acc: {epoch_train_acc:.2f}%, Val Acc: {epoch_val_acc:.2f}%')
    
    # 保存最佳模型
    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        torch.save(model.state_dict(), 'best_model.pth')
        print(f'保存最佳模型，验证准确率: {epoch_val_acc:.2f}%')

print("训练完成！最佳验证准确率：{:.2f}%".format(best_val_acc))

# ========== 训练结束后自动绘图 ==========
plt.figure(figsize=(12, 4))

# 损失曲线
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), train_losses, 'b-', label='训练损失')
plt.plot(range(1, epochs+1), val_losses, 'r-', label='验证损失')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('训练和验证损失曲线')
plt.legend()
plt.grid(True)

# 准确率曲线
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), train_accs, 'b-', label='训练准确率')
plt.plot(range(1, epochs+1), val_accs, 'r-', label='验证准确率')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('训练和验证准确率曲线')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
plt.show()
print("训练曲线已保存为 training_curves.png")

# ========== 绘制混淆矩阵 ==========
# 在验证集上收集所有预测
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for features, labels in val_loader:
        features = features.to(device)
        outputs = model(features)
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# 计算混淆矩阵
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['非唤醒', '唤醒'])
disp.plot(cmap='Blues')
plt.title('验证集混淆矩阵')
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()
print("混淆矩阵已保存为 confusion_matrix.png")