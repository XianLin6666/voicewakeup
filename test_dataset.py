from dataset import SpeechCommandsDataset
from torch.utils.data import DataLoader

# 测试训练集（开启数据增强）
train_dataset = SpeechCommandsDataset('train_files.txt', augment=True)
print(f"训练集样本数：{len(train_dataset)}")

# 取一个样本看看
feature, label = train_dataset[0]
print(f"特征形状：{feature.shape}，标签：{label}")

# 测试 DataLoader
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
for batch_feat, batch_label in train_loader:
    print(f"Batch 特征形状：{batch_feat.shape}，标签：{batch_label}")
    break