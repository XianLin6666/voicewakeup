import os
import glob
import random

# 数据集路径（请根据你的实际解压路径修改）
data_dir = './data/speech_commands_v0.02'  # 如果解压在项目根目录的data下
wake_word = 'yes'  # 定义唤醒词
# 选择一些非唤醒词作为负样本（可以包含所有其他词，但为了平衡，先选一部分）
non_wake_words = ['no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
background_noise = '_background_noise_'  # 背景噪声文件夹

def get_file_list_and_labels():
    file_paths = []
    labels = []
    
    # 遍历data_dir下的所有子文件夹
    for cls_name in os.listdir(data_dir):
        cls_path = os.path.join(data_dir, cls_name)
        if not os.path.isdir(cls_path):
            continue
        # 获取该类别下所有wav文件
        wav_files = glob.glob(os.path.join(cls_path, '*.wav'))
        # 确定标签
        if cls_name == wake_word:
            label = 1
        elif cls_name in non_wake_words or cls_name == background_noise:
            label = 0
        else:
            # 其他类别暂时忽略（可根据需要添加）
            continue
        
        for wav in wav_files:
            file_paths.append(wav)
            labels.append(label)
    
    return file_paths, labels

if __name__ == '__main__':
    files, labels = get_file_list_and_labels()
    print(f"正样本（唤醒词）数量：{sum(labels)}")
    print(f"负样本数量：{len(labels)-sum(labels)}")
    
    # 随机打乱并划分训练/验证集（8:2）
    data = list(zip(files, labels))
    random.shuffle(data)
    split = int(0.8 * len(data))
    train_data = data[:split]
    val_data = data[split:]
    
    # 保存文件列表，供Dataset使用
    with open('train_files.txt', 'w') as f:
        for path, lbl in train_data:
            f.write(f"{path}\t{lbl}\n")
    with open('val_files.txt', 'w') as f:
        for path, lbl in val_data:
            f.write(f"{path}\t{lbl}\n")
    
    print(f"训练集样本数：{len(train_data)}，验证集样本数：{len(val_data)}")