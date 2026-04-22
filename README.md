# 轻量级语音唤醒系统（Wake Word Detection）

基于 `PyTorch` 实现的二分类语音唤醒项目。  
当前实现以 **`yes`** 作为唤醒词（正样本），其余指定词和背景噪声作为非唤醒（负样本），模型结构为 **CNN + Transformer**，并支持导出 `ONNX` 进行部署验证。

## 项目亮点

- 端到端流程完整：数据整理、特征提取、训练、评估、导出、推理验证
- 特征工程清晰：`16kHz`、`40-dim log-Mel`、固定 `100` 帧
- 模型轻量：适合后续边缘设备部署与优化
- 提供多种脚本：本地录音测试、ONNX 推理测试、训练曲线与混淆矩阵可视化

## 项目结构

```text
voicewakeup/
├─ prepare_data.py          # 扫描数据集并生成 train_files.txt / val_files.txt
├─ dataset.py               # Dataset：加载音频 + 增强 + Mel 特征 + 归一化
├─ model.py                 # CNNTransformer 模型定义
├─ train.py                 # 训练主脚本，保存 best_model.pth，并绘图
├─ evaluate.py              # 在验证集上评估准确率
├─ export_onnx.py           # 导出 ONNX 模型
├─ test_onnx.py             # ONNX Runtime 推理验证
├─ test_my_voice.py         # 录音并做本地推理
├─ feature_extraction.py    # Mel 特征提取与可视化示例
├─ train_files.txt          # 训练集样本列表（路径\t标签）
├─ val_files.txt            # 验证集样本列表（路径\t标签）
├─ best_model.pth           # 训练得到的最优权重
├─ wakeword_model.onnx      # 导出的 ONNX 模型
└─ data/                    # Speech Commands 数据目录（已被 .gitignore 忽略）
```

## 环境要求

- Python 3.8+（推荐 3.8~3.11）
- Windows / Linux / macOS 均可
- 有 GPU 则自动使用 GPU（`train.py` 中自动检测）

## 安装依赖

```bash
pip install torch torchaudio librosa numpy matplotlib tqdm scikit-learn onnx onnxruntime sounddevice soundfile
```

如果你使用 Conda，也可以：

```bash
conda create -n wakeword python=3.10 -y
conda activate wakeword
pip install torch torchaudio librosa numpy matplotlib tqdm scikit-learn onnx onnxruntime sounddevice soundfile
```

## 数据准备

### 1. 下载并放置数据集

使用 Google Speech Commands V2（`speech_commands_v0.02`），目录放到：

```text
./data/speech_commands_v0.02
```

`prepare_data.py` 默认配置：

- 唤醒词：`yes`（标签 1）
- 非唤醒词：`no/up/down/left/right/on/off/stop/go` + `_background_noise_`（标签 0）

### 2. 生成训练/验证文件列表

```bash
python prepare_data.py
```

会在根目录生成：

- `train_files.txt`
- `val_files.txt`

每行格式为：`音频路径\t标签`

## 训练模型

```bash
python train.py
```

默认超参数（见 `train.py`）：

- `batch_size=32`
- `epochs=20`
- `learning_rate=0.001`

训练输出：

- `best_model.pth`（验证集最佳模型）
- `training_curves.png`（loss/acc 曲线）
- `confusion_matrix.png`（验证集混淆矩阵）

## 评估模型

```bash
python evaluate.py
```

脚本会加载 `best_model.pth` 并输出验证集准确率。

## 导出 ONNX 与推理验证

### 导出

```bash
python export_onnx.py
```

生成：

- `wakeword_model.onnx`

### 验证 ONNX 推理

```bash
python test_onnx.py
```

脚本会从验证集中取样本，对比真实标签与 ONNX 预测结果。

## 本地录音测试

```bash
python test_my_voice.py
```

脚本会：

1. 录制约 3 秒音频（`my_yes.wav`）
2. 提取同训练流程一致的 log-Mel 特征
3. 使用 `best_model.pth` 做二分类预测并输出唤醒概率

## 模型与特征说明

- 输入特征：`(1, 40, 100)`（通道, Mel 维, 时间帧）
- CNN 前端：
  - `Conv2d(1->16) + BN + ReLU + MaxPool2d(2)`
  - `Conv2d(16->32) + BN + ReLU`
- Transformer 编码：
  - `d_model=64`
  - `nhead=4`
  - `num_layers=1`
- 分类头：`Linear(64->2)` 输出非唤醒/唤醒

## 常见问题

### 1) 训练或推送很慢

请确认 `data/` 已被 `.gitignore` 忽略，不要把完整数据集提交到 GitHub。

### 2) Windows 出现 `LF will be replaced by CRLF`

这是行尾格式提示，不影响训练与推理。

### 3) 录音脚本无法运行

通常是音频设备或依赖问题，检查：

- `sounddevice`、`soundfile` 是否安装成功
- 系统麦克风权限是否开启

## 说明

- `main.py` 是模板脚本，不参与训练/评估主流程。
- 当前仓库包含已训练权重和可视化结果，便于快速复现与演示。

