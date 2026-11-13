# 食物美学评分系统 (Food Aesthetics Scoring System)

一个基于深度学习的食物图像美学评分和分析系统，使用 MobileNet 架构进行美学评估，并提供多维度的图像属性分析。

## 📋 项目简介

本项目实现了一个完整的食物图像美学评估系统，能够：
- 对食物图像进行美学评分（0-100分）
- 分析图像的多个视觉属性（清晰度、饱和度、色温等）
- 提供基于高分和低分样本对比的优化建议
- 支持模型训练、推理和优化部署

## ✨ 主要功能

### 1. 美学评分
- 使用基于 MobileNetV2 的神经网络模型
- 输出 0-100 的美学评分
- 支持批量图像评估

### 2. 属性分析
支持以下10个食物图像属性的计算和分析：
- **brightness**: 亮度
- **saturation**: 饱和度
- **contrast**: 对比度
- **clarity**: 清晰度/高光比例
- **warm**: 暖色调比例
- **colourfulness**: 色彩丰富度
- **size_difference**: 尺寸差异
- **color_difference**: 色彩差异
- **texture_difference**: 纹理差异
- **segment_colorfulness**: 分段色彩度

### 3. 优化建议
- 基于高分/低分样本的对比分析
- 生成KDE（核密度估计）分布图
- 提供各属性的理想范围（P5-P95百分位）

### 4. 模型优化
- 支持 INT8 量化优化
- 提供 TFLite 转换
- 对比浮点模型和量化模型的性能

## 🛠️ 技术栈

- **深度学习框架**: TensorFlow 2.x / Keras
- **预训练模型**: MobileNet / MobileNetV2
- **图像处理**: OpenCV, PIL
- **数据处理**: NumPy, Pandas
- **可视化**: Matplotlib, Seaborn

## 📦 安装依赖

```bash
pip install tensorflow
pip install opencv-python
pip install pillow
pip install numpy
pip install pandas
pip install scikit-learn
pip install matplotlib
pip install seaborn
pip install tqdm
```

或使用以下命令一次性安装：

```bash
pip install tensorflow opencv-python pillow numpy pandas scikit-learn matplotlib seaborn tqdm
```

## 🚀 快速开始

### 1. 模型推理

使用已训练模型对单张或多张图像进行美学评分：

```bash
python inference.py --image_path /path/to/image.jpg
```

或批量推理：

```bash
python inference.py --image_dir /path/to/images/ --output output/results.csv
```

**参数说明**:
- `--image_path`: 单张图像路径
- `--image_dir`: 图像文件夹路径
- `--weights_path`: 模型权重文件路径（默认：`trained_weights.h5`）
- `--output`: 输出CSV文件路径

### 2. 模型训练

基础训练：

```bash
python train.py --data_csv /path/to/dataset.csv --img_dir /path/to/images/
```

**参数说明**:
- `--data_csv`: 训练数据CSV文件（需包含图像路径和美学分数）
- `--img_dir`: 图像文件夹路径
- `--batch_size`: 批次大小（默认：32）
- `--epochs`: 训练轮数（默认：100）
- `--output_dir`: 输出目录（默认：`output/checkpoints`）

### 3. 优化训练（含量化）

训练并进行INT8量化优化：

```bash
python optimization_train.py --data_csv /path/to/dataset.csv --img_dir /path/to/images/
```

这将生成：
- Keras 模型（.h5）
- TFLite 量化模型（.tflite）

### 4. 使用主程序

主程序整合了推理和属性计算功能：

```bash
python main.py --image_path /path/to/image.jpg --weights_path trained_weights.h5
```

### 5. 生成优化建议

基于高分和低分样本的对比分析：

```bash
python suggestion.py
```

需要预先准备：
- `output/G_high_scores.csv`: 高分样本数据
- `output/G_low_scores.csv`: 低分样本数据

输出：
- KDE分布对比图（保存在 `output/visualizations/`）
- 属性阈值CSV（`output/attribute_thresholds.csv`）

### 6. 模型性能对比

比较浮点模型和量化模型的性能：

```bash
python comparison.py
```

## 📁 项目结构

```
food_aesthetics/
├── main.py                     # 主程序（推理+属性分析）
├── train.py                    # 基础模型训练
├── optimization_train.py       # 优化训练（含量化）
├── inference.py                # 推理脚本
├── suggestion.py               # 优化建议生成
├── comparison.py               # 模型性能对比
├── trained_weights.h5          # 预训练模型权重
├── 6.py                        # 实验脚本
├── 6.1.py                      # 实验脚本
└── README.md                   # 项目说明文档
```

## 🎯 使用场景

1. **食品摄影评估**: 自动评估食品图片的美学质量
2. **菜单优化**: 帮助餐厅选择高质量的菜品图片
3. **社交媒体**: 为美食博主提供图片质量建议
4. **电商平台**: 自动筛选和排序食品图片
5. **图像质量控制**: 批量检测和过滤低质量食物图片

## 📊 数据格式

### 训练数据CSV格式

CSV文件应包含以下列：
- `image_name` 或 `filename`: 图像文件名
- `aesthetic_score`: 美学分数（0-1之间的浮点数）
- 可选：10个属性分数列

示例：
```csv
image_name,aesthetic_score,brightness,saturation,contrast,clarity,warm,colourfulness,size_difference,color_difference,texture_difference,segment_colorfulness
food_001.jpg,0.85,0.72,0.68,0.75,0.80,0.65,0.70,0.60,0.55,0.58,0.72
food_002.jpg,0.45,0.40,0.35,0.42,0.38,0.50,0.48,0.45,0.40,0.42,0.46
```

## 🔧 模型架构

### NimaMobileNet
- **Backbone**: MobileNet / MobileNetV2
- **预训练权重**: ImageNet
- **输入尺寸**: 224 × 224 × 3
- **输出**: 
  - 主输出：美学分数（回归任务，sigmoid激活）
  - 可选：10个属性分数

### 训练配置
- **优化器**: Adam (lr=0.00001)
- **损失函数**: Mean Squared Error (MSE)
- **评估指标**: Mean Absolute Error (MAE)
- **Dropout**: 0.25

## 📈 模型优化

### INT8 量化
- 使用训练后量化（Post-Training Quantization）
- 将模型大小减小约 4 倍
- 推理速度提升约 2-4 倍
- MAE损失 < 0.01（精度几乎无损）

### 部署优化
- 支持 TensorFlow Lite 格式
- 适合移动端和边缘设备部署
- 低内存占用

## 🎨 属性说明

- **Clarity（清晰度）**: 高光像素比例，范围 0-1
- **Saturation（饱和度）**: HSV色彩空间S通道均值，范围 0-255
- **Warm（色温）**: 暖色调像素比例，范围 0-1
- **Model Score（模型评分）**: 神经网络预测的美学分数，范围 0-1

所有属性在输出时会被归一化到 0-100 的范围。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目仅供学习和研究使用。

## 📧 联系方式

如有问题或建议，请通过邮箱联系。

---

**注意**: 使用前请确保已下载或训练好模型权重文件 `trained_weights.h5`。
