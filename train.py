# filename: train_tensorflow_baseline.py

# import dependencies
import os
import argparse
import time
import pandas as pd
from pathlib import Path
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
# 导入 OpenCV
import cv2 as cv

# --- 辅助方法和属性列表 ---
# 10个食物属性的名称列表
ATTRIBUTE_NAMES = [
    "brightness", "saturation", "contrast", "clarity", "warm", "colourfulness",
    "size_difference", "color_difference", "texture_difference", "segment_colorfulness"
]


# --- Model Definition (NimaMobileNetV2 Single-Output) ---
class NimaMobileNetV2SingleOutput(tf.keras.Model):
    """
    基于 MobileNetV2 的单输出模型，只预测美学分数。
    原多输出模型修改为只保留美学分数输出头。
    """

    def __init__(self, training=True):
        super(NimaMobileNetV2SingleOutput, self).__init__()
        self.training = training

        # 1. 定义输入
        inputs = Input(shape=(224, 224, 3), name='input_image')

        # 2. 加载 MobileNetV2 Base
        self.base_model = MobileNetV2(
            input_shape=(224, 224, 3), include_top=False, pooling='avg',
            weights='imagenet',
            name='MobileNetV2_Base'
        )
        x = self.base_model(inputs)

        if self.training:
            self.dropout_layer = Dropout(0.25, name='dropout_1')
            x = self.dropout_layer(x)

        # 3. 共享特征层 (ReLU)
        shared_features = Dense(10, activation='relu', name='shared_dense_relu')(x)

        # --- 4. 定义单输出头 ---
        # 4.1. 主任务头: 美学分数 (回归输出)
        self.aesthetic_head = Dense(1, activation='sigmoid', name='aesthetic_score')(shared_features)

        # 5. 构建完整的 Functional API Model
        # 注意: outputs 现在只包含 aesthetic_score
        self.nima_model = Model(inputs=inputs, outputs=self.aesthetic_head, name="NimaMobileNetV2_Single")

    def call(self, x):
        return self.nima_model(x)


# --- 数据生成器 (修改为只加载美学分数) ---
class GPDDataGenerator(Sequence):
    """
    Keras Sequence for loading image data and single-output labels (aesthetic score).
    不再加载属性分数标签。
    """

    def __init__(self, df, img_dir, batch_size=32, target_size=(224, 224), shuffle=True):
        self.df = df
        self.img_dir = Path(img_dir)
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _load_and_preprocess(self, img_path):
        """加载、缩放图片并进行归一化"""
        try:
            pic = Image.open(img_path).convert("RGB")
            pic_res = pic.resize(self.target_size)
            photo = np.array(pic_res, dtype=np.float32)
            # 归一化到 [0, 1]
            return photo / 255.0
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return np.zeros((*self.target_size, 3), dtype=np.float32)

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_df = self.df.iloc[indexes]

        X = np.empty((len(batch_df), *self.target_size, 3), dtype=np.float32)
        # Y 现在是一个字典，只包含 'aesthetic_score'
        Y = {}
        Y['aesthetic_score'] = np.empty((len(batch_df), 1), dtype=np.float32)

        for i, row in enumerate(batch_df.itertuples()):
            img_path = self.img_dir / row.image
            X[i,] = self._load_and_preprocess(img_path)
            Y['aesthetic_score'][i] = getattr(row, 'aesthetic_score')

        return X, Y['aesthetic_score']  # ⚠️ 直接返回 X 和 aesthetic_score 数组 (单个输出)


# --- 主微调函数 (简化编译步骤) ---
def fine_tune_model(epochs, csv_file, img_dir, output_model_path, pretrained_model_path=None):
    print("Starting training process with TensorFlow/Keras (Single Output - Aesthetic Score Only)...")

    # 1. Data Loading and Preparation
    data_df = pd.read_csv(csv_file)
    data_df['image'] = data_df['image'].astype(str)

    required_cols = ['image', 'aesthetic_score']
    missing_cols = [col for col in required_cols if col not in data_df.columns]
    if missing_cols:
        raise ValueError(f"CSV file is missing required columns: {missing_cols}")

    train_df, val_df = train_test_split(data_df, test_size=0.2, random_state=42)

    # ⚠️ Data Generator 现在只返回 X 和 Y (aesthetic_score)
    train_generator = GPDDataGenerator(train_df, img_dir, batch_size=32, shuffle=True)
    val_generator = GPDDataGenerator(val_df, img_dir, batch_size=32, shuffle=False)
    print(f"Training set size: {len(train_df)}, Validation set size: {len(val_df)}")

    # 2. Model initialization
    model_wrapper = NimaMobileNetV2SingleOutput(training=True)  # 更改为单输出模型
    model = model_wrapper.nima_model
    model.build((None, 224, 224, 3))

    if pretrained_model_path:
        print(f"Attempting to load weights from: {pretrained_model_path}")
        try:
            model.load_weights(pretrained_model_path, by_name=True, skip_mismatch=True)
            print(f"Pre-trained weights loaded successfully from {pretrained_model_path}.")
        except Exception as e:
            print(f"WARNING: Failed to load pre-trained weights. {e}")

    # 3. Loss function and optimizer setup
    # ⚠️ 简化为只针对 aesthetic_score 的损失函数
    loss_function = MeanSquaredError()
    metrics = ['mae']

    # --- 4. Training Loop (两阶段微调) ---

    # 阶段 A: 冻结基模型，训练顶层
    base_model = model.get_layer('MobileNetV2_Base')
    base_model.trainable = False

    initial_learning_rate = 0.0001
    model.compile(
        optimizer=Adam(learning_rate=initial_learning_rate),
        loss=loss_function,  # 单损失函数
        metrics=metrics
    )

    print("\n--- Phase A: Training Top Layers (Epochs 1-5) ---")
    model.fit(train_generator, validation_data=val_generator, epochs=5)

    # 阶段 B: 解冻 MobileNetV2，精细微调
    base_model.trainable = True
    freeze_at = int(len(base_model.layers) * 0.7)
    for i, layer in enumerate(base_model.layers):
        if i < freeze_at:
            layer.trainable = False

    fine_tune_learning_rate = 0.00001
    model.compile(
        optimizer=Adam(learning_rate=fine_tune_learning_rate),
        loss=loss_function,  # 单损失函数
        metrics=metrics
    )

    checkpoint_filename = 'aesthetic_baseline_single_output.weights.h5'  # 更改文件名以区分
    checkpoint = ModelCheckpoint(
        os.path.join(output_model_path, checkpoint_filename),
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        save_weights_only=True,
        verbose=1
    )

    print("\n--- Phase B: Fine-Tuning Entire Model ---")
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        initial_epoch=5,
        callbacks=[checkpoint]
    )

    print(f"\nTraining completed. Best model weights saved to {output_model_path}/{checkpoint_filename}")


# --- 属性计算方法 (保持不变，但现在它们不再用于模型训练) ---

# GPDDataGenerator中不再需要这些方法，它们是独立的计算函数
class FoodAttributeCalculator:
    """
    用于计算食物美学属性的辅助类。
    这些方法不用于模型训练，但用于数据集中标签的生成（假设这些标签已经存在于 CSV 文件中）。
    """

    def _load_image(self, path):
        pic = Image.open(path).convert("RGB")
        width, height = pic.size
        s = max(224 / width, 224 / height)
        if width < height:
            pic_res = pic.resize((224, round(s * height)))
        else:
            pic_res = pic.resize((round(s * width), 224))
        return pic_res

    def _image2hsv(self, pic):
        # 注意: cv.imread 可能需要完整路径，这里假设 pic 是路径
        img_array = np.array(self._load_image(pic))
        return cv.cvtColor(np.float32(img_array), cv.COLOR_RGB2HSV)

    def brightness(self, path):
        image = self._image2hsv(path)
        return image[:, :, 2].mean()

    def saturation(self, path):
        image = self._image2hsv(path)
        return image[:, :, 1].mean()

    def contrast(self, path):
        image = self._image2hsv(path)
        return image[:, :, 2].std()

    def clarity(self, path, thresold=0.7, scaler=255):
        image = self._image2hsv(path)
        h, w, c = image.shape
        return np.sum(image[:, :, 2] / scaler > thresold) / (h * w)

    def warm(self, path):
        image = self._image2hsv(path)
        h, w, c = image.shape
        # H 范围 [0, 179]
        return np.sum((image[:, :, 0] < 30) | (image[:, :, 0] > 110)) / (h * w)

    def colourfulness(self, path):
        image = np.array(self._load_image(path))
        R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        rg = R - G
        yb = 0.5 * (R + G) - B
        sigma = np.sqrt(np.square(np.std(rg)) + np.square(np.std(yb)))
        mu = np.sqrt(np.square(np.mean(rg)) + np.square(np.mean(yb)))
        c = sigma + 0.3 * mu
        return c

    def _grabcut(self, path, n_iter=10):
        image = np.array(self._load_image(path))
        # GrabCut 期望输入是 np.uint8
        image_gc = image.astype(np.uint8)
        mask = np.zeros(image_gc.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        rect = (1, 1, image_gc.shape[1] - 2, image_gc.shape[0] - 2)
        cv.grabCut(image_gc, mask, rect, bgdModel, fgdModel, n_iter, cv.GC_INIT_WITH_RECT)
        return np.where((mask == cv.GC_PR_BGD) | (mask == cv.GC_BGD), 0, 1).astype('uint8')

    def size_difference(self, path):
        mask = self._grabcut(path)
        h, w = mask.shape
        counts = np.unique(mask, return_counts=True)

        # 检查背景和前景是否都存在
        if len(counts[1]) < 2:
            # 如果只有一种（前景或背景），返回 0.0
            return 0.0

            # 假设 0 是背景，1 是前景
        n_back = counts[1][0]
        n_for = counts[1][1]

        return (n_for - n_back) / (h * w)

    def color_difference(self, path):
        image = np.array(self._load_image(path))
        mask = self._grabcut(path)

        front_pixels = image[mask == 1].reshape(-1, 3)
        back_pixels = image[mask == 0].reshape(-1, 3)

        avg_front = front_pixels.mean(axis=0) if front_pixels.size > 0 else np.zeros(3)
        avg_back = back_pixels.mean(axis=0) if back_pixels.size > 0 else np.zeros(3)

        return np.linalg.norm(avg_front - avg_back)

    def texture_difference(self, path):
        image = np.array(self._load_image(path))
        mask = self._grabcut(path)

        front_pixels_count = np.sum(mask == 1)
        # 提取前景和背景区域，将其余部分置为黑色（0），并确保类型为 np.uint8
        front = np.where(mask[:, :, np.newaxis] == 1, image, 0).astype(np.uint8)
        back = np.where(mask[:, :, np.newaxis] == 0, image, 0).astype(np.uint8)
        back_pixels_count = np.sum(mask == 0)

        # Canny 边缘检测
        edges_front = cv.Canny(front, 50, 150)
        edges_back = cv.Canny(back, 50, 150)

        # 边缘密度：边缘像素总和 / 区域像素总数
        front_density = np.sum(edges_front > 0) / front_pixels_count if front_pixels_count > 0 else 0
        back_density = np.sum(edges_back > 0) / back_pixels_count if back_pixels_count > 0 else 0

        # 注意：原代码中使用 np.sum(edges_front) 可能会受到像素值的影响，
        # 通常边缘密度计算是边缘像素的数量 / 区域像素总数。这里使用 np.sum(edges_front > 0)
        # 来计算边缘像素数量，保持逻辑一致。

        return np.abs(front_density - back_density)

    def segment_colorfulness(self, path):
        image = np.array(self._load_image(path))
        mask = self._grabcut(path)
        (B, G, R) = cv.split(image.astype("float"))

        front_mask = (mask == 1)
        R_fg = R[front_mask]
        G_fg = G[front_mask]
        B_fg = B[front_mask]

        if R_fg.size == 0:
            return 0.0

        rg = np.absolute(R_fg - G_fg)
        yb = np.absolute(0.5 * (R_fg + G_fg) - B_fg)

        stdRoot = np.sqrt((rg.std() ** 2) + (yb.std() ** 2))
        meanRoot = np.sqrt((rg.mean() ** 2) + (yb.mean() ** 2))

        return stdRoot + (0.3 * meanRoot)


# --- 执行块 ---
if __name__ == '__main__':
    def check_dir(path):
        if not os.path.isdir(path):
            raise NotADirectoryError(f"Directory not found: {path}")
        return Path(path)


    parser = argparse.ArgumentParser(description="Train aesthetic score prediction model (TensorFlow Single Output)")
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (total)')
    parser.add_argument('--csv_file', type=str, required=True,
                        help='path to the csv file with image names and labels (only aesthetic score is used)')
    parser.add_argument('--img_dir', type=str, required=True, help='directory with all the images')
    parser.add_argument('--output_model_path', type=str, required=True,
                        help='path to save the trained model checkpoints')
    parser.add_argument('--pretrained_model_path', type=str, default=None,
                        help='path to the pretrained model weights (.h5 file)')
    args = parser.parse_args()

    if not os.path.exists(args.output_model_path):
        os.makedirs(args.output_model_path)

    fine_tune_model(args.epochs, args.csv_file, args.img_dir, args.output_model_path, args.pretrained_model_path)