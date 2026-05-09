# filename: optimization_train.py
# 功能: 训练基于 MobileNetV2 的美学分数预测模型，并进行训练后 INT8 量化。

# import dependencies
import os
import argparse
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


# --- Model Definition (无剪枝封装) ---
class NimaMobileNetV2SingleOutput(tf.keras.Model):
    """
    基于 MobileNetV2 的单输出模型，用于美学评分回归。
    """

    def __init__(self, training=True):
        super(NimaMobileNetV2SingleOutput, self).__init__()
        self.training = training

        # 1. 定义输入
        inputs = Input(shape=(224, 224, 3), name='input_image')

        # 2. 加载 MobileNetV2 Base
        self.base_model = MobileNetV2(
            input_shape=(224, 224, 3), include_top=False, pooling='avg',
            weights='imagenet'
        )
        x = self.base_model(inputs)
        # MobileNetV2 基础模型的实际名称是 'mobilenetv2_1.00_224'
        self.base_model_name = 'mobilenetv2_1.00_224'

        if self.training:
            self.dropout_layer = Dropout(0.25, name='dropout_1')
            x = self.dropout_layer(x)

        # 3. 共享特征层 (ReLU)
        shared_dense_relu = Dense(10, activation='relu', name='shared_dense_relu')(x)

        # 4. 定义单输出头: 美学分数 (回归输出，范围 [0, 1])
        self.aesthetic_head = Dense(1, activation='sigmoid', name='aesthetic_score')(shared_dense_relu)

        # 5. 构建完整的 Functional API Model
        self.nima_model = Model(inputs=inputs, outputs=self.aesthetic_head, name="NimaMobileNetV2_Single_Output")

    def call(self, x):
        return self.nima_model(x)


# --- 数据生成器 ---
class GPDDataGenerator(Sequence):
    """
    Keras Sequence for loading image data and single-output labels (aesthetic score).
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
            return photo / 255.0
        except Exception as e:
            # print(f"Error loading image {img_path}: {e}")
            return np.zeros((*self.target_size, 3), dtype=np.float32)

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_df = self.df.iloc[indexes]

        X = np.empty((len(batch_df), *self.target_size, 3), dtype=np.float32)
        Y = np.empty((len(batch_df), 1), dtype=np.float32)

        for i, row in enumerate(batch_df.itertuples()):
            img_path = self.img_dir / row.image

            X[i,] = self._load_and_preprocess(img_path)
            Y[i] = getattr(row, 'aesthetic_score')

        return X, Y


# --- 训练函数 ---
def fine_tune_model(epochs, csv_file, img_dir, output_model_path, pretrained_model_path=None):
    print("Starting training process with TensorFlow/Keras (Aesthetic Score Only)...")

    # 1. 数据准备
    data_df = pd.read_csv(csv_file)
    data_df['image'] = data_df['image'].astype(str)
    required_cols = ['image', 'aesthetic_score']
    if any(col not in data_df.columns for col in required_cols):
        raise ValueError(f"CSV file must contain columns: {required_cols}")

    train_df, val_df = train_test_split(data_df, test_size=0.2, random_state=42)
    train_generator = GPDDataGenerator(train_df, img_dir, batch_size=32, shuffle=True)
    val_generator = GPDDataGenerator(val_df, img_dir, batch_size=32, shuffle=False)
    print(f"Training set size: {len(train_df)}, Validation set size: {len(val_df)}")

    # 2. 模型构建
    model_wrapper = NimaMobileNetV2SingleOutput(training=True)
    model = model_wrapper.nima_model
    model.build((None, 224, 224, 3))

    if pretrained_model_path:
        print(f"Attempting to load weights from: {pretrained_model_path}")
        try:
            # 加载权重
            model.load_weights(pretrained_model_path, by_name=True, skip_mismatch=True)
            print(f"Pre-trained weights loaded successfully from {pretrained_model_path}.")
        except Exception as e:
            print(f"WARNING: Failed to load pre-trained weights. {e}")

    # 3. 阶段 A: 训练顶部层 (冻结基础模型)
    loss_function = MeanSquaredError()
    metrics = ['mae']

    # 修正: 使用模型包装器中定义的实际层名称
    base_model = model.get_layer(model_wrapper.base_model_name)
    base_model.trainable = False
    initial_learning_rate = 0.0001

    model.compile(
        optimizer=Adam(learning_rate=initial_learning_rate),
        loss=loss_function,
        metrics=metrics
    )

    print("\n--- Phase A: Training Top Layers (Epochs 1-5) ---")
    model.fit(train_generator, validation_data=val_generator, epochs=5)

    # 4. 阶段 B: 解冻并微调整个模型
    base_model.trainable = True
    # 冻结 MobileNetV2 约 70% 的层
    freeze_at = int(len(base_model.layers) * 0.7)
    for i, layer in enumerate(base_model.layers):
        if i < freeze_at:
            layer.trainable = False

    fine_tune_learning_rate = 0.00001
    model.compile(
        optimizer=Adam(learning_rate=fine_tune_learning_rate),
        loss=loss_function,
        metrics=metrics
    )

    checkpoint_filename = 'aesthetic_baseline_single_output.h5'
    checkpoint = ModelCheckpoint(
        os.path.join(output_model_path, checkpoint_filename),
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        save_weights_only=False,
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

    # 5. 训练后量化 (Post-Training Quantization)
    print(f"\nTraining completed. Best Keras model saved to {output_model_path}/{checkpoint_filename}")
    quantize_model(model, train_generator, output_model_path)


# --- 训练后量化函数 (PTQ) ---
def quantize_model(model, representative_dataset_generator, output_model_path):
    print("\n--- Starting Post-Training Integer Quantization ---")

    # 1. 创建 TFLite Converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # 2. 定义代表性数据集 (用于 INT8 校准)
    def representative_dataset():
        for i, (input_data, _) in enumerate(representative_dataset_generator):
            if i > 100:  # 使用 101 个批次进行校准
                break
            # TFLite Converter 需要一个包含输入数据的列表
            yield [input_data]

    converter.representative_dataset = representative_dataset
    # 3. 设置全整数量化 (INT8)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_quant_model = converter.convert()

    # 4. 保存模型
    quant_model_path = os.path.join(output_model_path, 'aesthetic_quantized_int8_model.tflite')
    with open(quant_model_path, 'wb') as f:
        f.write(tflite_quant_model)

    print(f"🎉 Integer Quantized TFLite model saved to: {quant_model_path}")
    print(f"💡 模型大小估计减少 4 倍。")


# --- 执行块 ---
if __name__ == '__main__':

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

    fine_tune_model(args.epochs, args.csv_file, args.img_dir,
                    args.output_model_path, args.pretrained_model_path)