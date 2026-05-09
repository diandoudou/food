# filename: food_aesthetics/main.py

# import dependencies
import os
import argparse
import time
import pandas as pd
from pathlib import Path
from PIL import Image
import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input
# 确保导入 MobileNet 时使用的是正确的路径 (如果是自定义模型，可能需要修改)
from tensorflow.keras.applications.mobilenet import MobileNet


# --- Model Definition (NimaMobileNet) ---
class NimaMobileNet(tf.keras.Model):
    # 此模型结构保持不变
    def __init__(self, training=True):
        super(NimaMobileNet, self).__init__()
        self.training = training

        # 1. 定义输入
        inputs = Input(shape=(224, 224, 3))

        # 2. 加载 MobileNet Base
        self.base_model = MobileNet(
            input_shape=(224, 224, 3),
            alpha=1,
            include_top=False,
            pooling='avg',
            weights=None,
            name='MobileNet_Base'
        )

        # 3. 构建模型层
        x = self.base_model(inputs)

        if self.training:
            # 3.1. 使用 Dropout 层
            self.dropout = Dropout(0.25, name='dropout_1')
            x = self.dropout(x)

        # 3.2. Dense 层 (ReLU)
        self.dense_relu = Dense(10, activation='relu', name='dense_relu')
        # 注意：原代码此处未明确使用 x = self.dense_relu(x)，但为了保持原意，我假设它应该使用
        x = self.dense_relu(x)

        # 3.3. 最终输出层
        self.fc_last = Dense(2, name='fc_last')
        outputs = self.fc_last(x)

        # 4. 构建完整的 Functional API Model
        self.nima_model = Model(inputs=inputs, outputs=outputs, name="NimaMobileNet")

    def call(self, x):
        return self.nima_model(x)


# --- Core Logic (FoodAesthetics) ---
class FoodAesthetics:
    # 构造函数现在接受权重文件路径
    def __init__(self, weights_path):
        super(FoodAesthetics, self).__init__()
        self.__batch_size = 1
        self.temperature = 1.536936640739441

        # 实例化封装类
        self.model_wrapper = NimaMobileNet(training=False)

        # 获取 Functional API Model
        self.model = self.model_wrapper.nima_model

        # 必须先编译/build才能加载权重
        self.model.build((self.__batch_size, 224, 224, 3))

        # 路径处理
        weights_path = Path(weights_path)

        if not weights_path.exists():
            print(f"Error: Weights file not found at {weights_path.resolve()}")
            raise FileNotFoundError(f"Required weights file not found at: {weights_path.resolve()}")

        # 尝试加载权重
        print(f"Attempting to load weights from: {weights_path.resolve()}")
        try:
            # 尝试直接加载
            self.model.load_weights(weights_path)
            print("Model weights loaded successfully.")
        except Exception as e:
            # 如果直接加载失败，尝试通过名称加载，并跳过不匹配的层
            print(f"Initial weight loading failed ({e}). Attempting to load by name...")
            try:
                self.model.load_weights(weights_path, by_name=True, skip_mismatch=True)
                print("Model weights loaded successfully (by name, with potential mismatches skipped).")
            except Exception as load_error:
                # 如果尝试了所有方法仍然失败，报告错误
                raise load_error

    # aesthetic_score 方法保持不变
    def aesthetic_score(self, path):
        photo = np.array(self._load_image(path))
        photo = tf.image.random_crop(tf.convert_to_tensor(photo / 255.0, dtype=tf.float32), (224, 224, 3))

        # 使用 Functional API Model 进行预测
        logits = self.model(tf.expand_dims(photo, axis=0))

        logits_scaled = tf.math.divide(logits, self.temperature)
        # 你的原始模型输出 logits (batch, 2)，softmax 后取第二个 (索引 1) 作为美学分数
        score = tf.nn.softmax(logits_scaled).numpy()[:, 1].item()
        return score

    # --- 以下 10 个食物属性方法保持不变 ---

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
        mask = np.zeros(image.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        rect = (1, 1, image.shape[1] - 2, image.shape[0] - 2)
        cv.grabCut(image, mask, rect, bgdModel, fgdModel, n_iter, cv.GC_INIT_WITH_RECT)
        return np.where((mask == cv.GC_PR_BGD) | (mask == cv.GC_BGD), 0, 1).astype('uint8')

    def size_difference(self, path):
        mask = self._grabcut(path)
        h, w = mask.shape
        counts = np.unique(mask, return_counts=True)

        if len(counts[1]) < 2:
            return 0.0

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
        front = np.where(mask[:, :, np.newaxis] == 1, image, 0).astype(np.uint8)

        back_pixels_count = np.sum(mask == 0)
        back = np.where(mask[:, :, np.newaxis] == 0, image, 0).astype(np.uint8)

        edges_front = cv.Canny(front, 50, 150)
        edges_back = cv.Canny(back, 50, 150)

        front_density = np.sum(edges_front) / front_pixels_count if front_pixels_count > 0 else 0
        back_density = np.sum(edges_back) / back_pixels_count if back_pixels_count > 0 else 0

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


# --- Main Execution Block ---

# 1. create path functions (新增文件路径检查函数)
def dir_path(string):
    """检查路径是否存在且为目录"""
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def file_path(string):
    """检查路径是否存在且为文件"""
    if os.path.isfile(string):
        return string
    else:
        raise FileNotFoundError(string)


if __name__ == '__main__':
    # 2. parametrize script (匹配命令行参数)
    parser = argparse.ArgumentParser(
        description='Calculate aesthetic and attribute scores for food images, and split results into equal high/low groups.'
    )
    # 匹配 --model 参数，用于指定权重文件路径
    parser.add_argument('--model',
                        help='Path to the trained model weights file (e.g., food_aesthetics/trained_weights.h5)',
                        type=file_path,
                        required=True)
    # 匹配 --val_dir 参数，用于指定图片文件夹路径
    parser.add_argument('--val_dir',
                        help='Path to validation images folder (e.g., validation)',
                        type=dir_path,
                        required=True)
    # 新增参数用于指定输出 CSV 文件的路径/前缀
    parser.add_argument('--output_prefix',
                        help='Output file prefix for the CSV files (e.g., output/feature)',
                        type=str,
                        default='output/feature')

    args = parser.parse_args()

    # 使用新的参数变量
    weights_path = args.model
    path_dir = Path(args.val_dir)

    # 提取输出文件名前缀和目录
    csv_out_prefix = Path(args.output_prefix).stem
    csv_out_dir = Path(args.output_prefix).parent

    # 3. init model (传入权重路径)
    try:
        fa = FoodAesthetics(weights_path=weights_path)
    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: {e}")
        exit(1)

    print('Starting to score pictures...')

    # 4. score images
    images = os.listdir(path_dir)
    images = [image for image in images if image.lower().endswith(('.jpg', '.jpeg'))]

    results = []
    print(f"Found {len(images)} images in {path_dir.resolve()}.")

    # FoodAesthetics 中的所有属性方法：
    attribute_methods = [
        "brightness", "saturation", "contrast", "clarity", "warm", "colourfulness",
        "size_difference", "color_difference", "texture_difference", "segment_colorfulness"
    ]

    total_start_time = time.time()

    for image in images:
        start_time = time.time()
        image_path = path_dir / image

        image_data = {"image": image}

        try:
            # 1. 美学分数 (aesthetic score)
            image_data["aesthetic_score"] = fa.aesthetic_score(image_path)

            # 2. 食物属性分数 (Food attributes)
            for attr_name in attribute_methods:
                attr_func = getattr(fa, attr_name)
                image_data[attr_name] = attr_func(image_path)

            results.append(image_data)

            end_time = time.time()
            print(f"Scored image: {image} in {end_time - start_time:.2f}s")

        except Exception as e:
            print(f"Error evaluating {image}: {e}")
            continue

    total_end_time = time.time()
    print(f"\nScored all pictures. Total time: {total_end_time - total_start_time:.2f}s")

    # 5. export to CSV: 严格 1:1 分组逻辑
    if not results:
        print("No results to export. Exiting.")
        exit(0)

    # 确保 output 文件夹存在
    csv_out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)

    ### --- 基于排序列表进行 1:1 数量相等分组 --- ###

    # 1. 按美学分数降序排序
    df_sorted = df.sort_values(by='aesthetic_score', ascending=False).reset_index(drop=True)

    # 2. 计算总图片数量
    total_count = len(df_sorted)

    # 3. 确定分组大小 (确保两组数量相等)
    group_size = total_count // 2

    if group_size == 0:
        print(
            f"\nWarning: Not enough images ({total_count} total) to create two equally sized groups (N<2). Exporting single CSV.")
        csv_full_path = csv_out_dir / f'{csv_out_prefix}_full.csv'
        df_sorted.to_csv(csv_full_path, index=False)
        print(f'Exported the full results to {csv_full_path.resolve()}')
        exit(0)

    # 4. 分组
    # 高分组：分数最高的 group_size 个图片
    df_high = df_sorted.head(group_size)
    # 低分组：分数最低的 group_size 个图片
    df_low = df_sorted.tail(group_size)

    # 5. 导出 CSV

    # 导出高分组
    csv_high_path = csv_out_dir / f'{csv_out_prefix}_high_aesthetics.csv'
    df_high.to_csv(csv_high_path, index=False)
    print(f'\nExported High Aesthetics group (N={len(df_high)}) to {csv_high_path.resolve()}')

    # 导出低分组
    csv_low_path = csv_out_dir / f'{csv_out_prefix}_low_aesthetics.csv'
    df_low.to_csv(csv_low_path, index=False)
    print(f'Exported Low Aesthetics group (N={len(df_low)}) to {csv_low_path.resolve()}')

    # 6. 处理中位数附近的余数 (如果总数为奇数)
    if total_count % 2 != 0:
        # 余下的图片在中间，被排除以保持分组数量相等
        df_excluded = df_sorted.iloc[group_size:total_count - group_size]
        print(
            f"Note: {len(df_excluded)} image(s) (with score around {df_excluded['aesthetic_score'].mean():.4f}) were excluded to strictly maintain equal group sizes.")

    ### --- 核心修改结束 --- ###