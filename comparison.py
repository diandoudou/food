import os
import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from pathlib import Path

# --- 假设您已经定义了 GPDDataGenerator 类和 val_df ---
# 假设您已在 Colab 中定义了这些变量或在环境中执行了训练脚本。
# 请根据您的实际情况替换 'output/checkpoints' 和其他路径。

OUTPUT_PATH = "output/checkpoints"  # 替换为您的 output_model_path
CHECKPOINT_H5 = "aesthetic_baseline_single_output.h5"
TFLITE_MODEL = "aesthetic_quantized_int8_model.tflite"

H5_PATH = os.path.join(OUTPUT_PATH, CHECKPOINT_H5)
TFLITE_PATH = os.path.join(OUTPUT_PATH, TFLITE_MODEL)


# 假设 val_generator 已经通过之前的 fine_tune_model 函数创建并可用

# --- 1. 评估 Keras 浮点模型 (Float32) ---
def evaluate_keras_model(model_path, val_generator):
    print(f"\n--- 1. 评估 Keras 浮点模型 (基准): {Path(model_path).name} ---")
    try:
        # 重新加载并编译模型
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(
            optimizer=Adam(learning_rate=0.00001),
            loss=MeanSquaredError(),
            metrics=['mae']
        )

        results = model.evaluate(val_generator, verbose=1)
        mae = results[1]
        print(f"✅ Keras 浮点模型 MAE: {mae:.4f}")
        return mae

    except Exception as e:
        print(f"❌ 无法加载或评估 Keras 模型: {e}")
        return None


# --- 2. 评估 TFLite 量化模型 (INT8) ---
def evaluate_tflite_model(tflite_path, val_generator):
    print(f"\n--- 2. 评估 TFLite INT8 量化模型: {Path(tflite_path).name} ---")
    try:
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # 获取量化参数
        input_scale, input_zero_point = input_details[0]['quantization']
        output_scale, output_zero_point = output_details[0]['quantization']

        all_predictions = []
        all_labels = []

        for X_batch, Y_batch in val_generator:
            # 输入量化 (Float32 -> INT8)
            X_quant = (X_batch / input_scale) + input_zero_point
            X_quant = np.clip(X_quant, -128, 127)
            X_quant = X_quant.astype(input_details[0]['dtype'])

            # 推理
            interpreter.set_tensor(input_details[0]['index'], X_quant)
            interpreter.invoke()

            # 获取 INT8 输出并反量化 (INT8 -> Float32)
            Y_quant = interpreter.get_tensor(output_details[0]['index'])
            Y_float = (Y_quant.astype(np.float32) - output_zero_point) * output_scale

            all_predictions.extend(Y_float.flatten())
            all_labels.extend(Y_batch.flatten())

        mae = mean_absolute_error(all_labels, all_predictions)
        print(f"✅ TFLite 量化模型 MAE: {mae:.4f}")
        return mae

    except Exception as e:
        print(f"❌ 无法评估 TFLite 模型: {e}")
        return None


# --- 3. 运行评估和对比 ---
def compare_model_performance(h5_path, tflite_path, val_generator):
    mae_float = evaluate_keras_model(h5_path, val_generator)
    mae_quant = evaluate_tflite_model(tflite_path, val_generator)

    if mae_float is not None and mae_quant is not None:
        delta = abs(mae_quant - mae_float)
        print("\n--- 结果对比 ---")
        print(f"原始 MAE (Float32): {mae_float:.4f}")
        print(f"量化 MAE (INT8):   {mae_quant:.4f}")
        print(f"性能差异 (绝对值):  {delta:.4f}")

        if delta < 0.01:
            print("结论: 量化成功！精度损失极小，可接受。")
        else:
            print("结论: 警告！量化导致的精度损失较大，请仔细检查。")

    # 文件大小对比
    size_float = os.path.getsize(h5_path) / (1024 * 1024)
    size_quant = os.path.getsize(tflite_path) / (1024 * 1024)
    print("\n--- 文件大小对比 ---")
    print(f"原始模型大小: {size_float:.2f} MB")
    print(f"量化模型大小: {size_quant:.2f} MB")
    print(f"大小减少百分比: {((size_float - size_quant) / size_float) * 100:.2f}%")