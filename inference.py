import tensorflow as tf
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from PIL import Image
import cv2 as cv  # Requires installation: pip install opencv-python

# --- Configuration and Scaling Ranges ---
TARGET_SIZE = (224, 224)

# Scaling Ranges for Min-Max normalization to 0-100.
# Only includes the 3 requested attributes and the model score.
SCALING_RANGES = {
    'saturation': {'min': 0, 'max': 255},
    'clarity': {'min': 0, 'max': 1},
    'warm': {'min': 0, 'max': 1},
    'model_score': {'min': 0.0, 'max': 1.0}
}

# The list of attributes to calculate and output
ATTRIBUTES_TO_CALCULATE = ['clarity', 'saturation', 'warm']


# --- Attribute Calculation Class (Simplified) ---
class ImageAttributes:
    """Calculates the requested aesthetic/compositional attributes of an image."""

    def __init__(self, target_size=TARGET_SIZE):
        self.target_size = target_size

    def _load_image(self, path):
        """Loads, converts to RGB, and scales the image maintaining aspect ratio."""
        pic = Image.open(path).convert("RGB")
        width, height = pic.size
        target_min = min(self.target_size)

        s = max(target_min / width, target_min / height)

        new_width = round(s * width)
        new_height = round(s * height)

        pic_res = pic.resize((new_width, new_height))
        return pic_res

    def _image2hsv(self, path):
        """Loads image, scales, and converts to HSV (uint8)."""
        pic = self._load_image(path)
        img_array = np.array(pic)
        return cv.cvtColor(img_array, cv.COLOR_RGB2HSV)

    # --- Requested Attribute Calculation Methods ---

    def saturation(self, path):
        """Saturation (Mean of S channel)."""
        image = self._image2hsv(path)
        return image[:, :, 1].mean()

    def clarity(self, path, threshold=200):
        """Clarity/Highlight proportion (Proportion of V pixels above a threshold)."""
        image = self._image2hsv(path)
        h, w, c = image.shape
        return np.sum(image[:, :, 2] > threshold) / (h * w)

    def warm(self, path):
        """Warm color proportion (H channel pixels in warm ranges)."""
        image = self._image2hsv(path)
        h, w, c = image.shape
        # H ranges: 0-180 for OpenCV uint8 input.
        return np.sum((image[:, :, 0] < 30) | (image[:, :, 0] > 110)) / (h * w)


# --- Attribute Calculation and Scaling Function ---
def calculate_and_scale_attributes(image_path, attribute_calculator):
    """Calculates the requested attributes and scales them to the 0-100 range."""
    attr_scores = {}

    for attr_name in ATTRIBUTES_TO_CALCULATE:
        try:
            method = getattr(attribute_calculator, attr_name)
            raw_value = method(image_path)

            min_val = SCALING_RANGES[attr_name]['min']
            max_val = SCALING_RANGES[attr_name]['max']

            if max_val == min_val:
                scaled_value = 0.0
            else:
                # Min-Max Scaling Formula: (x - min) / (max - min) * 100
                normalized_value = (raw_value - min_val) / (max_val - min_val)
                scaled_value = np.clip(normalized_value * 100, 0, 100)

            attr_scores[attr_name] = float(scaled_value)

        except Exception as e:
            print(f"⚠️ Attribute calculation failed ({attr_name}) for {Path(image_path).name}: {e}")
            attr_scores[attr_name] = np.nan

    return attr_scores


# --- Common Preprocessing Function ---
def preprocess_image_float(image_path, target_size):
    """Loads and preprocesses a single image, ready for TFLite quantization (Float32, 0-1)."""
    try:
        img = Image.open(image_path).convert("RGB")
        img_res = img.resize(target_size)
        # Convert to NumPy array and normalize to [0, 1]
        img_array = np.array(img_res, dtype=np.float32) / 255.0
        # Add batch dimension
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


# --- Inference Core Function: TFLite INT8 Quantized Model ---
def run_tflite_int8_inference(tflite_path, image_paths, attribute_calculator):
    """Batch inference for TFLite INT8 model and calculate the 3 requested attributes."""
    print(f"\n--- Starting TFLite INT8 Batch Inference ({len(image_paths)} images) ---")

    # Try loading TFLite interpreter
    try:
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"❌ TFLite Interpreter Loading Failed: {e}")
        return None

    # Get quantization parameters
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_scale, input_zero_point = input_details[0]['quantization']
    output_scale, output_zero_point = output_details[0]['quantization']

    results = []

    # Get model score scaling range (assumed 0-1 raw output)
    score_min = SCALING_RANGES['model_score']['min']
    score_max = SCALING_RANGES['model_score']['max']

    for img_path in image_paths:

        # 1. Attribute Calculation (Scaled 0-100) - Only the 3 requested attributes
        attr_scores = calculate_and_scale_attributes(img_path, attribute_calculator)

        # 2. Model Inference Preprocessing (Float32)
        input_float_data = preprocess_image_float(img_path, TARGET_SIZE)

        base_result = {'filename': Path(img_path).name}
        base_result.update(attr_scores)  # Add 3 attribute scores (0-100)

        if input_float_data is None:
            base_result.update({'score': np.nan, 'status': 'Load Failed'})
            results.append(base_result)
            continue

        try:
            # 3. Quantization: Float32 -> INT8
            input_int8_data = (input_float_data / input_scale) + input_zero_point
            input_int8_data = np.clip(input_int8_data, -128, 127)  # Clip to INT8 range
            input_int8_data = input_int8_data.astype(input_details[0]['dtype'])

            # 4. Inference
            interpreter.set_tensor(input_details[0]['index'], input_int8_data)
            interpreter.invoke()

            # 5. Dequantization: INT8 -> Float32
            output_int8_data = interpreter.get_tensor(output_details[0]['index'])
            raw_score = (output_int8_data.astype(np.float32) - output_zero_point) * output_scale
            raw_score = raw_score[0][0]

            # 6. Scale Model Score to 0-100
            if score_max == score_min:
                scaled_score = 0.0
            else:
                normalized_score = (raw_score - score_min) / (score_max - score_min)
                scaled_score = np.clip(normalized_score * 100, 0, 100)

            base_result.update({'score': float(scaled_score), 'status': 'Success'})  # Model Score (0-100)

        except Exception as e:
            print(f"⚠️ TFLite Inference Failed for {Path(img_path).name}: {e}")
            base_result.update({'score': np.nan, 'status': 'Inference Failed'})

        results.append(base_result)

    return results


# --- Main Execution Block ---
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Batch inference for TFLite INT8 Quantized model, calculating only clarity, saturation, and warm (4 total scores scaled to 0-100).")
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing all images to be scored.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the TFLite model file (.tflite).')
    parser.add_argument('--output_csv', type=str, default='inference_quantized_clarity_saturation_warm_results.csv',
                        help='Path to save the output CSV file.')

    args = parser.parse_args()

    if not args.model_path.lower().endswith('.tflite'):
        print(f"❌ Error: Model file must be a .tflite file.")
        exit()

    # Initialize the attribute calculator
    attr_calculator = ImageAttributes(target_size=TARGET_SIZE)

    # 1. Scan image files
    image_dir = Path(args.image_dir)
    image_paths = [str(p) for p in image_dir.glob('*.png')]  # Assuming only .png files

    if not image_paths:
        print(f"❌ Error: No .png images found in directory {args.image_dir}.")
    else:
        # 2. Execute inference and calculation
        results = run_tflite_int8_inference(args.model_path, image_paths, attr_calculator)

        # 3. Save results
        if results:
            df_results = pd.DataFrame(results)

            # Define column order: Filename -> 3 Attributes -> Model Score -> Status
            attribute_names = ATTRIBUTES_TO_CALCULATE
            column_order = ['filename'] + attribute_names + ['score', 'status']

            # Reindex to ensure correct column order and inclusion
            df_results = df_results.reindex(columns=column_order)

            df_results.to_csv(args.output_csv, index=False, float_format='%.4f')
            print(f"\n🎉 Batch Inference and Attribute Calculation complete. Results saved to {args.output_csv}.")
            print(f"The CSV output contains 4 score columns (3 attributes + 1 model score), all scaled to 0-100.")