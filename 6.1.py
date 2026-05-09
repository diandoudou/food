import tensorflow as tf
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from PIL import Image
import cv2 as cv  # Requires installation: pip install opencv-python

# --- Configuration and Scaling Ranges (Extracted from attribute_thresholds_summary.csv) ---
TARGET_SIZE = (224, 224)

# Min-Max Scaling Ranges (T_min (P5) to T_max (P95)) used to scale scores to 0-100.
SCALING_RANGES = {
    # Data extracted from uploaded CSV
    'clarity': {'min': 0.0759, 'max': 0.7937},
    'saturation': {'min': 0.1796, 'max': 0.6937},
    'warm': {'min': 0.1458, 'max': 0.938},
    'model_score': {'min': 0.0, 'max': 1.0}  # Assuming model raw output range is [0.0, 1.0]
}

# High-Aesthetic Target Averages (μ_High) for suggestion generation (raw values)
TARGET_AVERAGES_RAW = {
    'clarity': 0.3736,
    'saturation': 0.4201,
    'warm': 0.5707
}

# The list of attributes to calculate
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

        # Calculate scaling factor
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
    """
    Calculates the requested attributes, scales them to 0-100 using T_min/T_max,
    and returns both the scaled scores and the raw scores.
    """
    scaled_scores = {}
    raw_scores = {}

    for attr_name in ATTRIBUTES_TO_CALCULATE:
        try:
            method = getattr(attribute_calculator, attr_name)
            raw_value = method(image_path)
            raw_scores[attr_name] = float(raw_value)

            min_val = SCALING_RANGES[attr_name]['min']
            max_val = SCALING_RANGES[attr_name]['max']

            if max_val == min_val:
                scaled_value = 0.0
            else:
                # Min-Max Scaling Formula: (x - min) / (max - min) * 100
                normalized_value = (raw_value - min_val) / (max_val - min_val)
                scaled_value = np.clip(normalized_value * 100, 0, 100)

            scaled_scores[attr_name] = float(scaled_value)

        except Exception as e:
            print(f"⚠️ Attribute calculation failed ({attr_name}) for {Path(image_path).name}: {e}")
            raw_scores[attr_name] = np.nan
            scaled_scores[attr_name] = np.nan

    return scaled_scores, raw_scores


# --- Suggestion Function ---
def generate_suggestion(raw_scores):
    """
    Generates suggestions based on raw attribute scores compared to high-aesthetic
    target averages (μ_High from the CSV).
    """
    suggestions = []

    # Threshold for deviation (e.g., 15% deviation from the target average)
    DEVIATION_THRESHOLD_PERCENT = 0.15

    for attr, raw_value in raw_scores.items():
        if attr not in TARGET_AVERAGES_RAW or pd.isna(raw_value):
            continue

        target_avg = TARGET_AVERAGES_RAW[attr]
        # Tolerance based on percentage deviation
        threshold = target_avg * DEVIATION_THRESHOLD_PERCENT

        # Check if score is significantly lower than target
        if raw_value < (target_avg - threshold):
            suggestions.append(f"Increase {attr.capitalize()} (Target: {target_avg:.3f})")

        # Check if score is significantly higher than target
        elif raw_value > (target_avg + threshold):
            suggestions.append(f"Decrease {attr.capitalize()} (Target: {target_avg:.3f})")

    if not suggestions:
        return "Scores are generally balanced and close to high-aesthetic average."

    return "Consider adjusting: " + " | ".join(suggestions)


# --- Common Preprocessing Function ---
def preprocess_image_float(image_path, target_size):
    """Loads and preprocesses a single image for standard Keras float model input (Float32, 0-1)."""
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


# --- Inference Core Function: Keras H5 Float Model ---
def run_keras_float_inference(model_path, image_paths, attribute_calculator):
    """Batch inference for Keras H5 float model, calculate 3 attributes, and generate suggestions."""
    print(f"\n--- Starting Keras H5 Float Batch Inference ({len(image_paths)} images) ---")

    # Try loading Keras model
    try:
        # Load model without compiling, assuming it's a trained model
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        print(f"❌ Keras Model Loading Failed: {e}")
        return None

    results = []

    # Get model score scaling range
    score_min = SCALING_RANGES['model_score']['min']
    score_max = SCALING_RANGES['model_score']['max']

    for img_path in image_paths:

        # 1. Attribute Calculation (Scaled 0-100) & Raw Scores
        scaled_attr_scores, raw_attr_scores = calculate_and_scale_attributes(img_path, attribute_calculator)

        # 2. Generate Suggestion
        suggestion = generate_suggestion(raw_attr_scores)

        # 3. Model Inference Preprocessing (Float32)
        input_data = preprocess_image_float(img_path, TARGET_SIZE)

        base_result = {'filename': Path(img_path).name}
        base_result.update(scaled_attr_scores)  # Add 3 scaled attribute scores (0-100)
        base_result['suggestion'] = suggestion  # Add suggestion

        if input_data is None:
            base_result.update({'score': np.nan, 'status': 'Load Failed'})
            results.append(base_result)
            continue

        try:
            # 4. Model Prediction
            prediction = model.predict(input_data, verbose=0)
            raw_score = prediction[0][0]

            # 5. Scale Model Score to 0-100
            if score_max == score_min:
                scaled_score = 0.0
            else:
                normalized_score = (raw_score - score_min) / (score_max - score_min)
                scaled_score = np.clip(normalized_score * 100, 0, 100)  # Clip to [0, 100]

            base_result.update({'score': float(scaled_score), 'status': 'Success'})  # Model Score (0-100)

        except Exception as e:
            print(f"⚠️ Keras Inference Failed for {Path(img_path).name}: {e}")
            base_result.update({'score': np.nan, 'status': 'Inference Failed'})

        results.append(base_result)

    return results


# --- Main Execution Block ---
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Batch inference for Keras H5 Float model, calculating 3 attributes (clarity, saturation, warm), scaling scores (0-100), and generating aesthetic suggestions.")
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing all images to be scored.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the Keras model file (.h5).')
    parser.add_argument('--output_csv', type=str, default='inference_float_clarity_saturation_warm_suggestions.csv',
                        help='Path to save the output CSV file.')

    args = parser.parse_args()

    if not args.model_path.lower().endswith('.h5'):
        print(f"❌ Error: Model file must be a .h5 file.")
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
        results = run_keras_float_inference(args.model_path, image_paths, attr_calculator)

        # 3. Save results
        if results:
            df_results = pd.DataFrame(results)

            # Define column order: Filename -> 3 Attributes -> Model Score -> Suggestion -> Status
            attribute_names = ATTRIBUTES_TO_CALCULATE
            column_order = ['filename'] + attribute_names + ['score', 'suggestion', 'status']

            # Reindex to ensure correct column order and inclusion
            df_results = df_results.reindex(columns=column_order)

            df_results.to_csv(args.output_csv, index=False, float_format='%.4f')
            print(
                f"\n🎉 Keras H5 Float Model Inference, Attribute Calculation, and Suggestion Generation complete. Results saved to {args.output_csv}.")
            print(
                f"The CSV output contains 4 score columns (3 attributes + 1 model score), a suggestion column, and a status column.")