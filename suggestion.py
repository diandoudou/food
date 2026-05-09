# filename: visualize_attribute_comparison.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# List of 10 food attributes
ATTRIBUTE_NAMES = [
    "brightness", "saturation", "contrast", "clarity", "warm", "colourfulness",
    "size_difference", "color_difference", "texture_difference", "segment_colorfulness"
]


def load_data(high_scores_path, low_scores_path):
    """Loads data for high-scoring and low-scoring groups."""
    try:
        df_high = pd.read_csv(high_scores_path)
        df_low = pd.read_csv(low_scores_path)
        return df_high, df_low
    except FileNotFoundError:
        print(
            "❌ ERROR: Could not find G_high_scores.csv or G_low_scores.csv. Please ensure they are in the 'output/' directory.")
        return None, None


def plot_and_analyze_attributes(df_high, df_low):
    """
    Generates Kernel Density Estimate (KDE) plots for each attribute comparing G_High and G_Low,
    and exports a summary CSV of the calculated P5 and P95 thresholds.
    """
    # ⭐️ FIX: Use plain attribute names as the columns are named without '_score' suffix
    attribute_cols = ATTRIBUTE_NAMES

    # Create the output directory for visualizations
    output_dir = 'output/visualizations'
    os.makedirs(output_dir, exist_ok=True)

    # Configure plotting style
    sns.set_style("whitegrid")

    print("\n========================================")
    print(" 1. Generating Distribution Plots (KDE)")
    print("========================================")

    # List to store calculated thresholds (for printing and exporting CSV)
    threshold_data_list = []

    for i, attr_col in enumerate(tqdm(attribute_cols, desc="Processing Attributes")):
        attr_name = attr_col  # e.g., 'brightness'

        # Extract data for the current attribute - uses correct column name
        data_high = df_high[attr_col].dropna()
        data_low = df_low[attr_col].dropna()

        # --- Threshold Calculation (P5/P95 of the High Group) ---
        t_min = data_high.quantile(0.05)
        t_max = data_high.quantile(0.95)
        mu_high = data_high.mean()
        mu_low = data_low.mean()

        # Collect data for the summary table
        threshold_data_list.append({
            'Attribute': attr_name,
            'T_min (P5)': t_min,
            'T_max (P95)': t_max,
            'Ideal_Range_90%': f"[{t_min:.4f}, {t_max:.4f}]",
            'μ_High': mu_high,
            'μ_Low': mu_low,
        })

        # --- Plotting ---
        plt.figure(figsize=(10, 6))

        # Plot KDE for G_High
        sns.kdeplot(data_high, label=f'G_High (N={len(data_high)})', fill=True, alpha=0.5, color='green')
        # Plot KDE for G_Low
        sns.kdeplot(data_low, label=f'G_Low (N={len(data_low)})', fill=True, alpha=0.5, color='red')

        # Annotate P5 and P95 thresholds
        plt.axvline(t_min, color='blue', linestyle='--', linewidth=1.5, label=f'T_min (P5)={t_min:.3f}')
        plt.axvline(t_max, color='orange', linestyle='--', linewidth=1.5, label=f'T_max (P95)={t_max:.3f}')

        # Annotate means of G_High and G_Low
        plt.axvline(mu_high, color='green', linestyle=':', linewidth=1.5, label=f'μ_High={mu_high:.3f}')
        plt.axvline(mu_low, color='red', linestyle=':', linewidth=1.5, label=f'μ_Low={mu_low:.3f}')

        plt.title(f'Distribution Comparison for {attr_col}')
        plt.xlabel('Attribute Score (0 to 1)')
        plt.ylabel('Density')
        plt.xlim(0, 1)
        plt.legend()

        # Save plot
        plot_filename = os.path.join(output_dir, f'{attr_col}_kde_comparison.png')
        plt.savefig(plot_filename)
        plt.close()

    print(f"✔ All attribute distribution plots saved to {output_dir}/.")

    # 2. Output Threshold Analysis Results (Print to console)
    print("\n========================================")
    print(" 2. Attribute Threshold Analysis Table")
    print("========================================")

    df_thresholds_output = pd.DataFrame(threshold_data_list)

    # Prepare DataFrame for console printing with formatting
    df_thresholds_print = df_thresholds_output[
        ['Attribute', 'T_min (P5)', 'T_max (P95)', 'Ideal_Range_90%', 'μ_High', 'μ_Low']].copy()
    # Apply formatting to numerical columns for clean console output
    for col in ['T_min (P5)', 'T_max (P95)', 'μ_High', 'μ_Low']:
        df_thresholds_print[col] = df_thresholds_print[col].apply(lambda x: f"{x:.4f}")

    print("\n📊 Key Diagnostic Thresholds (Based on P5/P95 of G_High)")
    print(
        "----------------------------------------------------------------------------------------------------------------------")
    print(df_thresholds_print.to_string(index=False))
    print(
        "----------------------------------------------------------------------------------------------------------------------")
    print("\n**Interpretation Guide**:")
    print(
        "1. **T_min (P5)**: The critical point for diagnosing 'under-processing' of this attribute. Scores below this value significantly reduce aesthetic appeal.")
    print(
        "2. **T_max (P95)**: The critical point for diagnosing 'over-processing' of this attribute. Scores above this value might lead to distortion or unnatural results.")
    print(
        "3. **μ_High/μ_Low**: Mean scores of the High/Low groups. If μ_Low is far from the Ideal_Range, the attribute is a key defect in low-scoring images.")

    # 3. Export Threshold Analysis Results to a CSV file
    csv_filename = os.path.join('output', 'attribute_thresholds_summary.csv')

    # Ensure saved CSV columns contain the original float values (excluding the string 'Ideal_Range_90%')
    df_thresholds_save = df_thresholds_output[['Attribute', 'T_min (P5)', 'T_max (P95)', 'μ_High', 'μ_Low']].copy()

    df_thresholds_save.to_csv(csv_filename, index=False, float_format='%.4f')

    print(f"\n✅ Attribute threshold summary successfully exported to {csv_filename}")


# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":

    # Assumes CSV files are located in the 'output/' directory
    HIGH_PATH = os.path.join('output', 'G_high_scores.csv')
    LOW_PATH = os.path.join('output', 'G_low_scores.csv')

    df_high, df_low = load_data(HIGH_PATH, LOW_PATH)

    if df_high is not None and df_low is not None:
        plot_and_analyze_attributes(df_high, df_low)