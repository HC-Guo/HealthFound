#!/usr/bin/env python3
"""
Convert math_dapo dataset to use graded reward function.
This script modifies the data_source field to enable format-aware scoring.
"""

import pandas as pd
import sys

def convert_dataset(input_path: str, output_path: str, reward_type: str = "graded"):
    """Convert dataset to use graded reward.

    Args:
        input_path: Path to input parquet file
        output_path: Path to output parquet file
        reward_type: Type of reward ("graded" or "original")
    """
    print(f"Loading dataset from {input_path}...")
    df = pd.read_parquet(input_path)

    print(f"Original data_source values: {df['data_source'].unique()}")

    if reward_type == "graded":
        # Change data_source to use graded reward
        df['data_source'] = 'math_dapo_graded'
        print("Changed data_source to 'math_dapo_graded' (format-aware scoring)")
    else:
        # Keep original binary reward
        print("Keeping original binary reward")

    print(f"Saving to {output_path}...")
    df.to_parquet(output_path, index=False)

    print(f"✓ Done! Dataset saved with {len(df)} samples")
    print(f"  New data_source: {df['data_source'].unique()}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_to_graded_reward.py <input_path> [output_path] [reward_type]")
        print("Example: python convert_to_graded_reward.py /root/verl/data/dapo-math-17k.parquet")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else input_path.replace(".parquet", "-graded.parquet")
    reward_type = sys.argv[3] if len(sys.argv) > 3 else "graded"

    convert_dataset(input_path, output_path, reward_type)
