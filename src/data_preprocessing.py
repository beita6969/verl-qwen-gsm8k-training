#!/usr/bin/env python3
"""
GSM8K Dataset Preprocessing
Converts GSM8K dataset to format required for verl-agent PPO training
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


def extract_answer(answer_text: str) -> str:
    """Extract numerical answer from GSM8K answer format"""
    # GSM8K answers are in format: "#### 42"
    if "####" in answer_text:
        return answer_text.split("####")[1].strip()
    return answer_text.strip()


def format_prompt(question: str) -> str:
    """Format question as instruction prompt"""
    return f"""Solve the following math problem step by step.

Question: {question}

Please provide your solution with clear reasoning steps, and end with the final numerical answer.
"""


def process_gsm8k_sample(sample: Dict) -> Dict:
    """Process single GSM8K sample into training format"""
    question = sample['question']
    answer = sample['answer']

    # Extract final numerical answer
    final_answer = extract_answer(answer)

    return {
        'prompt': format_prompt(question),
        'question': question,
        'solution': answer,
        'answer': final_answer,
        'data_source': 'gsm8k'
    }


def download_and_process_gsm8k(output_dir: Path, split: str = "train"):
    """Download and process GSM8K dataset"""
    print(f"Downloading GSM8K {split} split...")
    dataset = load_dataset("gsm8k", "main", split=split)

    print(f"Processing {len(dataset)} samples...")
    processed_data = []

    for sample in tqdm(dataset):
        processed_sample = process_gsm8k_sample(sample)
        processed_data.append(processed_sample)

    # Convert to DataFrame and save as parquet
    df = pd.DataFrame(processed_data)
    output_file = output_dir / f"{split}.parquet"
    df.to_parquet(output_file, index=False)

    print(f"✓ Saved {len(df)} samples to {output_file}")
    print(f"  - Columns: {list(df.columns)}")
    print(f"  - File size: {output_file.stat().st_size / 1024:.1f} KB")

    return df


def validate_dataset(parquet_file: Path):
    """Validate processed dataset"""
    print(f"\nValidating {parquet_file}...")
    df = pd.read_parquet(parquet_file)

    print(f"Dataset statistics:")
    print(f"  - Total samples: {len(df)}")
    print(f"  - Columns: {list(df.columns)}")
    print(f"  - Missing values: {df.isnull().sum().sum()}")

    # Show sample
    print(f"\nSample entry:")
    sample = df.iloc[0]
    print(f"  Question: {sample['question'][:100]}...")
    print(f"  Answer: {sample['answer']}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Preprocess GSM8K dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/root/data/gsm8k",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing dataset"
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.validate_only:
        validate_dataset(output_dir / "train.parquet")
        validate_dataset(output_dir / "test.parquet")
    else:
        # Process train and test splits
        print("=" * 60)
        print("Processing GSM8K Dataset")
        print("=" * 60)

        train_df = download_and_process_gsm8k(output_dir, split="train")
        test_df = download_and_process_gsm8k(output_dir, split="test")

        print("\n" + "=" * 60)
        print("Dataset Processing Complete!")
        print("=" * 60)
        print(f"Train samples: {len(train_df)}")
        print(f"Test samples: {len(test_df)}")
        print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
