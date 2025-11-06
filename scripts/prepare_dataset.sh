#!/bin/bash
# GSM8K Dataset Preparation Script
# Automatically downloads and preprocesses GSM8K dataset for training

set -e

echo "=========================================="
echo "GSM8K Dataset Preparation"
echo "=========================================="
echo ""

# Configuration
DATA_DIR="${DATA_DIR:-/root/data/gsm8k}"
PYTHON_SCRIPT="${PYTHON_SCRIPT:-src/data_preprocessing.py}"

echo "Configuration:"
echo "  Data directory: $DATA_DIR"
echo "  Python script: $PYTHON_SCRIPT"
echo ""

# Step 1: Create data directory
echo "Step 1: Creating data directory..."
mkdir -p "$DATA_DIR"
echo "✓ Directory created: $DATA_DIR"
echo ""

# Step 2: Check dependencies
echo "Step 2: Checking Python dependencies..."
python3 -c "import datasets, pandas, tqdm" 2>/dev/null || {
    echo "Installing required packages..."
    pip install datasets pandas tqdm pyarrow -q
}
echo "✓ Dependencies installed"
echo ""

# Step 3: Download and preprocess GSM8K
echo "Step 3: Downloading and preprocessing GSM8K dataset..."
echo "This will download:"
echo "  - GSM8K train split (7,473 samples)"
echo "  - GSM8K test split (1,319 samples)"
echo ""

if [ -f "$PYTHON_SCRIPT" ]; then
    python3 "$PYTHON_SCRIPT" --output-dir "$DATA_DIR"
else
    echo "Warning: $PYTHON_SCRIPT not found, using inline preprocessing..."
    python3 << 'EOF'
import os
import json
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

def extract_answer(answer_text: str) -> str:
    """Extract numerical answer from GSM8K answer format"""
    if "####" in answer_text:
        return answer_text.split("####")[1].strip()
    return answer_text.strip()

def format_prompt(question: str) -> str:
    """Format question as instruction prompt"""
    return f"""Solve the following math problem step by step.

Question: {question}

Please provide your solution with clear reasoning steps, and end with the final numerical answer."""

def process_gsm8k_sample(sample: dict) -> dict:
    """Process single GSM8K sample into training format"""
    question = sample['question']
    answer = sample['answer']
    final_answer = extract_answer(answer)

    return {
        'prompt': format_prompt(question),
        'question': question,
        'solution': answer,
        'answer': final_answer,
        'data_source': 'gsm8k'
    }

# Process train and test splits
for split in ['train', 'test']:
    print(f"\nProcessing {split} split...")
    dataset = load_dataset("gsm8k", "main", split=split)

    processed_data = []
    for sample in tqdm(dataset):
        processed_sample = process_gsm8k_sample(sample)
        processed_data.append(processed_sample)

    df = pd.DataFrame(processed_data)
    output_file = os.path.join(os.environ.get('DATA_DIR', '/root/data/gsm8k'), f"{split}.parquet")
    df.to_parquet(output_file, index=False)

    print(f"✓ Saved {len(df)} samples to {output_file}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  File size: {os.path.getsize(output_file) / 1024:.1f} KB")
EOF
fi

echo ""
echo "=========================================="
echo "Dataset Preparation Complete!"
echo "=========================================="

# Step 4: Validate generated files
echo ""
echo "Validating generated files..."
echo ""

if [ -f "$DATA_DIR/train.parquet" ] && [ -f "$DATA_DIR/test.parquet" ]; then
    echo "✓ Files successfully created:"
    ls -lh "$DATA_DIR"/*.parquet
    echo ""

    # Show sample data
    echo "Sample from training data:"
    python3 << EOF
import pandas as pd
df = pd.read_parquet('$DATA_DIR/train.parquet')
print(f"Total training samples: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst sample:")
sample = df.iloc[0]
print(f"  Question: {sample['question'][:100]}...")
print(f"  Answer: {sample['answer']}")
EOF

    echo ""
    echo "✓ Dataset is ready for training!"
    echo ""
    echo "Usage:"
    echo "  python src/train_ppo.py --config configs/training_config.yaml"
    echo "  or"
    echo "  bash fix_torch_train_no_checkpoints.sh"
else
    echo "❌ Error: Dataset files not found!"
    exit 1
fi
