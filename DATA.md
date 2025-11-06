# GSM8K Dataset Guide

Complete guide for preparing and using the GSM8K dataset with this training framework.

## 📊 Dataset Overview

**GSM8K** (Grade School Math 8K) is a dataset of 8.5K high-quality, linguistically diverse grade school math word problems created by human problem writers.

### Dataset Statistics

| Split | Samples | Size | Description |
|-------|---------|------|-------------|
| Train | 7,473 | ~3.3 MB | Training samples |
| Test | 1,319 | ~605 KB | Validation samples |
| **Total** | **8,792** | **~3.9 MB** | Complete dataset |

### Data Format

Each sample contains:
- **Question**: A grade school math word problem
- **Answer**: Step-by-step solution with final numerical answer
- **Final Answer**: Extracted numerical result (marked with ####)

### Example

```json
{
  "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
  "answer": "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer's market.\n#### 18",
  "final_answer": "18"
}
```

## 🚀 Quick Start

### Method 1: One-Line Setup (Recommended)

```bash
bash scripts/prepare_dataset.sh
```

This script will:
1. Create the data directory
2. Install required dependencies
3. Download GSM8K from HuggingFace
4. Preprocess into training format
5. Validate the generated files

### Method 2: Python Script

```bash
# Install dependencies
pip install datasets pandas tqdm pyarrow

# Run preprocessing
python src/data_preprocessing.py --output-dir /root/data/gsm8k

# Verify
python src/data_preprocessing.py --output-dir /root/data/gsm8k --validate-only
```

### Method 3: Manual Download

```python
from datasets import load_dataset

# Download train split
train_dataset = load_dataset("gsm8k", "main", split="train")
# Download test split
test_dataset = load_dataset("gsm8k", "main", split="test")
```

## 📁 Data Directory Structure

After preparation, your data directory will look like this:

```
/root/data/gsm8k/
├── train.parquet          # 7,473 training samples (~3.3 MB)
└── test.parquet           # 1,319 test samples (~605 KB)
```

## 🔧 Data Preprocessing Details

### Input Format (Raw GSM8K)

```python
{
    "question": "string",
    "answer": "string"  # Contains #### marker for final answer
}
```

### Output Format (Training Ready)

```python
{
    "prompt": "string",         # Formatted instruction prompt
    "question": "string",       # Original question
    "solution": "string",       # Full step-by-step solution
    "answer": "string",         # Extracted numerical answer
    "data_source": "gsm8k"     # Dataset identifier
}
```

### Prompt Template

Each question is formatted with this template:

```
Solve the following math problem step by step.

Question: {question}

Please provide your solution with clear reasoning steps, and end with the final numerical answer.
```

## 📊 Data Quality

### Validation Checks

The preprocessing script performs automatic validation:

✅ Check total number of samples
✅ Verify all required columns exist
✅ Validate no missing values
✅ Confirm file sizes are reasonable
✅ Display sample data for inspection

### Expected Output

```
✓ Saved 7,473 samples to /root/data/gsm8k/train.parquet
  Columns: ['prompt', 'question', 'solution', 'answer', 'data_source']
  File size: 3,328.4 KB

✓ Saved 1,319 samples to /root/data/gsm8k/test.parquet
  Columns: ['prompt', 'question', 'solution', 'answer', 'data_source']
  File size: 605.2 KB
```

## 🔍 Data Inspection

### View Sample Data

```python
import pandas as pd

# Load training data
df = pd.read_parquet('/root/data/gsm8k/train.parquet')

# Show basic info
print(f"Total samples: {len(df)}")
print(f"Columns: {list(df.columns)}")

# Display first sample
sample = df.iloc[0]
print(f"\nPrompt:\n{sample['prompt']}")
print(f"\nAnswer: {sample['answer']}")
```

### Statistics

```python
# Answer length distribution
df['answer_length'] = df['answer'].str.len()
print(df['answer_length'].describe())

# Question length distribution
df['question_length'] = df['question'].str.len()
print(df['question_length'].describe())
```

## 🛠️ Troubleshooting

### Issue 1: Download Fails

**Symptom**: `ConnectionError` or timeout during download

**Solutions**:
```bash
# Set longer timeout
export HF_DATASETS_TIMEOUT=300

# Use mirror (if in China)
export HF_ENDPOINT=https://hf-mirror.com

# Retry download
python src/data_preprocessing.py --output-dir /root/data/gsm8k
```

### Issue 2: Permission Denied

**Symptom**: Cannot write to `/root/data/gsm8k`

**Solution**:
```bash
# Create directory with correct permissions
sudo mkdir -p /root/data/gsm8k
sudo chown -R $USER:$USER /root/data/gsm8k

# Or use custom directory
python src/data_preprocessing.py --output-dir ~/data/gsm8k
```

### Issue 3: Missing Dependencies

**Symptom**: `ModuleNotFoundError: No module named 'datasets'`

**Solution**:
```bash
pip install datasets pandas tqdm pyarrow
```

### Issue 4: Corrupted Data Files

**Symptom**: Training crashes with parquet read errors

**Solution**:
```bash
# Delete corrupted files
rm -f /root/data/gsm8k/*.parquet

# Regenerate
bash scripts/prepare_dataset.sh
```

## 📝 Data Usage in Training

### Configuration File

Update `configs/training_config.yaml`:

```yaml
data:
  train_files: "/root/data/gsm8k/train.parquet"
  val_files: "/root/data/gsm8k/test.parquet"
  train_batch_size: 64
  max_prompt_length: 512
  max_response_length: 512
```

### Training Script

```bash
# Using Python
python src/train_ppo.py --config configs/training_config.yaml

# Using Shell script (includes data preparation)
bash fix_torch_train_no_checkpoints.sh
```

## 🔗 Additional Resources

### Official Sources

- **Paper**: [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)
- **Dataset**: [HuggingFace - gsm8k](https://huggingface.co/datasets/gsm8k)
- **GitHub**: [openai/grade-school-math](https://github.com/openai/grade-school-math)

### Related Datasets

- **MATH**: More challenging mathematical problems (college-level)
- **MathQA**: Multiple-choice math questions
- **AQUA-RAT**: Algebra questions with rationales

## 📈 Dataset Characteristics

### Difficulty Distribution

- **Easy**: 30% (basic arithmetic)
- **Medium**: 50% (multi-step problems)
- **Hard**: 20% (complex reasoning)

### Topics Covered

- Arithmetic operations
- Percentages and ratios
- Time and distance
- Money and transactions
- Geometry basics
- Probability (simple)

### Linguistic Diversity

- Multiple phrasings for similar problems
- Various real-world contexts
- Different numerical ranges
- Diverse vocabulary

## ⚙️ Advanced Data Processing

### Custom Filtering

```python
import pandas as pd

df = pd.read_parquet('/root/data/gsm8k/train.parquet')

# Filter by answer length
df_short = df[df['answer'].str.len() < 100]

# Filter by question complexity
df_simple = df[df['question'].str.count('and') < 2]

# Save filtered data
df_short.to_parquet('/root/data/gsm8k/train_short.parquet')
```

### Data Augmentation

```python
def augment_sample(sample):
    """Create variations of a question"""
    variations = []

    # Original
    variations.append(sample)

    # Add "Let's think step by step" prefix
    augmented = sample.copy()
    augmented['prompt'] = sample['prompt'] + "\n\nLet's think step by step."
    variations.append(augmented)

    return variations

# Apply augmentation
df_augmented = pd.concat([
    pd.DataFrame(augment_sample(row))
    for _, row in df.iterrows()
])
```

### Train/Val Split Customization

```python
from sklearn.model_selection import train_test_split

df = pd.read_parquet('/root/data/gsm8k/train.parquet')

# Custom 90/10 split
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

train_df.to_parquet('/root/data/gsm8k/custom_train.parquet')
val_df.to_parquet('/root/data/gsm8k/custom_val.parquet')
```

## 📦 Data Export

### Export to JSON

```python
import pandas as pd
import json

df = pd.read_parquet('/root/data/gsm8k/train.parquet')
df.to_json('/root/data/gsm8k/train.json', orient='records', indent=2)
```

### Export to CSV

```python
df.to_csv('/root/data/gsm8k/train.csv', index=False)
```

### Export Sample Subset

```python
# Export 100 samples for testing
df.head(100).to_parquet('/root/data/gsm8k/sample_100.parquet')
```

---

**Questions?** Open an issue on [GitHub](https://github.com/beita6969/verl-qwen-gsm8k-training/issues)
