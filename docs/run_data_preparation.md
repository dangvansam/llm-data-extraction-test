# 📊 Data Preparation Pipeline

> **Part 0:** Extract entities from raw article data using Langextract pipeline with third-party model

---

## 📋 Overview

This pipeline processes raw Vietnamese news articles and extracts named entities (Person, Organizations, Address) using the Langextract method with a pre-trained LLM.

---

## 🚀 Quick Start

### Run Data Preparation

```bash
python scripts/data_preparation.py --categories "Chinh tri Xa hoi"
```

---

## ⚙️ Configuration

The data preparation pipeline uses the following default settings:

```python
from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

RAW_DATA_DIR = "data/raw"          # Input: Raw article data
PROCESSED_DATA_DIR = "data/processed"  # Output: Processed datasets

# Categories to process
categories = ["Chinh tri Xa hoi"]
```

---

## 📂 Input/Output

### Input
- **Location**: `data/raw/`
- **Format**: Raw Vietnamese news articles
- **Categories**: Configurable (default: "Chinh tri Xa hoi")

### Output
The pipeline generates the following files in `data/processed/`:

| File | Description | Format |
|------|-------------|--------|
| `train.json` | Training dataset | JSON |
| `test.json` | Test dataset | JSON |
| `train_finetuning_chat.jsonl` | Fine-tuning training data (chat format) | JSONL |
| `test_finetuning_chat.jsonl` | Fine-tuning test data (chat format) | JSONL |

---

## 📊 Data Format

### Standard JSON Format (`train.json`, `test.json`)
```json
{
  "text": "Article text content...",
  "entities": {
    "person": ["Nguyễn Văn A", "Trần Thị B"],
    "organizations": ["Công ty ABC", "Bộ Giáo dục"],
    "address": ["Hà Nội", "Thành phố Hồ Chí Minh"]
  }
}
```

### Chat Format (`train_finetuning_chat.jsonl`, `test_finetuning_chat.jsonl`)
```json
{
  "messages": [
    {
      "role": "user",
      "content": "Extract entities from: Article text..."
    },
    {
      "role": "assistant",
      "content": "{\"person\": [...], \"organizations\": [...], \"address\": [...]}"
    }
  ]
}
```

---

## 📝 Logs

### View Processing Logs

The pipeline writes detailed logs to:

📄 **[logs/data_preparation.log](logs/data_preparation.log)**

This log file contains:
- Processing progress for each article
- Entity extraction results
- Dataset statistics
- Error messages (if any)
- Performance metrics

---

## 🔍 Processing Steps

The data preparation pipeline performs the following steps:

1. **📖 Load Raw Data**
   - Reads articles from `data/raw/`
   - Filters by specified categories

2. **🤖 Extract Entities**
   - Uses Langextract method with LLM
   - Extracts: Person, Organizations, Address
   - Validates extracted entities

3. **✂️ Split Dataset**
   - Splits into train/test sets
   - Default ratio: 80/20

4. **💾 Save Processed Data**
   - Standard format → `train.json`, `test.json`
   - Chat format → `train_finetuning_chat.jsonl`, `test_finetuning_chat.jsonl`

5. **📈 Generate Statistics**
   - Dataset size
   - Entity distribution
   - Category breakdown

---

## 📊 Expected Output

After successful execution, you should see:

```
✅ Data preparation complete!
📁 Output directory: data/processed/

📊 Dataset Statistics:
   Total samples: 1100
   Train samples: 1000
   Test samples: 100

📈 Entity Distribution:
   Person: 2,453 entities
   Organizations: 1,876 entities
   Address: 3,142 entities
```