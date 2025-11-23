# Langextract Entity Extraction Pipeline

This guide explains how to use the Langextract pipeline to extract named entities from raw Vietnamese news articles.

## Overview

The pipeline processes raw text files containing Vietnamese news articles and extracts three types of named entities:
- **person**: Names of people (individuals, politicians, celebrities, etc.)
- **organizations**: Companies, institutions, organizations, agencies
- **address**: Locations, places, addresses, geographical names

## Data Structure

Your raw data should be organized as follows:

```
data/raw/
├── train/
│   ├── Doi song/        # Lifestyle category
│   ├── Van hoa/         # Culture category
│   ├── Kinh doanh/      # Business category
│   └── ...              # Other categories
└── test/
    ├── Doi song/
    ├── Van hoa/
    └── ...
```

Each category folder contains `.txt` files with article content.

## Output Format

The pipeline generates three types of output files:

### 1. JSON Dataset (`langextract_train.json`)
```json
[
  {
    "file_name": "DS_ VNE_ (1000).txt",
    "category": "Doi song",
    "text": "Article content...",
    "entities": {
      "person": ["Trương Hồi", "Phan Thị Bến"],
      "organizations": ["Bệnh viện T.Ư Huế"],
      "address": ["Huế", "Quảng Điền", "Phú Lộc"]
    }
  },
  ...
]
```

### 2. JSONL Dataset (`langextract_train.jsonl`)
Same structure but one JSON object per line (useful for streaming and LLM training).

### 3. Finetuning Format (`langextract_train_finetuning.jsonl`)
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a named entity recognition system..."
    },
    {
      "role": "user",
      "content": "Extract entities from this text:\n\n..."
    },
    {
      "role": "assistant",
      "content": "{\"person\": [...], \"organizations\": [...], \"address\": [...]}"
    }
  ]
}
```

## Usage

### Prerequisites

1. Set up your Gemini API key:
```bash
export GEMINI_API_KEY="your-api-key-here"
```

Or create a `.env` file:
```
GEMINI_API_KEY=your-api-key-here
```

2. Ensure you have the required dependencies installed (see `pyproject.toml`).

### Quick Test

Test the pipeline on a small sample first:

```bash
python scripts/test_extraction_sample.py
```

This will process 3-5 articles to verify everything works correctly.

### Process Specific Categories

Process only "Doi song" category for both train and test:

```bash
python scripts/extract_entities_from_raw.py --categories "Doi song"
```

Process multiple specific categories:

```bash
python scripts/extract_entities_from_raw.py --categories "Doi song" "Van hoa" "Kinh doanh"
```

### Process All Categories

Process all available categories in both train and test splits:

```bash
python scripts/extract_entities_from_raw.py --all
```

### Process Specific Split

Process only the train split:

```bash
python scripts/extract_entities_from_raw.py --split train --all
```

Process only the test split:

```bash
python scripts/extract_entities_from_raw.py --split test --categories "Doi song"
```

### Advanced Options

Customize extraction quality and performance:

```bash
python scripts/extract_entities_from_raw.py \
  --all \
  --passes 3 \          # More passes = better recall (default: 2)
  --workers 6 \         # More workers = faster processing (default: 4)
  --output-dir custom/path
```

### List Available Categories

See what categories are available:

```bash
python scripts/extract_entities_from_raw.py --list-categories
```

## Output Files

After processing, you'll find these files in `data/processed/`:

- `langextract_train.json` - Training dataset (JSON format)
- `langextract_train.jsonl` - Training dataset (JSONL format)
- `langextract_train_finetuning.jsonl` - Training data for LLM finetuning
- `langextract_test.json` - Test dataset (JSON format)
- `langextract_test.jsonl` - Test dataset (JSONL format)
- `langextract_test_finetuning.jsonl` - Test data for LLM evaluation
- `langextract_statistics.json` - Overall statistics

## Statistics

The statistics file contains information about:
- Total number of samples processed
- Entity counts by type (person, organizations, address)
- Average entities per sample
- Text length statistics
- Category distribution

Example:
```json
{
  "train": {
    "total_samples": 500,
    "total_entities": 1250,
    "entity_counts": {
      "person": 400,
      "organizations": 350,
      "address": 500
    },
    "avg_entities_per_sample": 2.5
  }
}
```

## Using the Extracted Data

### For Evaluation

Use the JSON/JSONL files to evaluate other NER models:

```python
from src.services.data_processor import DataProcessorService

# Load test dataset
test_data = DataProcessorService.load_dataset("data/processed/langextract_test.json")

# Use for evaluation
for sample in test_data:
    text = sample["text"]
    gold_entities = sample["entities"]
    # Run your model and compare with gold_entities
```

### For Finetuning

Use the finetuning files with LLM training frameworks:

```python
# For Hugging Face transformers
from datasets import load_dataset

dataset = load_dataset("json", data_files={
    "train": "data/processed/langextract_train_finetuning.jsonl",
    "test": "data/processed/langextract_test_finetuning.jsonl"
})

# Use with your finetuning pipeline
```

## Configuration

You can customize the extraction behavior in [src/config.py](../src/config.py):

```python
@dataclass
class NERLangExtractConfig(BaseNERConfig):
    model_id: str = "gemini-2.0-flash"  # Gemini model to use
    extraction_passes: int = 2           # Number of extraction passes
    max_workers: int = 4                 # Parallel workers
    max_char_buffer: int = 2000          # Chunk size for long docs
```

## Programmatic Usage

You can also use the components programmatically:

```python
from src.langextract_pipeline import LangExtractNERExtractor
from src.services.data_processor import DataProcessorService

# Initialize extractor
extractor = LangExtractNERExtractor()

# Extract from text
text = "Your article text here..."
entities = extractor.extract_entities(text)

# Process a folder of articles
articles = DataProcessorService.load_articles_from_folder("path/to/folder")
for article in articles:
    entities = extractor.extract_entities(article["text"])
    # Do something with entities
```

## Tips for Best Results

1. **Start Small**: Test with one category first before processing all categories
2. **Monitor API Usage**: Langextract uses Gemini API which has rate limits
3. **Adjust Passes**: More extraction passes = better recall but slower and more API calls
4. **Use Workers Wisely**: More workers = faster but may hit rate limits
5. **Check Statistics**: Review the statistics file to ensure quality

## Troubleshooting

### Rate Limit Errors
If you hit API rate limits, reduce the number of workers:
```bash
python scripts/extract_entities_from_raw.py --all --workers 2
```

### Memory Issues
For large datasets, process categories one at a time:
```bash
for category in "Doi song" "Van hoa" "Kinh doanh"; do
  python scripts/extract_entities_from_raw.py --categories "$category"
done
```

### Encoding Issues
The scripts handle both UTF-8 and Latin-1 encodings automatically. If you encounter other encodings, you may need to convert your files first.

## Next Steps

After extracting entities:

1. **Evaluate Quality**: Review some samples manually to check extraction quality
2. **Use for Evaluation**: Compare against other NER approaches (RAG, Finetuning, Prompt Engineering)
3. **Finetune LLM**: Use the finetuning files to train your own NER model
4. **Iterate**: Adjust extraction parameters and re-run if needed
