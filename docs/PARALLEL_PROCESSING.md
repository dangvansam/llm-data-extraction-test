# Parallel Processing & Crash Recovery

The extraction pipeline now supports parallel processing with automatic crash recovery and resume capability.

## Key Features

### 1. Parallel Processing
Articles are processed in parallel using ThreadPoolExecutor, significantly speeding up extraction:
- Multiple articles processed simultaneously
- Configurable number of workers with `--workers` flag
- Progress bar shows real-time completion status

### 2. Incremental Saving
Each article is saved immediately after processing:
- Individual JSON files stored in `data/processed/{split}/{category}/json/{article_name}.json`
- No data loss if the process crashes
- Can resume from where it stopped

### 3. Automatic Resume
The script automatically skips already-processed articles:
- Checks for existing JSON files before processing
- Loads cached results if available
- Only processes new/unprocessed articles

### 4. Final Aggregation
After all articles are processed, they're combined into single files:
- `langextract_train.json` - All training samples
- `langextract_test.json` - All test samples
- Plus JSONL and finetuning formats

## Directory Structure

```
data/processed/
├── train/
│   ├── Doi song/
│   │   └── json/
│   │       ├── DS_ VNE_ (1000).json
│   │       ├── DS_ VNE_ (1002).json
│   │       └── ...
│   ├── Van hoa/
│   │   └── json/
│   │       └── ...
│   └── ...
├── test/
│   ├── Doi song/
│   │   └── json/
│   │       └── ...
│   └── ...
├── langextract_train.json          # Combined training data
├── langextract_test.json           # Combined test data
├── langextract_train.jsonl
├── langextract_test.jsonl
├── langextract_train_finetuning.jsonl
└── langextract_test_finetuning.jsonl
```

## Usage Examples

### Basic Processing

```bash
# Process with default 4 workers
python scripts/extract_entities_from_raw.py --categories "Doi song"

# Process with more parallelism (faster but uses more API quota)
python scripts/extract_entities_from_raw.py --categories "Doi song" --workers 8
```

### Resume After Crash

If the process crashes or is interrupted, simply run the same command again:

```bash
# This will automatically skip already-processed articles
python scripts/extract_entities_from_raw.py --categories "Doi song"
```

The script will:
1. Check `data/processed/train/Doi song/json/` for existing files
2. Skip articles that are already processed
3. Only process remaining articles
4. Combine all results (old + new) into final JSON files

### Check Progress

Check how many articles have been processed:

```bash
python scripts/utils/check_progress.py --all
```

Or check specific category:

```bash
python scripts/utils/check_progress.py --split train --category "Doi song"
```

### Process All Categories with Resilience

```bash
# Process all categories - if it crashes, just restart
python scripts/extract_entities_from_raw.py --all --workers 6

# If interrupted, run again - it will resume automatically
python scripts/extract_entities_from_raw.py --all --workers 6
```

## Performance Tuning

### Workers vs API Rate Limits

- **Fewer workers (2-4)**: Slower but less likely to hit rate limits
- **More workers (6-10)**: Faster but may hit API rate limits

Example with conservative settings:
```bash
python scripts/extract_entities_from_raw.py --all --workers 3 --passes 2
```

Example with aggressive settings:
```bash
python scripts/extract_entities_from_raw.py --all --workers 8 --passes 3
```

### Monitoring

The script shows:
- Progress bar per category
- Number of already-processed samples found
- Failed articles (if any)
- Final statistics

```
Processing category: Doi song (train)
Found 45 already processed samples - will skip those
Doi song: 100%|████████████████| 100/100 [05:23<00:00,  3.23s/it]
Processed 100/100 articles from Doi song
```

## Individual JSON File Format

Each article gets its own JSON file:

**File**: `data/processed/train/Doi song/json/DS_ VNE_ (1000).json`

```json
{
  "file_name": "DS_ VNE_ (1000).txt",
  "category": "Doi song",
  "text": "Full article text...",
  "entities": {
    "person": ["Trương Hồi", "Phan Thị Bến"],
    "organizations": ["Bệnh viện T.Ư Huế"],
    "address": ["Huế", "Quảng Điền"]
  }
}
```

## Advantages

### 1. Crash Resilience
- ✅ Process can be interrupted at any time
- ✅ No need to reprocess completed articles
- ✅ Save API quota and time on retries

### 2. Parallel Speed
- ✅ Process multiple articles simultaneously
- ✅ Significantly faster than sequential processing
- ✅ Adjustable parallelism based on resources

### 3. Easy Debugging
- ✅ Individual files make it easy to inspect results
- ✅ Can manually fix or delete specific problematic files
- ✅ Clear tracking of what's been processed

### 4. Incremental Progress
- ✅ See results immediately as they're processed
- ✅ Don't have to wait for entire batch to complete
- ✅ Can analyze partial results while processing continues

## Troubleshooting

### Problem: Too many API errors

**Solution**: Reduce workers
```bash
python scripts/extract_entities_from_raw.py --categories "Doi song" --workers 2
```

### Problem: Want to reprocess a specific article

**Solution**: Delete its JSON file and run again
```bash
rm "data/processed/train/Doi song/json/DS_ VNE_ (1000).json"
python scripts/extract_entities_from_raw.py --categories "Doi song"
```

### Problem: Want to start fresh

**Solution**: Delete the category's json folder
```bash
rm -rf "data/processed/train/Doi song/json"
python scripts/extract_entities_from_raw.py --categories "Doi song"
```

### Problem: Check which articles failed

**Solution**: The script logs failed articles at the end
```
Failed articles (3): DS_TN_T_ (582).txt, DS_TN_T_ (614).txt, DS_TN_T_ (721).txt...
```

## Best Practices

1. **Start with one category** to test settings:
   ```bash
   python scripts/extract_entities_from_raw.py --categories "Doi song" --workers 4
   ```

2. **Check progress frequently** during large batches:
   ```bash
   python scripts/utils/check_progress.py --all
   ```

3. **Use appropriate worker count** based on your API quota:
   - Small quota: `--workers 2`
   - Medium quota: `--workers 4`
   - Large quota: `--workers 6-8`

4. **Resume safely** after interruptions:
   ```bash
   # Just run the same command - it will resume automatically
   python scripts/extract_entities_from_raw.py --all
   ```

5. **Keep individual JSON files** until you're satisfied with results:
   - They serve as backup
   - Easy to inspect and debug
   - Can regenerate combined files anytime

## Regenerating Combined Files

If you have all individual JSON files and want to regenerate the combined files:

```python
import json
from pathlib import Path

# Load all individual files
samples = []
for json_file in Path("data/processed/train/Doi song/json").glob("*.json"):
    with open(json_file) as f:
        samples.append(json.load(f))

# Save combined
with open("data/processed/langextract_train.json", "w", encoding="utf-8") as f:
    json.dump(samples, f, ensure_ascii=False, indent=2)
```

Or just run the extraction script again - it will skip processing and just recombine the files.
