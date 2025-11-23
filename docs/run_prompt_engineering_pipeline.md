# 🎯 Prompt Engineering Pipeline

> **Part 2:** Extract named entities using prompt engineering with LLMs

---

## 📋 Overview

This pipeline uses prompt engineering techniques to extract Vietnamese Named Entity Recognition (NER) from text using large language models (LLMs). It leverages advanced prompting strategies without requiring model fine-tuning.

---

## 🚀 Quick Start

### Run Prompt Engineering Pipeline

```bash
python scripts/run_prompt_engineering_pipeline.py
```

---

## ⚙️ Configuration

### Basic Configuration (RAW Mode)

```python
from src.config import NERPromptEngineeringConfig, ExtractionMode

config = NERPromptEngineeringConfig(
    # Model settings
    model_name="Qwen/Qwen3-4B",

    # Extraction mode
    extraction_mode=ExtractionMode.RAW,
    enable_thinking=False,

    # Prompt settings
    add_schema=False,

    # Generation settings
    temperature=0.0,
    max_new_tokens=512,
)
```

### Configuration Templates

<details>
<summary><b>📝 RAW Mode (No Thinking)</b></summary>

```python
config = NERPromptEngineeringConfig(
    model_name="Qwen/Qwen3-4B",
    extraction_mode=ExtractionMode.RAW,
    enable_thinking=False,
)
# Fast extraction with direct JSON output
```
</details>

<details>
<summary><b>🤔 RAW Mode (With Thinking)</b></summary>

```python
config = NERPromptEngineeringConfig(
    model_name="Qwen/Qwen3-4B",
    extraction_mode=ExtractionMode.RAW,
    enable_thinking=True,
)
# Model reasons before extraction, may improve quality
```
</details>

<details>
<summary><b>🔒 Structured Output Mode (No Thinking)</b></summary>

```python
config = NERPromptEngineeringConfig(
    model_name="Qwen/Qwen3-4B",
    extraction_mode=ExtractionMode.STRUCTURED_OUTPUT,
    enable_thinking=False,
)
# Guaranteed valid JSON output with schema validation
```
</details>

<details>
<summary><b>🔒🤔 Structured Output Mode (With Thinking)</b></summary>

```python
config = NERPromptEngineeringConfig(
    model_name="Qwen/Qwen3-4B",
    extraction_mode=ExtractionMode.STRUCTURED_OUTPUT,
    enable_thinking=True,
)
# Best quality: reasoning + guaranteed valid JSON
```
</details>

---

## 📂 Input Requirements

### Required Files

Before starting evaluation, ensure these files exist:

| File | Path | Description |
|------|------|-------------|
| Test Data | `data/processed/test.json` | Test dataset for evaluation |

### Data Format

```json
{
  "text": "Article text in Vietnamese...",
  "entities": {
    "person": ["Nguyễn Văn A", "Trần Thị B"],
    "organizations": ["Công ty ABC"],
    "address": ["Hà Nội"]
  }
}
```

> 💡 **Tip**: Run [data preparation](run_data_preparation.md) first to generate these files.

---

## 📝 Logs and Monitoring

### Evaluation Logs

The pipeline writes detailed logs to:

📄 **[logs/run_prompt_engineering_pipeline.log](logs/run_prompt_engineering_pipeline.log)**


---

## 🎯 Performance Metrics

### Extraction Metrics

Monitor these metrics during evaluation:

| Metric | Description | Good Value |
|--------|-------------|------------|
| `precision` | Correct predictions / Total predictions | > 0.80 |
| `recall` | Correct predictions / Total ground truth | > 0.75 |
| `f1` | Harmonic mean of precision & recall | > 0.78 |
| `throughput` | Samples processed per second | > 1.0 |

### Performance by Extraction Mode

| Configuration | Throughput | F1 Score | Memory |
|--------------|------------|----------|--------|
| RAW (no thinking) | ~2.5 samples/s | ~0.81 | ~4GB |
| RAW (with thinking) | ~1.8 samples/s | ~0.83 | ~4GB |
| Structured (no thinking) | ~2.2 samples/s | ~0.83 | ~4GB |
| Structured (with thinking) | ~1.5 samples/s | ~0.85 | ~4GB |

---

## 🛠️ Troubleshooting

### Common Issues

<details>
<summary><b>❌ Test Dataset Not Found</b></summary>

**Symptoms:**
```
FileNotFoundError: Test dataset not found
```

**Solution:**
Run data preparation first:
```bash
python scripts/run_data_preparation.py
```
</details>

<details>
<summary><b>❌ Out of Memory (OOM)</b></summary>

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Use smaller model: `model_name="Qwen/Qwen3-4B"` instead of larger models
2. Reduce max tokens: `max_new_tokens=256`
3. Enable quantization in model loading
4. Process samples in smaller batches
</details>

<details>
<summary><b>❌ Invalid JSON Output</b></summary>

**Symptoms:**
```
JSONDecodeError: Expecting value
```

**Solutions:**
1. Use structured output mode: `extraction_mode=ExtractionMode.STRUCTURED_OUTPUT`
2. Add schema to prompt: `add_schema=True`
3. Reduce temperature: `temperature=0.0`
4. Check model compatibility with structured output
</details>

<details>
<summary><b>❌ Low F1 Score</b></summary>

**Solutions:**
1. Enable thinking mode: `enable_thinking=True`
2. Use structured output: `extraction_mode=ExtractionMode.STRUCTURED_OUTPUT`
3. Add schema to prompt: `add_schema=True`
4. Try different models
5. Check data quality and format
</details>

---

## 🔍 Extraction Modes Explained

### RAW Mode
- Model generates free-form JSON
- Faster extraction
- May produce invalid JSON occasionally
- Good for quick testing

### STRUCTURED_OUTPUT Mode
- Uses schema-based generation
- Guarantees valid JSON
- Slightly slower
- Recommended for production

### Thinking Mode
- Model shows reasoning process
- Improves extraction quality
- Increases processing time
- Useful for complex entities

## ⏱️ Evaluation Time Estimates

| Configuration | Dataset Size | GPU | Time |
|--------------|--------------|-----|------|
| RAW (no thinking) | 100 samples | L40S 48GB | ~40s |
| RAW (with thinking) | 100 samples | L40S 48GB | ~55s |
| Structured (no thinking) | 100 samples | L40S 48GB | ~45s |
| Structured (with thinking) | 100 samples | L40S 48GB | ~65s |
| RAW (no thinking) | 1000 samples | L40S 48GB | ~6.5 min |

---

## 📊 Comparison with Other Methods

| Method | Speed | Quality | Training Required | Memory |
|--------|-------|---------|------------------|--------|
| **Prompt Engineering** | ⭐⭐⭐ | ⭐⭐⭐ | ❌ No | Low |
| Fine-Tuning | ⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ Yes | Medium |
| RAG | ⭐⭐ | ⭐⭐⭐⭐ | ❌ No | Medium |

**When to use Prompt Engineering:**
- Quick prototyping and testing
- No labeled training data available
- Need immediate results without training
- Limited computational resources
- Experimenting with different approaches

