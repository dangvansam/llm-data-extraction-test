# NER Extraction Methods - Evaluation Guide

This guide explains how to evaluate and compare different NER extraction methods in this project.

## Available Extraction Methods

### 1. RAW Mode
- **Description**: Standard text generation without structured output constraints
- **Dependencies**: `transformers`, `torch`
- **GPU**: Optional (can run on CPU)
- **Pros**: Simple, no extra dependencies, works on CPU
- **Cons**: Slower, requires JSON parsing, no guarantee of valid output

### 2. STRUCTURED_OUTPUT Mode (vLLM)
- **Description**: Uses vLLM with JSON schema validation for structured output
- **Dependencies**: `vllm`
- **GPU**: Required
- **Pros**: Very fast (10-50x speedup), guaranteed valid JSON, batch processing
- **Cons**: Requires GPU, requires vLLM installation

### 3. OUTLINES Mode
- **Description**: Uses Outlines library with Pydantic models for constraint-guided generation
- **Dependencies**: `outlines`
- **GPU**: Optional (can run on CPU)
- **Pros**: Type-safe, guaranteed valid JSON, works on CPU
- **Cons**: Slightly slower than vLLM

## Configuration Options

Each method can be configured with:

- **model_name**: Which LLM to use (e.g., "Qwen/Qwen3-4B")
- **use_chat_format**: Use chat message format (system/user/assistant)
- **add_schema**: Include JSON schema in the prompt
- **temperature**: Sampling temperature (lower = more deterministic)
- **max_new_tokens**: Maximum tokens to generate

## Evaluation Scripts

### 1. Full Evaluation Script

**File**: `scripts/evaluate_extraction_methods.py`

Comprehensive evaluation that compares multiple methods on the full test dataset.

**Features**:
- Tests multiple methods in parallel
- Calculates precision, recall, F1 score per entity type
- Measures throughput and processing time
- Generates detailed comparison report
- Saves results as JSON and text report

**Usage**:

```bash
# Basic usage
python scripts/evaluate_extraction_methods.py

# Edit the script to select which methods to test:
methods_to_test = [
    "RAW",                  # Basic RAW mode
    "RAW_WITH_SCHEMA",      # RAW with schema in prompt
    "RAW_CHAT_FORMAT",      # RAW with chat format
    "STRUCTURED_OUTPUT",    # vLLM (requires GPU)
    "OUTLINES",             # Outlines library
]

# Set sample limit for quick testing
sample_limit = 10  # Test on 10 samples only
```

**Output**:
- `results/method_comparison/evaluation_results.json` - Detailed results
- `results/method_comparison/evaluation_report.txt` - Human-readable report

### 2. Quick Evaluation Script

**File**: `scripts/quick_evaluate.py`

Lightweight script for quick testing on a few samples.

**Features**:
- Fast testing on 5-10 samples
- Shows detailed output for each sample
- Good for debugging and quick experiments
- Simple exact-match accuracy metric

**Usage**:

```bash
# Test RAW mode
python scripts/quick_evaluate.py

# Edit the script to test different configurations:
quick_evaluate(
    extraction_mode=ExtractionMode.RAW,
    use_chat_format=False,
    add_schema=False,
    num_samples=5,
)

# Test with schema
quick_evaluate(
    extraction_mode=ExtractionMode.RAW,
    use_chat_format=False,
    add_schema=True,
    num_samples=5,
)

# Test vLLM
quick_evaluate(
    extraction_mode=ExtractionMode.STRUCTURED_OUTPUT,
    use_chat_format=True,
    add_schema=True,
    num_samples=5,
)
```

## Evaluation Metrics

### Precision
- Percentage of predicted entities that are correct
- `TP / (TP + FP)`

### Recall
- Percentage of true entities that were found
- `TP / (TP + FN)`

### F1 Score
- Harmonic mean of precision and recall
- `2 * (Precision * Recall) / (Precision + Recall)`

### Throughput
- Samples processed per second
- Important for production deployment

## Example Workflow

### Step 1: Prepare Test Dataset

```bash
# Generate test dataset with ground truth labels
python scripts/generate_dataset.py
```

This creates `data/processed/ner_test_dataset.json`

### Step 2: Quick Test

```bash
# Quick test to verify everything works
python scripts/quick_evaluate.py
```

### Step 3: Full Evaluation

```bash
# Run full evaluation
python scripts/evaluate_extraction_methods.py
```

### Step 4: Analyze Results

Check the generated reports in `results/method_comparison/`:

```bash
# View text report
cat results/method_comparison/evaluation_report.txt

# View detailed JSON results
cat results/method_comparison/evaluation_results.json
```

## Expected Results

Typical performance characteristics:

| Method | Speed | F1 Score | Valid JSON | GPU Required |
|--------|-------|----------|------------|--------------|
| RAW | 1x (baseline) | ~0.75-0.85 | ~90-95% | No |
| RAW + Schema | 1x | ~0.78-0.87 | ~95-98% | No |
| STRUCTURED_OUTPUT | 10-50x | ~0.80-0.90 | 100% | Yes |
| OUTLINES | 2-5x | ~0.80-0.90 | 100% | No |

*Note: Actual results depend on model quality, dataset, and hardware*

## Tips for Best Results

### 1. Optimize for Accuracy
- Use `add_schema=True` to include JSON schema in prompts
- Lower temperature (0.1) for more consistent outputs
- Use chat format for chat-tuned models

### 2. Optimize for Speed
- Use STRUCTURED_OUTPUT mode with vLLM (requires GPU)
- Increase batch size
- Use smaller models (Qwen3-4B vs Qwen3-72B)

### 3. Optimize for Resources
- Use RAW or OUTLINES mode if no GPU available
- Lower `max_new_tokens` if responses are short
- Use quantization (4-bit) for lower memory usage

## Troubleshooting

### ImportError: vLLM not installed

```bash
pip install vllm
```

### ImportError: Outlines not installed

```bash
pip install outlines
```

### CUDA Out of Memory

- Reduce `vllm_gpu_memory_utilization` (default: 0.9)
- Use smaller model
- Reduce batch size
- Enable 4-bit quantization

### Slow Performance

- Use vLLM for production workloads
- Enable batch processing
- Use GPU instead of CPU

## Advanced Usage

### Custom Evaluation Metrics

Modify `calculate_metrics()` in `evaluate_extraction_methods.py`:

```python
def calculate_metrics(self, predictions, ground_truth):
    # Add your custom metrics here
    # e.g., partial matching, entity linking, etc.
    pass
```

### Testing on Custom Dataset

```python
evaluator = NERMethodEvaluator(
    test_dataset_path=Path("path/to/your/dataset.json")
)
```

### A/B Testing Different Prompts

Create multiple configs with different `add_schema` values:

```python
configs = {
    "no_schema": NERPromptEngineeringConfig(add_schema=False),
    "with_schema": NERPromptEngineeringConfig(add_schema=True),
}
```

## Integration with Training

After evaluating different methods:

1. **Choose best method** based on your requirements (accuracy vs speed)
2. **Use same configuration** for training data generation:

```python
# Export training data with same settings
DataProcessorService.export_for_finetuning(
    samples=train_samples,
    output_path=output_path,
    format="chat",
    add_schema=True,  # Match your best config
)
```

3. **Fine-tune model** on this data
4. **Re-evaluate** fine-tuned model using the same scripts

This ensures consistency between training and inference!
