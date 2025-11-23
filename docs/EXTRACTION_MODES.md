# NER Extraction Modes

This guide explains the three extraction modes available for Named Entity Recognition (NER) and when to use each one.

## Overview

The NER extraction system supports three modes:

1. **RAW** - Standard generation without structured output
2. **STRUCTURED_OUTPUT** - vLLM with JSON schema validation
3. **OUTLINES** - Outlines library with Pydantic models for type-safe extraction

## Mode Comparison

| Feature | RAW | STRUCTURED_OUTPUT | OUTLINES |
|---------|-----|-------------------|----------|
| Speed | Standard | ⚡ Very Fast (10-50x) | ⚡ Fast |
| Structured Output | ❌ No | ✅ Yes (JSON Schema) | ✅ Yes (Pydantic) |
| Valid JSON Guaranteed | ❌ No | ✅ Yes | ✅ Yes |
| GPU Required | ❌ No | ✅ Yes | ❌ No |
| CPU Support | ✅ Yes | ❌ No | ✅ Yes |
| Parsing Required | ✅ Yes (regex) | ❌ No | ❌ No |
| Dependencies | transformers | vLLM | outlines |
| Batch Processing | Basic | ⚡ Optimized | Standard |
| Best For | Simple cases, CPU | Production, batch | Type safety, complex schemas |

## 1. RAW Mode

Standard generation without structured output constraints.

### Features
- ✅ Works on CPU or GPU
- ✅ No additional dependencies
- ❌ Requires JSON parsing from text
- ❌ May produce invalid JSON
- ❌ Slower inference

### Installation
```bash
pip install transformers torch
```

### Usage

```python
from src.config import ExtractionMode, NERPromptEngineeringConfig
from src.prompt_engineering import PromptNERExtractor

config = NERPromptEngineeringConfig(
    model_name="Qwen/Qwen3-4B",
    extraction_mode=ExtractionMode.RAW,
    temperature=0.1,
    max_new_tokens=2048,
)

extractor = PromptNERExtractor(config)
entities = extractor.extract_entities(text)
```

### When to Use
- Testing and development
- CPU-only environments
- Small-scale extraction
- When dependencies must be minimal

## 2. STRUCTURED_OUTPUT Mode (vLLM)

High-performance inference with guaranteed valid JSON using vLLM's structured output.

### Features
- ✅ 10-50x faster than RAW mode
- ✅ Guaranteed valid JSON via schema
- ✅ Optimized batch processing
- ✅ Lower GPU memory usage
- ❌ GPU required
- ❌ Linux only

### Installation
```bash
pip install vllm
```

### Usage

```python
from src.config import ExtractionMode, NERPromptEngineeringConfig
from src.prompt_engineering import PromptNERExtractor

config = NERPromptEngineeringConfig(
    model_name="Qwen/Qwen3-4B",
    extraction_mode=ExtractionMode.STRUCTURED_OUTPUT,
    temperature=0.1,
    max_new_tokens=2048,
    vllm_tensor_parallel_size=1,
    vllm_gpu_memory_utilization=0.9,
)

extractor = PromptNERExtractor(config)
entities = extractor.extract_entities(text)
```

### Batch Processing

```python
# Efficiently process many texts
texts = [text1, text2, text3, ...]
results = extractor.batch_extract(texts)
```

### When to Use
- Production deployments
- Large-scale batch processing
- When speed is critical
- GPU available (Linux)

## 3. OUTLINES Mode

Type-safe structured generation using Outlines with Pydantic models.

### Features
- ✅ Type-safe with Pydantic models
- ✅ Guaranteed valid output
- ✅ Works on CPU or GPU
- ✅ Cross-platform (Windows, Mac, Linux)
- ✅ Complex schema support
- ❌ Slower than vLLM
- ❌ No optimized batch processing

### Installation
```bash
pip install outlines
```

### Usage

```python
from src.config import ExtractionMode, NERPromptEngineeringConfig
from src.prompt_engineering import PromptNERExtractor

config = NERPromptEngineeringConfig(
    model_name="Qwen/Qwen3-4B",
    extraction_mode=ExtractionMode.OUTLINES,
    temperature=0.1,
    max_new_tokens=2048,
)

extractor = PromptNERExtractor(config)
entities = extractor.extract_entities(text)
```

### Pydantic Model

The extraction uses this Pydantic model:

```python
from pydantic import BaseModel

class NEREntities(BaseModel):
    person: List[str]
    organizations: List[str]
    address: List[str]
```

### When to Use
- Type safety is important
- CPU-only environments
- Windows/Mac development
- Complex schema validation needed
- When vLLM is not available

## Configuration

### Using ExtractionMode Enum

```python
from src.config import ExtractionMode, NERPromptEngineeringConfig

# RAW mode
config = NERPromptEngineeringConfig(
    extraction_mode=ExtractionMode.RAW
)

# STRUCTURED_OUTPUT mode
config = NERPromptEngineeringConfig(
    extraction_mode=ExtractionMode.STRUCTURED_OUTPUT
)

# OUTLINES mode
config = NERPromptEngineeringConfig(
    extraction_mode=ExtractionMode.OUTLINES
)
```

### Backward Compatibility

The old `use_vllm` parameter is still supported but deprecated:

```python
# Old way (deprecated)
config = NERPromptEngineeringConfig(use_vllm=True)

# New way (recommended)
config = NERPromptEngineeringConfig(
    extraction_mode=ExtractionMode.STRUCTURED_OUTPUT
)
```

## Performance Comparison

Typical performance on a single GPU (RTX 4090):

| Mode | Single Text | Batch (10 texts) | Batch (100 texts) |
|------|-------------|------------------|-------------------|
| RAW | ~5-10s | ~50-100s | ~500-1000s |
| STRUCTURED_OUTPUT | ~0.5-1s | ~2-5s | ~10-20s |
| OUTLINES | ~2-4s | ~20-40s | ~200-400s |

*Performance varies based on hardware, model size, and text length*

## Example Outputs

All modes produce the same output format:

```python
{
    "person": ["Nguyễn Văn A", "Trần Thị B"],
    "organizations": ["Công ty FPT", "Vingroup"],
    "address": ["Hà Nội", "TP. Hồ Chí Minh"]
}
```

The difference is:
- **RAW**: May have parsing errors or invalid JSON
- **STRUCTURED_OUTPUT**: Always valid, fastest
- **OUTLINES**: Always valid, type-safe

## Error Handling

### RAW Mode
May return empty lists if JSON parsing fails:
```python
# Parsing failed
{"person": [], "organizations": [], "address": []}
```

### STRUCTURED_OUTPUT Mode
Always returns valid structure:
```python
# Even if extraction finds nothing
{"person": [], "organizations": [], "address": []}
```

### OUTLINES Mode
Validates against Pydantic model:
```python
# Type-validated output
{"person": ["..."], "organizations": ["..."], "address": ["..."]}
```

## Best Practices

### 1. Choose the Right Mode

- **Development**: Use `RAW` or `OUTLINES` (works on CPU)
- **Production**: Use `STRUCTURED_OUTPUT` (fastest, requires GPU)
- **Type Safety**: Use `OUTLINES` (Pydantic validation)
- **Windows/Mac**: Use `OUTLINES` (vLLM not supported)

### 2. Batch Processing

Always use batch processing for multiple texts:

```python
# Good: Batch processing
results = extractor.batch_extract(texts)

# Bad: Sequential processing
results = [extractor.extract_entities(t) for t in texts]
```

### 3. GPU Memory Management

For STRUCTURED_OUTPUT mode, adjust memory usage:

```python
config = NERPromptEngineeringConfig(
    extraction_mode=ExtractionMode.STRUCTURED_OUTPUT,
    vllm_gpu_memory_utilization=0.7,  # Reduce if OOM
)
```

### 4. Temperature Settings

Use low temperature for consistent extraction:

```python
config = NERPromptEngineeringConfig(
    temperature=0.1,  # Lower = more consistent
    top_p=0.9,
)
```

## Troubleshooting

### vLLM Import Error
```
ImportError: vLLM is not installed
```
**Solution**: Install vLLM: `pip install vllm` (Linux + GPU only)

### Outlines Import Error
```
ImportError: Outlines is not installed
```
**Solution**: Install outlines: `pip install outlines`

### Invalid JSON in RAW Mode
```
Warning: No JSON found in response
```
**Solution**: Switch to `STRUCTURED_OUTPUT` or `OUTLINES` mode

### Out of Memory (STRUCTURED_OUTPUT)
```
CUDA out of memory
```
**Solution**: Reduce `vllm_gpu_memory_utilization` or use smaller batch sizes

## Complete Example

See [scripts/example_extraction_modes.py](../scripts/example_extraction_modes.py) for complete examples demonstrating all three modes.

## Migration Guide

### From RAW to STRUCTURED_OUTPUT

```python
# Before
config = NERPromptEngineeringConfig()
# or
config = NERPromptEngineeringConfig(use_vllm=False)

# After
config = NERPromptEngineeringConfig(
    extraction_mode=ExtractionMode.STRUCTURED_OUTPUT
)
```

### From use_vllm to extraction_mode

```python
# Before
config = NERPromptEngineeringConfig(use_vllm=True)

# After
config = NERPromptEngineeringConfig(
    extraction_mode=ExtractionMode.STRUCTURED_OUTPUT
)
```

## References

- [vLLM Documentation](https://docs.vllm.ai/)
- [Outlines Documentation](https://outlines-dev.github.io/outlines/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
