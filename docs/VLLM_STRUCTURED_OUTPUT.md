# vLLM Structured Output for NER Extraction

This guide explains how to use vLLM with structured outputs for faster and more reliable Named Entity Recognition (NER) extraction.

## Overview

vLLM is an optimized inference engine that provides:
- **Faster inference** - Up to 10-30x speedup over standard transformers
- **Structured outputs** - Guarantees valid JSON responses that match a schema
- **Batch processing** - Efficient processing of multiple texts simultaneously
- **Lower memory usage** - Better GPU memory management with PagedAttention

## Installation

Install vLLM:

```bash
pip install vllm
```

Note: vLLM requires:
- CUDA-capable GPU
- Linux operating system
- Python 3.8+

## Configuration

### Basic Configuration

```python
from src.config import NERPromptEngineeringConfig

config = NERPromptEngineeringConfig(
    model_name="Qwen/Qwen3-4B",
    use_vllm=True,                      # Enable vLLM
    vllm_structured_output=True,        # Enable structured JSON output
    vllm_tensor_parallel_size=1,        # Number of GPUs for tensor parallelism
    vllm_gpu_memory_utilization=0.9,    # GPU memory to use (0.0-1.0)
    temperature=0.1,
    max_new_tokens=2048,
)
```

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_vllm` | bool | False | Enable vLLM inference engine |
| `vllm_structured_output` | bool | True | Use JSON schema for structured output |
| `vllm_tensor_parallel_size` | int | 1 | Number of GPUs for tensor parallelism |
| `vllm_gpu_memory_utilization` | float | 0.9 | GPU memory utilization (0.0-1.0) |

## Usage Examples

### 1. Basic Single Text Extraction

```python
from src.config import NERPromptEngineeringConfig
from src.prompt_engineering import PromptNERExtractor

# Configure with vLLM
config = NERPromptEngineeringConfig(
    model_name="Qwen/Qwen3-4B",
    use_vllm=True,
    vllm_structured_output=True,
)

# Initialize extractor
extractor = PromptNERExtractor(config)

# Extract entities
text = "Công ty FPT tại Hà Nội do CEO Nguyễn Văn A lãnh đạo."
entities = extractor.extract_entities(text)

print(entities)
# Output: {
#   "person": ["Nguyễn Văn A"],
#   "organizations": ["Công ty FPT"],
#   "address": ["Hà Nội"]
# }
```

### 2. Batch Processing

vLLM excels at batch processing multiple texts:

```python
texts = [
    "Thủ tướng Phạm Minh Chính gặp gỡ doanh nghiệp tại Đà Nẵng.",
    "Vingroup đầu tư 5000 tỷ vào Bình Dương.",
    "Bác sĩ Trần Thị B làm việc tại Bệnh viện Bạch Mai.",
]

# Batch extract - much faster than sequential processing
results = extractor.batch_extract(texts, show_progress=True)

for text, entities in zip(texts, results):
    print(f"Text: {text}")
    print(f"Entities: {entities}\n")
```

### 3. Without Structured Output

You can disable structured output for free-form generation:

```python
config = NERPromptEngineeringConfig(
    model_name="Qwen/Qwen3-4B",
    use_vllm=True,
    vllm_structured_output=False,  # Disable structured output
)

extractor = PromptNERExtractor(config)
entities = extractor.extract_entities(text)
```

### 4. Multi-GPU Configuration

For larger models or faster processing, use tensor parallelism:

```python
config = NERPromptEngineeringConfig(
    model_name="Qwen/Qwen3-4B",
    use_vllm=True,
    vllm_tensor_parallel_size=2,  # Use 2 GPUs
    vllm_gpu_memory_utilization=0.85,
)

extractor = PromptNERExtractor(config)
```

## Structured Output Details

### JSON Schema

The structured output uses the following JSON schema:

```json
{
  "type": "object",
  "properties": {
    "person": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Names of people"
    },
    "organizations": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Names of companies, institutions, organizations"
    },
    "address": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Locations, places, addresses, geographical names"
    }
  },
  "required": ["person", "organizations", "address"]
}
```

### Benefits of Structured Output

1. **Guaranteed Valid JSON** - No parsing errors or malformed responses
2. **Type Safety** - Always returns arrays for each entity type
3. **Consistency** - Same format across all responses
4. **Faster Parsing** - No need for regex or complex parsing logic
5. **Better Accuracy** - Model is guided to produce correct format

## Performance Comparison

Typical performance improvements with vLLM:

| Configuration | Time per Text | Batch (10 texts) | Speedup |
|--------------|---------------|------------------|---------|
| Transformers | ~5-10s | ~50-100s | 1x |
| vLLM | ~0.5-1s | ~5-10s | 10-20x |
| vLLM + Batch | ~0.5-1s | ~2-5s | 20-50x |

*Note: Actual performance depends on hardware, model size, and text length*

## Advanced Usage

### Custom Sampling Parameters

```python
from vllm import SamplingParams
from vllm.sampling_params import StructuredOutputsParams

# Direct access to vLLM model for custom parameters
if extractor.config.use_vllm:
    json_schema = extractor._get_json_schema()
    structured_params = StructuredOutputsParams(json=json_schema)

    sampling_params = SamplingParams(
        max_tokens=2048,
        temperature=0.1,
        top_p=0.9,
        top_k=50,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        structured_outputs=structured_params,
    )

    # Use custom parameters
    outputs = extractor.vllm_model.generate(
        prompts=[prompt],
        sampling_params=sampling_params,
    )
```

### Memory Optimization

For limited GPU memory:

```python
config = NERPromptEngineeringConfig(
    model_name="Qwen/Qwen3-4B",
    use_vllm=True,
    vllm_gpu_memory_utilization=0.7,  # Reduce memory usage
    max_new_tokens=1024,  # Reduce max tokens
)
```

## Troubleshooting

### Out of Memory (OOM) Errors

1. Reduce `vllm_gpu_memory_utilization`: Try 0.7 or 0.6
2. Reduce `max_new_tokens`: Try 1024 or 512
3. Use smaller batch sizes
4. Use a smaller model

### Slow First Request

vLLM pre-allocates GPU memory on first request. This is normal and subsequent requests will be much faster.

### Import Errors

If you get `ImportError: vLLM is not installed`:

```bash
pip install vllm
```

Make sure you have:
- CUDA-capable GPU
- Linux OS (vLLM doesn't support Windows)
- Compatible CUDA version

## Best Practices

1. **Use Structured Output** - Always enable for JSON responses
2. **Batch When Possible** - Process multiple texts together for best performance
3. **Tune Memory Usage** - Adjust `gpu_memory_utilization` based on your GPU
4. **Low Temperature** - Use 0.1-0.3 for consistent NER extraction
5. **Monitor GPU Usage** - Use `nvidia-smi` to check memory usage

## Comparison: Transformers vs vLLM

| Feature | Transformers | vLLM |
|---------|-------------|------|
| Speed | ✓ Standard | ✓✓✓ Very Fast |
| Memory Efficiency | ✓ Standard | ✓✓✓ Excellent |
| Structured Output | ✗ No | ✓✓✓ Yes |
| Batch Processing | ✓ Basic | ✓✓✓ Optimized |
| GPU Support | ✓ Yes | ✓ Yes (Required) |
| CPU Support | ✓ Yes | ✗ No |
| Multi-GPU | ✓ Basic | ✓✓✓ Advanced |

## Example Script

See [scripts/example_vllm_ner.py](../scripts/example_vllm_ner.py) for complete examples including:
- Basic single text extraction
- Batch processing
- Performance comparison
- Advanced configurations

## References

- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM Structured Outputs Guide](https://docs.vllm.ai/en/latest/serving/structured_outputs.html)
- [PagedAttention Paper](https://arxiv.org/abs/2309.06180)
