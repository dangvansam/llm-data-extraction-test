# Configuration Guide

This guide explains the configuration system for the NER extraction project.

## Overview

The project now uses **method-specific configuration classes** for better modularity and easier customization:

- `BaseNERConfig` - Base class with common settings
- `NERPromptEngineeringConfig` - Prompt engineering method
- `NERRagConfig` - RAG method
- `NERFineTuningConfig` - Fine-tuning method
- `NERLangExtractConfig` - LangExtract method
- `NERConfig` - Backward-compatible combined config

## Configuration Classes

### 1. BaseNERConfig

Base class inherited by all method-specific configs.

```python
from src.config import BaseNERConfig

config = BaseNERConfig(
    entity_types=["person", "organizations", "address"],
    temperature=0.1,
    top_p=0.9,
    max_length=2048
)
```

**Common Parameters:**
- `entity_types` - List of entity types to extract
- `temperature` - Model temperature (0-1)
- `top_p` - Top-p sampling parameter
- `max_length` - Maximum input sequence length
- `data_dir` - Data directory path
- `results_dir` - Results directory path

---

### 2. NERPromptEngineeringConfig

Configuration for prompt engineering method.

```python
from src.config import NERPromptEngineeringConfig

config = NERPromptEngineeringConfig(
    model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
    use_openai=False,
    openai_model="gpt-4o-mini",
    max_new_tokens=512,
    do_sample=True,
    batch_size=1
)
```

**Key Parameters:**
- `model_name` - HuggingFace model to use
- `use_openai` - Whether to use OpenAI API
- `openai_model` - OpenAI model name
- `max_new_tokens` - Maximum tokens to generate
- `do_sample` - Enable sampling
- `batch_size` - Batch size for processing

**Usage Example:**

```python
from src.prompt_engineering import PromptNERExtractor
from src.config import NERPromptEngineeringConfig

# Default configuration
config = NERPromptEngineeringConfig()
extractor = PromptNERExtractor(config=config)

# Custom configuration
custom_config = NERPromptEngineeringConfig(
    model_name="meta-llama/Llama-3.3-70B-Instruct",
    temperature=0.2,
    use_openai=True  # Use OpenAI instead
)
extractor = PromptNERExtractor(config=custom_config, use_openai=True)
```

---

### 3. NERRagConfig

Configuration for RAG method.

```python
from src.config import NERRagConfig

config = NERRagConfig(
    model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
    embedding_model="BAAI/bge-small-en-v1.5",
    normalize_embeddings=True,
    chunk_size=512,
    chunk_overlap=50,
    top_k_retrieval=3,
    similarity_threshold=0.7,
    index_type="flat",
    use_gpu_index=False
)
```

**Key Parameters:**

**LLM Settings:**
- `model_name` - Language model for generation

**Embedding Settings:**
- `embedding_model` - Sentence transformer model
- `normalize_embeddings` - Normalize embedding vectors

**Chunking Settings:**
- `chunk_size` - Size of text chunks
- `chunk_overlap` - Overlap between chunks

**Retrieval Settings:**
- `top_k_retrieval` - Number of documents to retrieve
- `similarity_threshold` - Minimum similarity score

**Vector Store:**
- `index_type` - FAISS index type ("flat" or "ivf")
- `use_gpu_index` - Use GPU for indexing
- `index_path` - Path to save/load index

**Usage Example:**

```python
from src.rag_pipeline import RAGNERExtractor
from src.config import NERRagConfig

# High accuracy configuration (more retrieval)
high_acc_config = NERRagConfig(
    top_k_retrieval=5,
    similarity_threshold=0.5,
    chunk_size=256,  # Smaller chunks
    chunk_overlap=100
)

# Fast configuration (less retrieval)
fast_config = NERRagConfig(
    top_k_retrieval=2,
    chunk_size=1024,  # Larger chunks
    chunk_overlap=25
)

extractor = RAGNERExtractor(config=high_acc_config, corpus=train_data)
```

---

### 4. NERFineTuningConfig

Configuration for fine-tuning method.

```python
from src.config import NERFineTuningConfig

config = NERFineTuningConfig(
    model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",

    # Training
    learning_rate=1e-4,
    num_epochs=3,
    batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    weight_decay=0.01,

    # LoRA
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,

    # Quantization
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_quant_type="nf4",

    # Optimization
    fp16=True,
    gradient_checkpointing=False,
    max_grad_norm=1.0,

    # Logging
    logging_steps=10,
    save_steps=100,
    save_total_limit=3
)
```

**Key Parameters:**

**Training:**
- `learning_rate` - Learning rate
- `num_epochs` - Number of training epochs
- `batch_size` - Training batch size
- `gradient_accumulation_steps` - Gradient accumulation
- `warmup_steps` - Learning rate warmup
- `weight_decay` - Weight decay

**LoRA:**
- `lora_r` - LoRA rank
- `lora_alpha` - LoRA alpha
- `lora_dropout` - LoRA dropout
- `lora_target_modules` - Modules to apply LoRA

**Quantization:**
- `load_in_4bit` - Use 4-bit quantization
- `bnb_4bit_compute_dtype` - Compute dtype
- `bnb_4bit_quant_type` - Quantization type
- `bnb_4bit_use_double_quant` - Double quantization

**Usage Example:**

```python
from src.finetuning import FineTunedNERExtractor
from src.config import NERFineTuningConfig

# Low resource configuration (for 16GB GPU)
low_mem_config = NERFineTuningConfig(
    batch_size=2,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    lora_r=8  # Smaller rank
)

# High performance configuration (for 40GB+ GPU)
high_perf_config = NERFineTuningConfig(
    batch_size=8,
    gradient_accumulation_steps=1,
    lora_r=32,
    lora_alpha=64
)

extractor = FineTunedNERExtractor(config=low_mem_config)
extractor.train(train_dataset, val_dataset)
```

---

### 5. NERLangExtractConfig

Configuration for LangExtract method.

```python
from src.config import NERLangExtractConfig

config = NERLangExtractConfig(
    model_id="gemini-2.0-flash-exp",

    # Extraction
    extraction_passes=2,
    max_workers=10,
    max_char_buffer=2000,

    # API
    api_key_env="GOOGLE_API_KEY",
    timeout=60,
    retry_attempts=3,

    # Output
    save_jsonl=True,
    create_visualization=True,
    include_attributes=True,

    # Performance
    batch_delay=0.1,
    chunk_overlap_chars=100
)
```

**Key Parameters:**

**Model:**
- `model_id` - Gemini model ID

**Extraction:**
- `extraction_passes` - Number of independent passes
- `max_workers` - Parallel workers
- `max_char_buffer` - Chunk size for long documents

**API:**
- `api_key_env` - Environment variable for API key
- `timeout` - API timeout (seconds)
- `retry_attempts` - Number of retries

**Output:**
- `save_jsonl` - Save results as JSONL
- `create_visualization` - Generate HTML viz
- `include_attributes` - Extract entity attributes

**Usage Example:**

```python
from src.langextract_pipeline import LangExtractNERExtractor
from src.config import NERLangExtractConfig

# High accuracy configuration (more passes)
high_acc_config = NERLangExtractConfig(
    extraction_passes=3,
    max_workers=20,
    max_char_buffer=1000  # Smaller chunks
)

# Fast configuration (fewer passes)
fast_config = NERLangExtractConfig(
    extraction_passes=1,
    max_workers=10,
    max_char_buffer=3000
)

# Production configuration
prod_config = NERLangExtractConfig(
    model_id="gemini-2.0-pro-exp",  # More accurate
    extraction_passes=2,
    retry_attempts=5,
    timeout=120
)

extractor = LangExtractNERExtractor(config=high_acc_config)
```

---

## Common Configuration Patterns

### Pattern 1: Quick Prototyping

```python
from src.config import DEFAULT_PROMPT_CONFIG
from src.prompt_engineering import PromptNERExtractor

# Use defaults for quick start
extractor = PromptNERExtractor(config=DEFAULT_PROMPT_CONFIG)
```

### Pattern 2: Custom Entity Types

```python
from src.config import NERPromptEngineeringConfig

# Extract different entities
config = NERPromptEngineeringConfig(
    entity_types=["product", "company", "technology"]
)
```

### Pattern 3: Resource-Constrained Environment

```python
from src.config import NERFineTuningConfig

config = NERFineTuningConfig(
    batch_size=1,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    lora_r=8
)
```

### Pattern 4: Production Deployment

```python
from src.config import NERRagConfig

config = NERRagConfig(
    model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
    top_k_retrieval=5,
    similarity_threshold=0.7,
    use_gpu_index=True,
    index_path=Path("models/production_index.faiss")
)
```

### Pattern 5: Multi-Language Support

```python
from src.config import NERRagConfig

# Vietnamese
vi_config = NERRagConfig(
    embedding_model="BAAI/bge-m3",  # Multilingual
    model_name="meta-llama/Meta-Llama-3.1-8B-Instruct"
)

# Chinese
zh_config = NERRagConfig(
    embedding_model="BAAI/bge-large-zh-v1.5"
)
```

---

## Environment Variables

### Required for LangExtract

```bash
# .env file
GOOGLE_API_KEY=your-google-api-key-here
```

### Optional for OpenAI

```bash
# .env file
OPENAI_API_KEY=your-openai-api-key-here
```

### Loading Environment Variables

```python
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
```

---

## Default Configurations

Pre-configured defaults for each method:

```python
from src.config import (
    DEFAULT_PROMPT_CONFIG,      # Prompt engineering
    DEFAULT_RAG_CONFIG,          # RAG
    DEFAULT_FINETUNING_CONFIG,   # Fine-tuning
    DEFAULT_LANGEXTRACT_CONFIG,  # LangExtract
    DEFAULT_CONFIG               # Backward compatible
)
```

---

## Backward Compatibility

Existing code using `NERConfig` continues to work:

```python
from src.config import NERConfig

# Old code still works
config = NERConfig(
    model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
    learning_rate=1e-4,
    batch_size=4
)
```

---

## Tips and Best Practices

### 1. Start with Defaults

```python
# Use defaults first
from src.config import DEFAULT_PROMPT_CONFIG

# Then customize
custom_config = NERPromptEngineeringConfig(
    **vars(DEFAULT_PROMPT_CONFIG),  # Copy defaults
    temperature=0.2  # Override specific values
)
```

### 2. Save Configurations

```python
import json
from dataclasses import asdict

config = NERFineTuningConfig(learning_rate=2e-4)

# Save to file
with open("my_config.json", "w") as f:
    json.dump(asdict(config), f, indent=2, default=str)
```

### 3. Load from File

```python
import json

# Load from file
with open("my_config.json", "r") as f:
    config_dict = json.load(f)

config = NERFineTuningConfig(**config_dict)
```

### 4. Experiment Tracking

```python
# Save config with results
results = {
    "config": asdict(config),
    "accuracy": 0.85,
    "f1_score": 0.82
}

with open("experiment_001.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
```

---

## Summary

The new configuration system provides:

✅ **Method-specific configs** for clarity
✅ **Type hints** for better IDE support
✅ **Default values** for quick start
✅ **Backward compatibility** with existing code
✅ **Flexible customization** for all parameters
✅ **Clear documentation** for each setting

Choose the config class that matches your method and customize as needed!
