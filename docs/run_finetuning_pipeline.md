# 🎯 Fine-Tuning Pipeline

> **Part 3:** Fine-tune LLM for NER extraction using LoRA or full fine-tuning

---

## 📋 Overview

This pipeline fine-tunes a large language model (LLM) specifically for Vietnamese Named Entity Recognition (NER) tasks. It supports both parameter-efficient fine-tuning with LoRA and full model fine-tuning.

---

## 🚀 Quick Start

### Run Fine-Tuning

```bash
python scripts/run_finetuning_pipeline.py
```

---

## ⚙️ Configuration

### Basic Configuration (LoRA)

```python
from src.config import NERFineTuningConfig, CHECKPOINTS_DIR, PROCESSED_DATA_DIR

config = NERFineTuningConfig(
    # Model settings
    model_name="Qwen/Qwen3-4B",
    max_seq_length=2048,
    load_in_4bit=True,           # 4-bit quantization for memory efficiency
    load_in_8bit=False,
    full_finetuning=False,       # Use LoRA

    # Prompt settings
    add_schema=False,

    # Training settings
    max_steps=100000,
    num_epochs=3,
    batch_size=4,
    learning_rate=1e-4,
    warmup_steps=100,
    gradient_accumulation_steps=4,
    lr_scheduler_type="linear",
    weight_decay=0.01,
    optim="adamw_torch_fused",

    # LoRA settings (only used if full_finetuning=False)
    lora_r=16,                   # LoRA rank
    lora_alpha=32,               # LoRA alpha
    lora_dropout=0.05,           # LoRA dropout

    # Logging and checkpointing
    logging_steps=10,
    save_steps=100,
    save_total_limit=3,
    eval_steps=100,

    # Output directory
    output_dir=CHECKPOINTS_DIR / "Qwen3-4B_finetuned_2048_lora_r16_a32_drop0.05_lr1e-4_warmup100_decay0.01_acc4_bs4",

    # Data paths
    train_data_path=PROCESSED_DATA_DIR / "train_finetuning_chat.jsonl",
    val_data_path=PROCESSED_DATA_DIR / "test_finetuning_chat.jsonl",

    # Training options
    resume_from_checkpoint=None,    # Set to checkpoint path to resume training
    report_to="wandb",               # "none", "wandb", or "tensorboard"
)
```

### Configuration Templates

<details>
<summary><b>💾 Memory-Efficient LoRA (4-bit, Rank 16)</b></summary>

```python
config = NERFineTuningConfig(
    model_name="Qwen/Qwen3-4B",
    load_in_4bit=True,
    full_finetuning=False,
    lora_r=16,
    batch_size=4,
    learning_rate=1e-4,
)
# Memory: ~4-6GB VRAM
```
</details>

<details>
<summary><b>⚡ High Performance LoRA (4-bit, Rank 64)</b></summary>

```python
config = NERFineTuningConfig(
    model_name="Qwen/Qwen3-4B",
    load_in_4bit=True,
    full_finetuning=False,
    lora_r=64,
    lora_alpha=128,
    batch_size=4,
    learning_rate=2e-4,
)
# Memory: ~5-7GB VRAM
```
</details>

<details>
<summary><b>🔥 Full Fine-Tuning (Best Quality)</b></summary>

```python
config = NERFineTuningConfig(
    model_name="Qwen/Qwen3-4B",
    load_in_8bit=True,
    full_finetuning=True,
    batch_size=2,
    learning_rate=5e-5,
)
# Memory: ~16GB+ VRAM
```
</details>

---

## 📂 Input Requirements

### Required Files

Before starting fine-tuning, ensure these files exist:

| File | Path | Description |
|------|------|-------------|
| Training Data | `data/processed/train_finetuning_chat.jsonl` | Chat format training data |
| Validation Data | `data/processed/test_finetuning_chat.jsonl` | Chat format validation data |

### Data Format

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

> 💡 **Tip**: Run [data preparation](run_data_preparation.md) first to generate these files.

---

## 📝 Logs and Monitoring

### Training Logs

The pipeline writes detailed logs to:

📄 **[logs/run_finetuning_pipeline.log](logs/run_finetuning_pipeline.log)**

This log file contains:
- Model initialization details
- Training progress (loss, learning rate, metrics)
- Checkpoint saving events
- GPU memory usage
- Validation results
- Error messages (if any)

### Monitor Logs in Real-time

```bash
# Tail the log file
tail -f logs/run_finetuning_pipeline.log
```

### WandB Dashboard

If you set `report_to="wandb"`, view your training in real-time:

```
🚀 View run at https://wandb.ai/your-username/project/runs/run-id
```

---

## 📊 Training Process

### Phase 1: Initialization
```
2025-11-23 23:56:17 | INFO | FINE-TUNING
2025-11-23 23:56:17 | INFO | Model: Qwen/Qwen3-4B
2025-11-23 23:56:17 | INFO | Quantization: 4-bit
2025-11-23 23:56:17 | INFO | Training mode: LoRA
Loading checkpoint shards: 100%|██████████| 3/3 [00:23<00:00]
trainable params: 33,030,144 || all params: 4,055,498,240 || trainable%: 0.8145
```

### Phase 2: Data Loading
```
2025-11-23 23:56:47 | INFO | Loading training dataset...
2025-11-23 23:56:47 | INFO | Loaded 1000 samples
Map: 100%|████████████████| 1000/1000 [00:00<00:00]
```

### Phase 3: Training
```
{'loss': 2.6852, 'learning_rate': 9e-06, 'epoch': 0.16}
{'loss': 2.5272, 'learning_rate': 1.9e-05, 'epoch': 0.32}
{'loss': 2.31, 'learning_rate': 2.9e-05, 'epoch': 0.48}
...
{'eval_loss': 1.6788, 'eval_samples_per_second': 5.286, 'epoch': 1.59}
```

### Phase 4: Checkpointing
```
Saving checkpoint at step 100...
Saving checkpoint at step 200...
Model saved to: results/checkpoints/Qwen3-4B_finetuned/final
```

---

## 📈 Expected Output

### Console Output

```
✅ TRAINING COMPLETE!
========================================
Model saved to: results/checkpoints/Qwen3-4B_finetuned/final
Training time: 6345.23 seconds (105.75 minutes)
Training samples: 1000

Final training loss: 1.4258
Final validation loss: 1.6788
```

### Output Files

After training, you'll find:

```
results/checkpoints/Qwen3-4B_finetuned/
├── final/                           # Final model
│   ├── adapter_model.bin           # LoRA weights
│   ├── adapter_config.json         # LoRA config
│   ├── tokenizer.json              # Tokenizer
│   └── README.md                   # Model card
├── checkpoint-100/                  # Intermediate checkpoint
├── checkpoint-200/
└── training_args.bin                # Training arguments
```

---

## 🎯 Performance Metrics

### Training Metrics

Monitor these metrics during training:

| Metric | Description | Good Value |
|--------|-------------|------------|
| `loss` | Training loss | Decreasing trend |
| `eval_loss` | Validation loss | < 2.0 |
| `learning_rate` | Current learning rate | Follows schedule |
| `grad_norm` | Gradient norm | < 10.0 |
| `mean_token_accuracy` | Token-level accuracy | > 0.65 |

### Memory Usage

```
GPU: NVIDIA L40S
Max memory: 44.403 GB
Reserved memory: 5.375 GB  (LoRA 4-bit)
```

---

## 🛠️ Troubleshooting

### Common Issues

<details>
<summary><b>❌ Out of Memory (OOM)</b></summary>

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Reduce batch size: `batch_size=2` or `batch_size=1`
2. Increase gradient accumulation: `gradient_accumulation_steps=8`
3. Enable 4-bit quantization: `load_in_4bit=True`
4. Reduce sequence length: `max_seq_length=1024`
5. Use smaller LoRA rank: `lora_r=8`
</details>

<details>
<summary><b>❌ Training Data Not Found</b></summary>

**Symptoms:**
```
FileNotFoundError: Training data not found
```

**Solution:**
Run data preparation first:
```bash
python scripts/run_data_preparation.py
```
</details>

<details>
<summary><b>❌ Loss Diverging (NaN or Inf)</b></summary>

**Symptoms:**
```
{'loss': nan, 'grad_norm': inf}
```

**Solutions:**
1. Reduce learning rate: `learning_rate=5e-5`
2. Increase warmup steps: `warmup_steps=500`
3. Add gradient clipping: `max_grad_norm=1.0`
4. Check data quality
</details>

<details>
<summary><b>❌ Slow Training Speed</b></summary>

**Solutions:**
1. Enable fused optimizer: `optim="adamw_torch_fused"`
2. Increase batch size if memory allows
3. Use FP16 training
4. Check GPU utilization with `nvidia-smi`
</details>

---

## 💡 Best Practices

### Memory Optimization
```python
# For 8GB GPU
config = NERFineTuningConfig(
    load_in_4bit=True,
    batch_size=2,
    gradient_accumulation_steps=8,  # Effective batch size: 16
    lora_r=16,
)
```

### Quality Optimization
```python
# For best quality (requires 16GB+ GPU)
config = NERFineTuningConfig(
    load_in_8bit=True,
    full_finetuning=True,
    batch_size=4,
    learning_rate=5e-5,
    num_epochs=5,
)
```

### Resume Training
```python
config = NERFineTuningConfig(
    resume_from_checkpoint="results/checkpoints/Qwen3-4B_finetuned/checkpoint-500",
    # ... other settings
)
```

---

## 📚 Related Documentation

- 📖 [Fine-Tuning Notebook](notebooks/finetuning_pipeline_experiments.ipynb)
- 📖 [Configuration Reference](src/config.py)
- 📖 [Fine-Tuning Module](src/finetuning.py)
- 📖 [Data Preparation Guide](run_data_preparation.md)
- 📖 [Evaluation Guide](docs/EVALUATION_GUIDE.md)

---

## ⏭️ Next Steps

After fine-tuning is complete:

1. **Evaluate the Model**
   ```bash
   python scripts/evaluate_extraction_methods.py
   ```

2. **Use the Fine-Tuned Model**
   ```python
   from src.finetuning import FineTunedNERExtractor
   from src.config import NERFineTuningConfig

   config = NERFineTuningConfig(
       trained_model_path="results/checkpoints/Qwen3-4B_finetuned/final"
   )
   extractor = FineTunedNERExtractor(config)
   entities = extractor.extract_entities("Your text here")
   ```

3. **Compare with Other Methods**
   - [Prompt Engineering Pipeline](run_prompt_engineering_pipeline.md)
   - [RAG Pipeline](run_rag_pipeline.md)
   - [Evaluation Comparison](docs/EVALUATION_GUIDE.md)

---

## ⏱️ Training Time Estimates

| Configuration | Dataset Size | GPU | Time |
|--------------|--------------|-----|------|
| LoRA r=16, 4-bit | 1000 samples | L40S 48GB | ~1.5 hours |
| LoRA r=64, 4-bit | 1000 samples | L40S 48GB | ~2 hours |
| Full FT, 8-bit | 1000 samples | L40S 48GB | ~4 hours |
| LoRA r=16, 4-bit | 10000 samples | L40S 48GB | ~15 hours |

> 💡 **Note**: Times are estimates and may vary based on batch size, sequence length, and hardware.

---

## 🔍 Tips & Tricks

### Effective Batch Size
```python
# These are equivalent:
# Option 1: Large batch
batch_size=16, gradient_accumulation_steps=1

# Option 2: Small batch with accumulation (memory efficient)
batch_size=2, gradient_accumulation_steps=8
```

### Learning Rate Scheduling
```python
# Cosine schedule with warmup (recommended)
lr_scheduler_type="cosine"
warmup_steps=100

# Linear schedule
lr_scheduler_type="linear"
warmup_steps=100

# Constant learning rate
lr_scheduler_type="constant"
```

### LoRA Parameters
```python
# Rank vs Quality vs Speed
lora_r=8   # Fast, less expressive
lora_r=16  # Balanced (recommended)
lora_r=64  # Slow, more expressive

# Alpha (typically 2x rank)
lora_alpha=2 * lora_r
```

---

<div align="center">

**Generated with [Claude Code](https://claude.com/claude-code)** 🤖

</div>
