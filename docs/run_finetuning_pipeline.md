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

```python
from src.config import NERFineTuningConfig, CHECKPOINTS_DIR, PROCESSED_DATA_DIR

config = NERFineTuningConfig(
  # Model settings
  model_name="Qwen/Qwen3-4B-Instruct-2507",
  max_seq_length=2048,
  load_in_4bit=True,
  load_in_8bit=False,
  full_finetuning=False,

  # Prompt settings
  add_schema=False,

  # Training settings
  max_steps=3000,
  num_train_epochs=1,
  batch_size=4,
  learning_rate=2e-5,
  warmup_steps=10,
  gradient_accumulation_steps=1,
  lr_scheduler_type="linear",
  weight_decay=0.01,
  optim="adamw_torch_fused",  # adamw_8bit

  # LoRA settings (only used if full_finetuning=False)
  lora_r=8,
  lora_alpha=16,
  lora_dropout=0.02,

  # Logging and checkpointing
  logging_steps=10,
  save_steps=500,
  save_total_limit=3,
  eval_steps=500,

  # Output directory
  output_dir=CHECKPOINTS_DIR / "Qwen3-4B_finetuned_2048_lora_r8_a16_drop0.05_lr2e-5_warmup10_decay0.01_acc1_bs4",

  # Data paths
  train_data_path=PROCESSED_DATA_DIR / "train_finetuning_chat.jsonl",
  val_data_path=PROCESSED_DATA_DIR / "test_finetuning_chat.jsonl",

  # Training options
  resume_from_checkpoint=None,    # Set to checkpoint path to resume training
  report_to="wandb",               # "none", "wandb", or "tensorboard"
)
```

---

## 📝 Logs and Monitoring

### Training Logs

The pipeline writes detailed logs to:

📄 **[logs/run_finetuning_pipeline.log](logs/run_finetuning_pipeline.log)**
📄 **[Wandb Training Log](https://wandb.ai/dangvansam98/huggingface/runs/f57dii1x?nw=nwuserdangvansam98)**


### Testing Logs

📄 **[logs/run_prompt_engineering_pipeline_finetuned_model.log](logs/run_prompt_engineering_pipeline_finetuned_model.log)**