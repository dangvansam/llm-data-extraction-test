import sys

from loguru import logger

sys.path.append(".")

from src.config import (
    CHECKPOINTS_DIR,
    PROCESSED_DATA_DIR,
    NERFineTuningConfig,
)
from src.pipeline.finetuning_pipeline import FineTunedNERExtractor

if __name__ == "__main__":
    
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

    if not config.train_data_path.exists():
        logger.error(f"Training data not found: {config.train_data_path}")
        raise ValueError("Please run data preparation script first")

    if not config.val_data_path.exists():
        logger.warning(f"Validation data not found: {config.val_data_path}")
        config.val_data_path = None

    logger.info("=" * 40)
    logger.info("FINE-TUNING")
    logger.info("=" * 40)
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Max sequence length: {config.max_seq_length}")
    logger.info(f"Enable thinking: {config.enable_thinking}")
    logger.info(f"Quantization: {'4-bit' if config.load_in_4bit else '8-bit' if config.load_in_8bit else 'None'}")
    logger.info(f"Training mode: {'Full finetuning' if config.full_finetuning else 'LoRA'}")
    logger.info(f"Epochs: {config.num_train_epochs}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Output: {config.output_dir}")
    logger.info("=" * 40)

    logger.info("Initializing model...")
    extractor = FineTunedNERExtractor(config)

    logger.info("Starting fine-tuning...")
    trainer_stats = extractor.train()

    logger.success("=" * 40)
    logger.success("TRAINING COMPLETE!")
    logger.success("=" * 40)
    logger.info(f"Model saved to: {config.output_dir / 'final'}")

    if trainer_stats:
        runtime = trainer_stats.metrics.get('train_runtime', 0)
        logger.info(f"Training time: {runtime:.2f} seconds ({runtime/60:.2f} minutes)")
        logger.info(f"Training samples: {trainer_stats.metrics.get('train_samples', 0)}")

        if 'train_loss' in trainer_stats.metrics:
            logger.info(f"Final training loss: {trainer_stats.metrics['train_loss']:.4f}")