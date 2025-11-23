from pathlib import Path
from typing import Dict, List

import torch
from datasets import Dataset
from loguru import logger
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

from .config import NERFineTuningConfig
from .data_processor import DataProcessor
from .utils import parse_response


class FineTunedNERExtractor:
    """NER extraction using fine-tuned LLM."""

    def __init__(
        self,
        config: NERFineTuningConfig = None
    ):
        """
        Initialize fine-tuned NER extractor.

        Args:
            config: Configuration object (contains all training settings)
            model_path: Path to fine-tuned model (if loading existing model)
        """
        self.config = config or NERFineTuningConfig()
        self.model = None
        self.tokenizer = None
        
        self._initialize_model()

    def _initialize_model(self):
        """Initialize model, tokenizer, and LoRA adapters for training."""
        logger.info(f"Loading tokenizer: {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            truncation_side="left",
            model_max_length=self.config.max_seq_length
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        quantization_config = None
        if self.config.load_in_4bit or self.config.load_in_8bit:
            logger.info(f"Configuring {'4-bit' if self.config.load_in_4bit else '8-bit'} quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=self.config.load_in_4bit,
                load_in_8bit=self.config.load_in_8bit,
                bnb_4bit_use_double_quant=True if self.config.load_in_4bit else False,
                bnb_4bit_quant_type="nf4" if self.config.load_in_4bit else None,
                bnb_4bit_compute_dtype=torch.float16 if self.config.load_in_4bit else None,
            )

        logger.info(f"Loading model: {self.config.model_name}")
        logger.info(f"Max sequence length: {self.config.max_seq_length}")
        logger.info(f"Full finetuning: {self.config.full_finetuning}")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=quantization_config,
            dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )

        # Prepare model for training if using quantization
        if quantization_config:
            self.model = prepare_model_for_kbit_training(self.model)

        # Configure LoRA if not doing full finetuning
        if not self.config.full_finetuning:
            logger.info("Configuring LoRA adapters...")
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=[
                    "q_proj", "k_proj",
                    "v_proj", "o_proj",
                    "gate_proj", "up_proj",
                    "down_proj"
                ],
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )

            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        else:
            logger.info("Using full finetuning mode")
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Trainable parameters: {trainable_params:,}")

        logger.success("Model initialized successfully")

    def format_function(self, samples: List[Dict]) -> Dict[str, List[str]]:
        """Tokenize samples for training using chat template."""
        formatted_texts = []
        for messages in samples["messages"]:
            formatted_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=self.config.enable_thinking
            )
            formatted_texts.append(formatted_text)

        return {"text": formatted_texts}

    def train(self):
        """
        Fine-tune the model.
        """

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_dir = output_dir / "logs"

        logger.info("Loading training dataset...")
        train_dataset = DataProcessor.load_jsonl(Path(self.config.train_data_path))
        train_dataset = Dataset.from_list(train_dataset)
        train_dataset = train_dataset.map(self.format_function, batched=True, remove_columns=["messages"])

        if self.config.val_data_path:
            logger.info("Loading validation dataset...")
            val_dataset = DataProcessor.load_jsonl(Path(self.config.val_data_path))
            val_dataset = Dataset.from_list(val_dataset)
            val_dataset = val_dataset.map(self.format_function, batched=True, remove_columns=["messages"])

        # Training configuration
        sft_config = SFTConfig(
            output_dir=str(output_dir),
            logging_dir=str(log_dir),
            num_train_epochs=self.config.num_train_epochs,
            max_steps=self.config.max_steps,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            lr_scheduler_type=self.config.lr_scheduler_type,
            optim=self.config.optim,
            logging_steps=self.config.logging_steps,
            save_strategy="steps",
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            eval_strategy="steps" if val_dataset else "no",
            eval_steps=self.config.eval_steps if val_dataset else None,
            fp16=True if self.model.dtype == torch.float16 else False,
            bf16=True if self.model.dtype == torch.bfloat16 else False,
            report_to=self.config.report_to,
            dataset_text_field="text",
            packing=False,
            remove_unused_columns=False,
            seed=2025,
            resume_from_checkpoint=self.config.resume_from_checkpoint,
        )

        # Create trainer
        logger.info("Creating SFT Trainer...")
        trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            args=sft_config
        )

        if torch.cuda.is_available():
            gpu_stats = torch.cuda.get_device_properties(0)
            start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
            max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
            logger.info(f"GPU: {gpu_stats.name}")
            logger.info(f"Max memory: {max_memory} GB")
            logger.info(f"Reserved memory: {start_gpu_memory} GB")

        logger.info("Starting training...")

        trainer_stats = trainer.train(resume_from_checkpoint=self.config.resume_from_checkpoint)

        if torch.cuda.is_available():
            used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
            used_memory_for_training = round(used_memory - start_gpu_memory, 3)
            used_percentage = round(used_memory / max_memory * 100, 3)
            training_percentage = round(used_memory_for_training / max_memory * 100, 3)

            logger.info(f"Training time: {trainer_stats.metrics['train_runtime']:.2f} seconds")
            logger.info(f"Training time: {round(trainer_stats.metrics['train_runtime']/60, 2)} minutes")
            logger.info(f"Peak reserved memory: {used_memory} GB")
            logger.info(f"Peak reserved memory for training: {used_memory_for_training} GB")
            logger.info(f"Peak reserved memory % of max: {used_percentage}%")
            logger.info(f"Peak training memory % of max: {training_percentage}%")

        final_output_dir = output_dir / "final"
        logger.info(f"Saving final model to {final_output_dir}")
        self.model.save_pretrained(final_output_dir)
        self.tokenizer.save_pretrained(final_output_dir)

        logger.success("Training complete!")

        return trainer_stats

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities from text using fine-tuned model.

        Args:
            text: Input text

        Returns:
            Dictionary with extracted entities
        """
        if self.model is None:
            raise ValueError("Model not loaded. Train or load a model first.")

        text = DataProcessor.preprocess_text(text)

        input = DataProcessor.format_chat_message(
            text=text,
            tokenizer=self.tokenizer,
            add_schema=self.config.add_schema,
            enable_thinking=self.config.enable_thinking
        )

        inputs = self.tokenizer(
            input,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=self.config.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

        entities = parse_response(response, self.config.entity_types)

        return entities