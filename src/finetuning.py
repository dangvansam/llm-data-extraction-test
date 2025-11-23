"""Fine-tuning approach for NER extraction."""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from tqdm import tqdm

from .config import NERFineTuningConfig
from .data_loader import NERDataLoader


class FineTunedNERExtractor:
    """NER extraction using fine-tuned LLM."""

    def __init__(self, config: NERFineTuningConfig = None, model_path: Optional[Path] = None):
        """
        Initialize fine-tuned NER extractor.

        Args:
            config: Configuration object
            model_path: Path to fine-tuned model (if loading existing model)
        """
        self.config = config or NERFineTuningConfig()
        self.model = None
        self.tokenizer = None

        if model_path:
            self.load_model(model_path)

    def prepare_training_data(self, samples: List[Dict]) -> Dataset:
        """
        Prepare data for fine-tuning.

        Args:
            samples: List of samples with 'text' and 'entities' keys

        Returns:
            HuggingFace Dataset
        """
        formatted_samples = []

        for sample in samples:
            text = sample["text"]
            entities = sample["entities"]

            # Create instruction-response pair
            instruction = f"""Extract named entities from the following text. Return a JSON object with keys: person, organizations, address.

Text: {text}"""

            response = json.dumps(entities, ensure_ascii=False)

            # Format for instruction tuning
            formatted_text = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{response}<|eot_id|>"""

            formatted_samples.append({"text": formatted_text})

        return Dataset.from_list(formatted_samples)

    def tokenize_function(self, examples):
        """Tokenize examples for training."""
        outputs = self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=self.config.max_length,
            padding="max_length",
        )
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs

    def train(self, train_dataset: List[Dict], val_dataset: List[Dict] = None, output_dir: Path = None):
        """
        Fine-tune the model.

        Args:
            train_dataset: Training data
            val_dataset: Validation data
            output_dir: Directory to save the model
        """
        output_dir = output_dir or self.config.output_dir
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("Preparing training data...")
        train_ds = self.prepare_training_data(train_dataset)

        if val_dataset:
            val_ds = self.prepare_training_data(val_dataset)
        else:
            val_ds = None

        # Load tokenizer
        print(f"Loading tokenizer: {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # Tokenize datasets
        train_ds = train_ds.map(self.tokenize_function, batched=True, remove_columns=["text"])
        if val_ds:
            val_ds = val_ds.map(self.tokenize_function, batched=True, remove_columns=["text"])

        # Configure quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        # Load model
        print(f"Loading model: {self.config.model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        # Prepare model for training
        model = prepare_model_for_kbit_training(model)

        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Get PEFT model
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            logging_steps=10,
            save_steps=100,
            evaluation_strategy="steps" if val_ds else "no",
            eval_steps=100 if val_ds else None,
            save_total_limit=3,
            fp16=True,
            report_to="none",
            remove_unused_columns=False,
        )

        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
        )

        # Train
        print("Starting training...")
        trainer.train()

        # Save model
        print(f"Saving model to {output_dir}")
        model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        self.model = model

        print("Training complete!")

    def load_model(self, model_path: Path):
        """
        Load fine-tuned model.

        Args:
            model_path: Path to saved model
        """
        print(f"Loading fine-tuned model from {model_path}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Configure quantization for inference
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        # Load LoRA weights
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.model.eval()

        print("Model loaded successfully")

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

        # Preprocess text
        text = NERDataLoader.preprocess_text(text)

        # Create prompt
        prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Extract named entities from the following text. Return a JSON object with keys: person, organizations, address.

Text: {text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.config.max_length)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

        # Parse response
        entities = self._parse_response(response)

        return entities

    def _parse_response(self, response: str) -> Dict[str, List[str]]:
        """Parse JSON response from model."""
        entities = {"person": [], "organizations": [], "address": []}

        try:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())

                for key in self.config.entity_types:
                    if key in parsed:
                        value = parsed[key]
                        if isinstance(value, list):
                            entities[key] = value
                        elif isinstance(value, str):
                            entities[key] = [value] if value else []
            else:
                print(f"Warning: No JSON found in response: {response[:100]}")

        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse JSON: {e}")

        return entities

    def batch_extract(self, texts: List[str], show_progress: bool = True) -> List[Dict[str, List[str]]]:
        """
        Extract entities from multiple texts.

        Args:
            texts: List of input texts
            show_progress: Whether to show progress bar

        Returns:
            List of entity dictionaries
        """
        results = []

        iterator = tqdm(texts, desc="Extracting entities (Fine-tuned)") if show_progress else texts

        for text in iterator:
            entities = self.extract_entities(text)
            results.append(entities)

        return results

    def evaluate_on_dataset(self, dataset: List[Dict]) -> tuple:
        """
        Run extraction on a dataset.

        Args:
            dataset: List of samples with 'text' and 'entities' keys

        Returns:
            Tuple of (predictions, ground_truth)
        """
        texts = [sample["text"] for sample in dataset]
        predictions = self.batch_extract(texts)
        ground_truth = [sample["entities"] for sample in dataset]

        return predictions, ground_truth
