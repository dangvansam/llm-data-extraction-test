import gc
import time
from typing import Dict, List

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import StructuredOutputsConfig
from vllm.sampling_params import StructuredOutputsParams

from .config import NERPromptEngineeringConfig
from .data_processor import DataProcessor
from .schema import ExtractionMode, NEREntities
from .utils import parse_response


class PromptNERExtractor:
    """NER extraction using prompt engineering."""

    def __init__(self, config: NERPromptEngineeringConfig = None):
        """
        Initialize prompt-based NER extractor.

        Args:
            config: Configuration object
        """
        self.config = config or NERPromptEngineeringConfig()
        self.model = None
        self.vllm_model = None

        tokenizer_name = self.config.tokenizer_name or self.config.model_name 
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        logger.info("Initializing PromptNERExtractor")
        logger.info(f"Model: {self.config.model_name}")
        logger.info(f"Tokenizer: {tokenizer_name}")
        logger.info(f"Extraction mode: {self.config.extraction_mode.value}")
        logger.info(f"Max length: {self.config.max_length}")
        logger.info(f"Max new tokens: {self.config.max_new_tokens}")
        logger.info(f"Temperature: {self.config.temperature}")
        logger.info(f"Top-p: {self.config.top_p}")
        logger.info(f"Add schema: {self.config.add_schema}")
        logger.info(f"Entity types: {self.config.entity_types}")

        mode = self.config.extraction_mode

        if mode == ExtractionMode.STRUCTURED_OUTPUT:
            self._load_vllm_model()
        else:
            self._load_local_model()

    def cleanup(self):
        """Clean up models and free GPU memory."""
        logger.info("Cleaning up models and freeing memory")

        try:
            if self.vllm_model is not None:
                self.vllm_model.llm_engine.engine_core.shutdown()
                if hasattr(self.vllm_model, 'llm_engine') and hasattr(self.vllm_model.llm_engine, 'model_executor'):
                    if hasattr(self.vllm_model.llm_engine.model_executor, 'driver_worker'):
                        del self.vllm_model.llm_engine.model_executor.driver_worker
                del self.vllm_model
                self.vllm_model = None

            if self.model is not None:
                del self.model
                self.model = None

            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None

            gc.collect()
            torch.cuda.empty_cache()

            logger.success("Memory cleanup completed")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def __del__(self):
        """Destructor to ensure cleanup when object is destroyed."""
        self.cleanup()

    def _load_local_model(self):
        """Load local LLM model."""
        logger.info(f"Loading model: {self.config.model_name}")
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            dtype="auto",
            device_map="auto"
        )

        logger.success("Model loaded successfully")

    def _load_vllm_model(self):
        """Load vLLM model."""
        logger.info(f"Loading vLLM model: {self.config.model_name}")
        
        self.vllm_model = LLM(
            model=self.config.model_name,
            max_model_len=self.config.max_length,
            gpu_memory_utilization=self.config.vllm_gpu_memory_utilization,
            kv_cache_memory_bytes=1024 * 1024 * 1024,
            structured_outputs_config=StructuredOutputsConfig(
                backend=self.config.vllm_structured_outputs_backend
            )
        )

        logger.success("vLLM model loaded successfully")

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities from text.

        Args:
            text: Input text

        Returns:
            Dictionary with extracted entities
        """
        start_time = time.time()

        text = DataProcessor.preprocess_text(text)
        input = DataProcessor.format_chat_message(
            text=text,
            tokenizer=self.tokenizer,
            add_schema=self.config.add_schema,
            enable_thinking=self.config.enable_thinking
        )

        if self.config.extraction_mode == ExtractionMode.STRUCTURED_OUTPUT:
            response = self._query_vllm_model(input)
            entities = parse_response(response, self.config.entity_types)
        else:
            response = self._query_local_model(input)
            entities = parse_response(response, self.config.entity_types)

        elapsed_time = time.time() - start_time
        logger.info(f"Extraction completed in {elapsed_time:.2f}s")

        return entities

    def _query_local_model(self, prompt: str) -> str:
        """
        Query local LLM model.

        Args:
            prompt: Either a string prompt or list of chat messages
        """
        inputs = self.tokenizer(
            prompt,
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

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        logger.debug(f"LLM response: {response}")
        return response

    def _query_vllm_model(self, prompt: str) -> str:
        """Query vLLM model with structured output."""
        json_schema = NEREntities.get_json_schema()
        structured_outputs_params = StructuredOutputsParams(json=json_schema)
        sampling_params = SamplingParams(
            max_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            structured_outputs=structured_outputs_params,
        )

        outputs = self.vllm_model.generate(
            prompts=[prompt],
            sampling_params=sampling_params,
        )

        response = outputs[0].outputs[0].text
        logger.debug(f"vLLM response: {response}")
        return response