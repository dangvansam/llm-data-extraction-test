import json
import re
from typing import Dict, List

import torch
from loguru import logger
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import NERPromptEngineeringConfig
from .data_loader import NERDataLoader
from .schema import ExtractionMode, NEREntities

try:
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None
    StructuredOutputsParams = None

try:
    import outlines
    OUTLINES_AVAILABLE = True
except ImportError:
    OUTLINES_AVAILABLE = False
    outlines = None


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
        self.tokenizer = None
        self.vllm_model = None
        self.outlines_model = None

        # Determine extraction mode
        mode = self.config.extraction_mode

        # Load appropriate model based on mode
        if mode == ExtractionMode.STRUCTURED_OUTPUT:
            if not VLLM_AVAILABLE:
                raise ImportError("vLLM is not installed. Install it with: pip install vllm")
            self._load_vllm_model()
        elif mode == ExtractionMode.OUTLINES:
            if not OUTLINES_AVAILABLE:
                raise ImportError("Outlines is not installed. Install it with: pip install outlines")
            self._load_outlines_model()
        else:  # RAW mode
            self._load_local_model()

    def _load_local_model(self):
        """Load local LLM model."""
        logger.info(f"Loading model: {self.config.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype="auto",
            device_map="auto"
        )

        logger.success("Model loaded successfully")

    def _load_vllm_model(self):
        """Load vLLM model for efficient inference."""
        logger.info(f"Loading vLLM model: {self.config.model_name}")

        self.vllm_model = LLM(
            model=self.config.model_name,
            max_model_len=self.config.max_length,
            tensor_parallel_size=self.config.vllm_tensor_parallel_size,
            gpu_memory_utilization=self.config.vllm_gpu_memory_utilization,
            kv_cache_memory_bytes=1024 * 1024 * 1024
        )

        logger.success("vLLM model loaded successfully")

    def _load_outlines_model(self):
        """Load model with Outlines for structured generation."""
        logger.info(f"Loading Outlines model: {self.config.model_name}")

        # Load base model and tokenizer
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        # Wrap with Outlines
        self.outlines_model = outlines.models.transformers(
            base_model,
            tokenizer
        )

        logger.success("Outlines model loaded successfully")

    def create_prompt(self, text: str, use_chat_format: bool = False) -> str:
        """
        Create extraction prompt.

        NOTE: This prompt format is also used in data_processor.py for training data.
        Any changes here should be reflected in DataProcessorService.create_ner_prompt()
        to maintain consistency between training and inference.

        Args:
            text: Input text
            use_chat_format: If True, returns formatted chat messages (for chat-tuned models)

        Returns:
            Formatted prompt (string) or chat messages (for chat models)
        """
        # Get system instruction from centralized source (same as training data)
        system_message = NEREntities.get_system_instruction(add_schema=self.config.add_schema)

        if use_chat_format:
            # Return chat messages format (matches training data structure)
            # This will be used with apply_chat_template for chat models
            logger.debug(f"Chat Prompt:\n{system_message}\nUser: {text}")
            return [
                {"role": "system", "content": system_message},
                {"role": "user", "content": text}
            ]
        else:
            # Return traditional single-prompt format (for base models)
            prompt = f"""{system_message}

Text:
{text}

JSON output:"""
            logger.debug(f"Instruction Prompt:\n{prompt}")
            return prompt

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities from text.

        Args:
            text: Input text

        Returns:
            Dictionary with extracted entities
        """
        text = NERDataLoader.preprocess_text(text)

        # Create prompt in the format matching training data
        prompt = self.create_prompt(text, use_chat_format=self.config.use_chat_format)

        mode = self.config.extraction_mode

        if mode == ExtractionMode.STRUCTURED_OUTPUT:
            response = self._query_vllm_model(prompt)
            entities = self._parse_response(response)
        elif mode == ExtractionMode.OUTLINES:
            entities = self._query_outlines_model(prompt)
        else:  # RAW mode
            response = self._query_local_model(prompt)
            entities = self._parse_response(response)

        return entities

    def _query_local_model(self, prompt) -> str:
        """
        Query local LLM model.

        Args:
            prompt: Either a string prompt or list of chat messages
        """
        # Handle chat format vs string prompt
        if isinstance(prompt, list):
            # Chat messages format - apply chat template if available
            if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template is not None:
                # Use tokenizer's chat template
                formatted_prompt = self.tokenizer.apply_chat_template(
                    prompt,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # Fallback: manually format chat messages
                logger.warning("Chat template not available, using manual formatting")
                system_msg = prompt[0]["content"]
                user_msg = prompt[1]["content"]
                formatted_prompt = f"""{system_msg}

Text:
{user_msg}

JSON output:"""
        else:
            # String prompt - use as is
            formatted_prompt = prompt

        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            # enable_thinking=self.config.enable_thinking
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
        return response

    def _query_vllm_model(self, prompt: str) -> str:
        """Query vLLM model with structured output."""
        # Use structured output with JSON schema
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
        return response

    def _query_outlines_model(self, prompt: str) -> Dict[str, List[str]]:
        """Query model using Outlines for structured generation."""
        # Create structured generator with Pydantic model
        generator = outlines.generate.json(
            self.outlines_model,
            NEREntities,
            max_tokens=self.config.max_new_tokens,
        )

        # Generate structured output
        result = generator(prompt)

        # Parse result - Outlines returns JSON string
        if isinstance(result, str):
            entities_obj = NEREntities.model_validate_json(result)
        else:
            # Result might already be a dict
            entities_obj = NEREntities(**result)

        # Convert to dictionary
        entities = {
            "person": entities_obj.person,
            "organizations": entities_obj.organizations,
            "address": entities_obj.address,
        }

        return entities

    def _parse_response(self, response: str) -> Dict[str, List[str]]:
        """
        Parse JSON response from model.

        Args:
            response: Model output

        Returns:
            Parsed entities dictionary
        """
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
                logger.info(f"No JSON found in response: {response}")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e} {response}")

        return entities

    def batch_extract(
        self, texts: List[str], show_progress: bool = True
    ) -> List[Dict[str, List[str]]]:
        """
        Extract entities from multiple texts.

        Args:
            texts: List of input texts
            show_progress: Whether to show progress bar

        Returns:
            List of entity dictionaries
        """
        mode = self.config.extraction_mode

        # Use optimized batch processing for vLLM
        if mode == ExtractionMode.STRUCTURED_OUTPUT:
            return self._batch_extract_vllm(texts, show_progress)

        # Use optimized batch processing for Outlines
        if mode == ExtractionMode.OUTLINES:
            return self._batch_extract_outlines(texts, show_progress)

        # Standard sequential processing for RAW mode
        results = []
        iterator = tqdm(texts, desc="Extracting entities") if show_progress else texts

        for text in iterator:
            entities = self.extract_entities(text)
            results.append(entities)

        return results

    def _batch_extract_vllm(
        self, texts: List[str], show_progress: bool = True
    ) -> List[Dict[str, List[str]]]:
        """
        Batch extract using vLLM for better performance.

        Args:
            texts: List of input texts
            show_progress: Whether to show progress bar

        Returns:
            List of entity dictionaries
        """
        # Preprocess texts
        processed_texts = [NERDataLoader.preprocess_text(text) for text in texts]

        # Create prompts (using chat format if configured)
        prompts = [self.create_prompt(text, use_chat_format=self.config.use_chat_format) for text in processed_texts]

        # Setup sampling params with structured output
        json_schema = NEREntities.get_json_schema()
        structured_outputs_params = StructuredOutputsParams(json=json_schema)
        sampling_params = SamplingParams(
            max_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            structured_outputs=structured_outputs_params,
        )

        # Batch generate
        if show_progress:
            logger.info(f"Batch extracting entities from {len(prompts)} texts using vLLM")

        outputs = self.vllm_model.generate(
            prompts=prompts,
            sampling_params=sampling_params,
        )

        # Parse responses
        results = []
        iterator = tqdm(outputs, desc="Parsing responses") if show_progress else outputs

        for output in iterator:
            response = output.outputs[0].text
            entities = self._parse_response(response)
            results.append(entities)

        return results

    def _batch_extract_outlines(
        self, texts: List[str], show_progress: bool = True
    ) -> List[Dict[str, List[str]]]:
        """
        Batch extract using Outlines for structured generation.

        Args:
            texts: List of input texts
            show_progress: Whether to show progress bar

        Returns:
            List of entity dictionaries
        """
        # Preprocess texts
        processed_texts = [NERDataLoader.preprocess_text(text) for text in texts]

        # Create prompts (using chat format if configured)
        prompts = [self.create_prompt(text, use_chat_format=self.config.use_chat_format) for text in processed_texts]

        # Create structured generator
        generator = outlines.generate.json(
            self.outlines_model,
            NEREntities,
            max_tokens=self.config.max_new_tokens,
        )

        if show_progress:
            logger.info(f"Batch extracting entities from {len(prompts)} texts using Outlines")

        # Process sequentially (Outlines doesn't have native batch support like vLLM)
        results = []
        iterator = tqdm(prompts, desc="Extracting with Outlines") if show_progress else prompts

        for prompt in iterator:
            result = generator(prompt)

            # Parse result
            if isinstance(result, str):
                entities_obj = NEREntities.model_validate_json(result)
            else:
                entities_obj = NEREntities(**result)

            entities = {
                "person": entities_obj.person,
                "organizations": entities_obj.organizations,
                "address": entities_obj.address,
            }
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


