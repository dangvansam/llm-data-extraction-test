from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

from .schema import ExtractionMode

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
OUTPUTS_DIR = MODELS_DIR / "outputs"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

NUM_TRAIN_SAMPLES = 1000
NUM_TEST_SAMPLES = 100

for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, CHECKPOINTS_DIR, OUTPUTS_DIR, RESULTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


@dataclass
class BaseNERConfig:
    """Base configuration for NER extraction."""

    # Entity types
    entity_types: List[str] = field(default_factory=lambda: ["person", "organizations", "address"])

    temperature: float = 0.1
    top_p: float = 0.9
    max_length: int = 2048
    max_new_tokens: int = 1024
    do_sample: bool = True
    enable_thinking: bool = True

    data_dir: Path = field(default=PROCESSED_DATA_DIR)
    results_dir: Path = field(default=RESULTS_DIR)


@dataclass
class NERPromptEngineeringConfig(BaseNERConfig):
    """Configuration for Prompt Engineering method."""
    max_length: int = 4096

    # Model settings
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    tokenizer_name: str = None

    # Extraction mode
    extraction_mode: ExtractionMode = ExtractionMode.RAW

    # Prompt format
    add_schema: bool = False  # Include JSON schema in system instruction for better structured output

    # vLLM settings (used when extraction_mode = STRUCTURED_OUTPUT)
    vllm_structured_outputs_backend: str = "auto" # "auto", "xgrammar", "guidance", "outlines"
    vllm_gpu_memory_utilization: float = 0.3

    # Performance
    batch_size: int = 1


@dataclass
class NERRagConfig(BaseNERConfig):
    """Configuration for RAG method."""
    max_length: int = 4096

    # LLM settings
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"

    # Embedding settings
    embedding_model: str = "AITeamVN/Vietnamese_Embedding_v2"
    # embedding_model: str = "BAAI/bge-small-en-v1.5"
    # embedding_model: str = "intfloat/multilingual-e5-large-instruct"
    # embedding_model: str = "Snowflake/snowflake-arctic-embed-l-v2.0"
    normalize_embeddings: bool = True
    
    rerank_model: str = "AITeamVN/Vietnamese_Reranker"
    # rerank_model: str = "Alibaba-NLP/gte-multilingual-reranker-base"
    # rerank_model: str = "BAAI/bge-reranker-v2-m3"

    # Chunking settings
    chunk_size: int = 1024
    chunk_overlap: int = 0

    # Retrieval settings
    top_k_retrieval: int = 3
    similarity_threshold: Optional[float] = None

    # Vector store settings
    index_type: str = "flat"
    use_gpu_index: bool = False

    # Index paths
    index_path: Optional[Path] = None

    vllm_structured_outputs_backend: str = "auto"
    vllm_gpu_memory_utilization: float = 0.3



@dataclass
class NERFineTuningConfig(BaseNERConfig):
    """Configuration for Fine-tuning method."""

    # Model settings
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    full_finetuning: bool = False

    # Prompt settings
    add_schema: bool = False  # Include JSON schema in system instruction

    # Training settings
    learning_rate: float = 1e-4
    num_train_epochs: int = 2
    max_steps: int = 0
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 100
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    optim: str = "adamw_torch_fused"  # adamw_8bit

    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # Quantization settings
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    # Training optimization
    fp16: bool = True
    gradient_checkpointing: bool = False
    max_grad_norm: float = 1.0

    # Logging and saving
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 1
    eval_steps: Optional[int] = 500

    # Data paths
    train_data_path: Optional[Path] = field(default=PROCESSED_DATA_DIR / "train_finetuning_chat.jsonl")
    val_data_path: Optional[Path] = field(default=PROCESSED_DATA_DIR / "val_finetuning_chat.jsonl")

    # Output paths
    output_dir: Path = field(default=CHECKPOINTS_DIR / "finetuned")

    # Training options
    resume_from_checkpoint: Optional[str] = None
    report_to: str = "none"  # "none", "wandb", or "tensorboard"


@dataclass
class NERLangExtractConfig(BaseNERConfig):
    """Configuration for LangExtract method."""

    # Gemini model settings
    # model_id: str = "gemini-2.5-flash"  # or "gemini-2.0-flash", "gemini-2.5-pro"
    model_id: str = "gemini-2.5-pro"  # or "gemini-2.0-flash", "gemini-2.5-pro"

    # Extraction settings
    extraction_passes: int = 2  # Number of independent extraction passes
    max_workers: int = 5  # Parallel processing workers
    max_char_buffer: int = 2000  # Chunk size for long documents

    # API settings
    api_key: str = os.environ.get("GEMINI_API_KEY")
    timeout: int = 60
    retry_attempts: int = 2

    # Output settings
    save_jsonl: bool = True
    create_visualization: bool = True
    include_attributes: bool = True  # Extract rich entity attributes

    # Performance optimization
    batch_delay: float = 0.1  # Delay between batches to avoid rate limits
    chunk_overlap_chars: int = 100  # Overlap between chunks


# Default configurations for each method
DEFAULT_PROMPT_CONFIG = NERPromptEngineeringConfig()
DEFAULT_RAG_CONFIG = NERRagConfig()
DEFAULT_FINETUNING_CONFIG = NERFineTuningConfig()
DEFAULT_LANGEXTRACT_CONFIG = NERLangExtractConfig()
