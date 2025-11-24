# Vietnamese NER Extraction: Multi-Method Comparison

> A comprehensive project comparing different approaches for Vietnamese Named Entity Recognition (NER): Prompt Engineering, RAG, Fine-Tuning and LangExtract

---

## Project Overview

This project implements and evaluates different methods for extracting named entities (Person, Organizations, Address) from Vietnamese text:

1. **Prompt Engineering** - Zero-shot extraction using LLM prompting strategies
2. **RAG (Retrieval-Augmented Generation)** - Few-shot learning with semantic retrieval and reranking
3. **Fine-Tuning** - Parameter-efficient fine-tuning with LoRA

### Key Features

- **Multi-method comparison** with standardized evaluation
- **Vietnamese language focus** with specialized models
- **Production-ready** with vLLM, structured outputs, and GPU optimization
- **Comprehensive docs** for each pipeline
- **Modular architecture** with clean service-oriented design

---

## Project Structure

```
.
data/
   raw/                      # Raw Vietnamese articles
   processed/                # Processed datasets (JSON, JSONL)

src/
   config.py                 # Configuration for all methods
   schema.py                 # Entity schema definitions
   data_processor.py         # Data processing utilities
   prompt_engineering.py     # Prompt engineering pipeline
   rag_pipeline.py           # RAG with retrieval + reranking
   finetuning.py             # Fine-tuning with LoRA
   utils.py                  # Evaluation metrics & utilities

scripts/
   run_data_preparation.py   # Data extraction & preparation
   run_prompt_engineering_pipeline.py
   run_rag_pipeline.py
   run_finetuning_pipeline.py

notebooks/                   # Jupyter notebooks for experiments
   prompt_pipeline_experiments.ipynb
   rag_pipeline_experiments.ipynb
   finetuning_pipeline_experiments.ipynb

docs/                        # Documentation
   run_data_preparation.md
   run_prompt_engineering_pipeline.md
   run_rag_pipeline.md
   run_finetuning_pipeline.md

results/                     # Evaluation results
logs/                        # Pipeline execution logs
models/
    checkpoints/            # Fine-tuned model checkpoints
```

---

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/dangvansam/llm-data-extraction-test.git
cd llm-data-extraction-test

# Install dependencies
uv sync
```

### 2. Data Preparation (Optional)

Extract entities from raw articles using LangExtract:

```bash
python scripts/run_data_preparation.py --categories "Chinh tri Xa hoi"
```

**Output:** `data/processed/{train,test}.json` and fine-tuning datasets

**[Full Documentation](docs/run_data_preparation.md)**

### 3. Run Extraction Pipelines

#### 3.1. Prompt Engineering

```bash
python scripts/run_prompt_engineering_pipeline.py
```

**[Full Documentation](docs/run_prompt_engineering_pipeline.md)**

#### 3.2. RAG Pipeline

```bash
python scripts/run_rag_pipeline.py
```

**[Full Documentation](docs/run_rag_pipeline.md)**

#### 3.3. Fine-Tuning

```bash
python scripts/run_finetuning_pipeline.py
```

**[Full Documentation](docs/run_finetuning_pipeline.md)**

---

## Benchmark Results

### Performance Comparison

| Method | F1 Score | Precision | Recall | Accuracy | Speed (samples/s) |
|--------|----------|-----------|--------|----------|-------------------|
| **Prompt Engineering (Base Model)** | 0.5702 | 0.6682 | 0.4972 | 0.4065 | 0.26 |
| **Prompt Engineering (Fine-tuned)** | 0.6724 | 0.7533 | 0.6072 | 0.5128 | 0.21 |
| **RAG** | 0.8116 | 0.8166 | 0.8068 | 0.6877 | 0.06 |

### Per-Entity Performance

#### Prompt Engineering (Model: Qwen3-4B-Instruct-2507)

| Entity Type | Precision | Recall | F1 Score | Accuracy |
|-------------|-----------|--------|----------|----------|
| Person | 0.7384 | 0.6554 | 0.6944 | 0.5510 |
| Organizations | 0.6730 | 0.5536 | 0.6075 | 0.4388 |
| Address | 0.6212 | 0.3856 | 0.4758 | 0.3174 |

#### Prompt Engineering (Fine-tuned Model: Qwen3-4B-Instruct-2507)

| Entity Type | Precision | Recall | F1 Score | Accuracy |
|-------------|-----------|--------|----------|----------|
| Person | 0.8008 | 0.7228 | 0.7598 | 0.6292 |
| Organizations | 0.7220 | 0.6023 | 0.6567 | 0.4913 |
| Address | 0.7574 | 0.5627 | 0.6457 | 0.4802 |

#### RAG Pipeline

| Entity Type | Precision | Recall | F1 Score | Accuracy |
|-------------|-----------|--------|----------|----------|
| Person | 0.8516 | 0.8165 | 0.8337 | 0.7420 |
| Organizations | 0.8569 | 0.8168 | 0.8363 | 0.7101 |
| Address | 0.7729 | 0.7947 | 0.7836 | 0.6492 |

---

### Configuration Used

- **Model:** Qwen/Qwen3-4B-Instruct-2507
- **Dataset:** 100 test samples
- **Extraction Mode:** Structured Output with Thinking
- **Embedding (RAG):** AITeamVN/Vietnamese_Embedding_v2
- **Reranker (RAG):** AITeamVN/Vietnamese_Reranker

---

## Use Cases & Method Selection

### When to Use Each Method

**Prompt Engineering**
- Quick prototyping and testing
- No labeled training data available
- Limited computational resources
- Experimenting with different approaches

**RAG**
- Need better quality than prompt engineering
- Have labeled examples but no time to train
- Want few-shot learning benefits
- Corpus examples are representative

**Fine-Tuning**
- Best quality required for production
- Have sufficient labeled training data (1000+ samples)
- Can afford training time and resources
- Domain-specific vocabulary and patterns

---

## Configuration

### Model Settings

All pipelines use `Qwen/Qwen3-4B-Instruct-2507` by default. Configure in `src/config.py`:

```python
# 1. Prompt Engineering
config = NERPromptEngineeringConfig(
    model_name="Qwen/Qwen3-4B-Instruct-2507",
    extraction_mode=ExtractionMode.STRUCTURED_OUTPUT,
    enable_thinking=True
)


# 2. RAG
config = NERRagConfig(
    model_name="Qwen/Qwen3-4B-Instruct-2507",
    embedding_model="AITeamVN/Vietnamese_Embedding_v2",
    rerank_model="AITeamVN/Vietnamese_Reranker",
    top_k_retrieval=3
)


# 3. Fine-Tuning
config = NERFineTuningConfig(
   model_name="Qwen/Qwen3-4B-Instruct-2507",
   output_dir="models/checkpoints/finetuned",
)
# Testing fine-tuning with Prompt Engineering
config = NERPromptEngineeringConfig(
    model_name="models/checkpoints/finetuned",
    extraction_mode=ExtractionMode.RAW,
)
```

---

## Key Features

### RAG Pipeline Enhancements

- **Semantic Search:** Uses Vietnamese embedding models for accurate retrieval
- **Reranking:** Cross-encoder reranking for improved relevance
  - Retrieves 3x candidates initially
  - Reranks with transformer-based scorer
  - Returns top-k best matches

### Prompt Engineering

- **Multiple Modes:**
  - RAW: Fast, free-form JSON generation
  - STRUCTURED_OUTPUT: Schema-validated JSON with vLLM
- **Thinking Mode:** Optional reasoning before extraction
- **Zero-shot:** No training required

### Fine-Tuning

- **LoRA:** Parameter-efficient training
- **Quantization:** 4-bit/8-bit support for memory efficiency
- **Flexible:** Full fine-tuning or LoRA options

---

## Logging

All pipelines write detailed logs to `logs/`:

- `logs/data_preparation.log`
- `logs/run_prompt_engineering_pipeline.log`
- `logs/run_rag_pipeline.log`
- `logs/run_finetuning_pipeline.log`

---

## Technologies

### Core Libraries

- **LLM Framework:** vLLM (high-throughput inference)
- **Training:** Transformers, PEFT, TRL
- **Retrieval:** FAISS, Sentence-Transformers
- **Evaluation:** Scikit-learn custom metrics
- **Logging:** Loguru

### Models

- **LLM:** Qwen3-4B-Instruct-2507
- **Embedding:** AITeamVN/Vietnamese_Embedding_v2
- **Reranker:** AITeamVN/Vietnamese_Reranker
- **LangExtract:** Google Gemini 2.5 Pro

---

## Documentation

Each pipeline has comprehensive documentation:

- [Data Preparation Guide](docs/run_data_preparation.md)
- [Prompt Engineering Pipeline](docs/run_prompt_engineering_pipeline.md)
- [RAG Pipeline](docs/run_rag_pipeline.md)
- [Fine-Tuning Pipeline](docs/run_finetuning_pipeline.md)

---

## Evaluation

Metrics calculated:
- **Accuracy:** Correct predictions / Total samples
- **Precision:** Correct predictions / Total predictions
- **Recall:** Correct predictions / Total ground truth
- **F1 Score:** Harmonic mean of precision & recall
- **Per-entity metrics:** Separate scores for Person, Organizations, Address