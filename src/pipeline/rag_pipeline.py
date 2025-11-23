import gc
import json
import time
from pathlib import Path
from typing import Dict, List

import faiss
import torch
from loguru import logger
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import StructuredOutputsConfig
from vllm.sampling_params import StructuredOutputsParams

from src.config import NERRagConfig
from src.data_processor import DataProcessor
from src.schema import NEREntities
from src.utils import parse_response


class RAGNERExtractor:
    """NER extraction using Retrieval-Augmented Generation."""

    def __init__(self, config: NERRagConfig = None, corpus: List[Dict] = None):
        """
        Initialize RAG-based NER extractor.

        Args:
            config: Configuration object
            corpus: Corpus of documents for retrieval
        """
        self.config = config or NERRagConfig()
        self.corpus = corpus or []

        # Load embedding model
        logger.info(f"Loading embedding model: {self.config.embedding_model}")
        self.embedding_model = SentenceTransformer(self.config.embedding_model)

        # Load reranker model if configured
        self.reranker = None
        self.reranker_tokenizer = None
        if hasattr(self.config, 'rerank_model') and self.config.rerank_model:
            logger.info(f"Loading reranker model: {self.config.rerank_model}")

            if "Vietnamese_Reranker" in self.config.rerank_model:
                self.reranker_tokenizer = AutoTokenizer.from_pretrained(self.config.rerank_model)
                self.reranker = AutoModelForSequenceClassification.from_pretrained(self.config.rerank_model)
                self.reranker.eval()
                logger.success("Reranker loaded as transformer model")
            else:
                try:
                    self.reranker = CrossEncoder(self.config.rerank_model, max_length=512)
                    logger.success("Reranker loaded as CrossEncoder")
                except Exception as e:
                    logger.warning(f"Failed to load as CrossEncoder, trying transformer model: {e}")
                    self.reranker_tokenizer = AutoTokenizer.from_pretrained(self.config.rerank_model)
                    self.reranker = AutoModelForSequenceClassification.from_pretrained(self.config.rerank_model)
                    self.reranker.eval()
                    logger.success("Reranker loaded as transformer model")

        self._load_llm()

        self.index = None
        self.corpus_texts = []

        if self.corpus:
            self.build_index(self.corpus)

    def _load_llm(self):
        """Load language model for generation using vLLM."""
        logger.info(f"Loading LLM with vLLM: {self.config.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.vllm_model = LLM(
            model=self.config.model_name,
            max_model_len=self.config.max_length,
            gpu_memory_utilization=self.config.vllm_gpu_memory_utilization,
            kv_cache_memory_bytes=1024 * 1024 * 1024,
            structured_outputs_config=StructuredOutputsConfig(
                backend=self.config.vllm_structured_outputs_backend
            )
        )
        logger.success("LLM loaded successfully with vLLM")

    def build_index(self, corpus: List[Dict]):
        """
        Build FAISS index from corpus.

        Args:
            corpus: List of documents with 'text' and 'entities' keys
        """
        logger.info("Building FAISS index...")

        self.corpus = corpus
        self.corpus_texts = [doc["text"] for doc in corpus]

        # Generate embeddings
        embeddings = self.embedding_model.encode(
            self.corpus_texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True
        )

        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity with normalized vectors)
        self.index.add(embeddings.astype("float32"))

        logger.success(f"Index built with {len(self.corpus_texts)} documents")

    def save_index(self, index_path: Path):
        """Save FAISS index to disk."""
        faiss.write_index(self.index, str(index_path))
        # Save corpus metadata
        metadata_path = index_path.with_suffix(".json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.corpus, f, ensure_ascii=False, indent=2)

    def load_index(self, index_path: Path):
        """Load FAISS index from disk."""
        self.index = faiss.read_index(str(index_path))
        # Load corpus metadata
        metadata_path = index_path.with_suffix(".json")
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.corpus = json.load(f)
        self.corpus_texts = [doc["text"] for doc in self.corpus]

    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Retrieve relevant documents with optional reranking.

        Args:
            query: Query text
            top_k: Number of documents to retrieve

        Returns:
            List of retrieved documents
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        top_k = top_k or self.config.top_k_retrieval

        initial_k = top_k * 3 if self.reranker else top_k

        query_embedding = self.embedding_model.encode(
            sentences=[query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        scores, indices = self.index.search(query_embedding, initial_k)

        retrieved = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.corpus):
                doc = self.corpus[idx].copy()
                doc["retrieval_score"] = float(score)
                retrieved.append(doc)

        if self.reranker and retrieved:
            logger.debug(f"Reranking {len(retrieved)} documents")
            retrieved = self.rerank(query, retrieved, top_k)

        return retrieved[:top_k]

    def rerank(self, query: str, documents: List[Dict], top_k: int) -> List[Dict]:
        """
        Rerank documents using cross-encoder or transformer model.

        Args:
            query: Query text
            documents: List of retrieved documents
            top_k: Number of top documents to keep

        Returns:
            Reranked list of documents
        """
        pairs = [[query, doc["text"]] for doc in documents]

        if isinstance(self.reranker, CrossEncoder):
            rerank_scores = self.reranker.predict(pairs)
        else:
            with torch.no_grad():
                inputs = self.reranker_tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=2304
                )
                inputs = {k: v.to(self.reranker.device) for k, v in inputs.items()}
                rerank_scores = self.reranker(**inputs, return_dict=True).logits.view(-1, ).float()
                rerank_scores = rerank_scores.cpu().numpy()

        for doc, score in zip(documents, rerank_scores):
            doc["rerank_score"] = float(score)

        reranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)

        logger.debug(f"Top rerank scores: {[doc['rerank_score'] for doc in reranked[:3]]}")

        return reranked[:top_k]

    def create_rag_prompt(self, text: str, retrieved_docs: List[Dict]) -> str:
        """
        Create RAG prompt with retrieved context.

        Args:
            text: Input text
            retrieved_docs: Retrieved documents

        Returns:
            Formatted prompt
        """
        # Format retrieved examples
        context = "Here are some similar examples with their extracted entities:\n\n"

        for i, doc in enumerate(retrieved_docs, 1):
            context += f"Example {i}:\n"
            context += f"Text: {doc['text'][:200]}...\n"
            context += f"Entities: {json.dumps(doc.get('entities', {}), ensure_ascii=False)}\n\n"

        prompt = f"""You are an expert named entity recognition system. Extract named entities from the given text.

{context}

Now extract entities from the following text:

Extract the following entity types:
- person: Names of people
- organizations: Names of companies, institutions, organizations
- address: Locations, places, addresses, geographical names

Return ONLY a valid JSON object with these exact keys: "person", "organizations", "address"
Each key should have a list of extracted entities. If no entities found for a type, use an empty list.

Text:
{text}

JSON output:"""

        return prompt

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities using RAG.

        Args:
            text: Input text

        Returns:
            Dictionary with extracted entities
        """
        start_time = time.time()
        logger.info("Starting RAG-based entity extraction")

        # Preprocess text
        text = DataProcessor.preprocess_text(text)

        # Retrieve relevant documents
        retrieved_docs = self.retrieve(text)

        # Create prompt with context
        prompt = self.create_rag_prompt(text, retrieved_docs)

        # Generate response
        response = self._query_vllm_model(prompt)

        # Parse response
        entities = parse_response(response, self.config.entity_types)

        elapsed_time = time.time() - start_time
        logger.info(f"RAG extraction completed in {elapsed_time:.2f}s")

        return entities

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

    def cleanup(self):
        """Clean up resources and free GPU memory."""
        try:
            logger.info("Cleaning up RAG extractor resources...")

            if hasattr(self, 'vllm_model') and self.vllm_model is not None:
                self.vllm_model.llm_engine.engine_core.shutdown()
                if hasattr(self.vllm_model, 'llm_engine') and hasattr(self.vllm_model.llm_engine, 'model_executor'):
                    if hasattr(self.vllm_model.llm_engine.model_executor, 'driver_worker'):
                        del self.vllm_model.llm_engine.model_executor.driver_worker
                del self.vllm_model
                self.vllm_model = None

            if hasattr(self, 'reranker') and self.reranker is not None:
                del self.reranker
                self.reranker = None

            if hasattr(self, 'reranker_tokenizer') and self.reranker_tokenizer is not None:
                del self.reranker_tokenizer
                self.reranker_tokenizer = None

            if hasattr(self, 'embedding_model') and self.embedding_model is not None:
                del self.embedding_model
                self.embedding_model = None

            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None

            gc.collect()
            torch.cuda.empty_cache()

            logger.success("RAG extractor cleanup completed")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def __del__(self):
        """Destructor to ensure cleanup when object is destroyed."""
        self.cleanup()
