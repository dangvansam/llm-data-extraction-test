"""RAG-based approach for NER extraction."""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import NERRagConfig
from .data_loader import NERDataLoader


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
        print(f"Loading embedding model: {self.config.embedding_model}")
        self.embedding_model = SentenceTransformer(self.config.embedding_model)

        # Load LLM
        self._load_llm()

        # Initialize vector store
        self.index = None
        self.corpus_texts = []

        if self.corpus:
            self.build_index(self.corpus)

    def _load_llm(self):
        """Load language model for generation."""
        print(f"Loading LLM: {self.config.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

        print("LLM loaded successfully")

    def build_index(self, corpus: List[Dict]):
        """
        Build FAISS index from corpus.

        Args:
            corpus: List of documents with 'text' and 'entities' keys
        """
        print("Building FAISS index...")

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

        print(f"Index built with {len(self.corpus_texts)} documents")

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
        Retrieve relevant documents.

        Args:
            query: Query text
            top_k: Number of documents to retrieve

        Returns:
            List of retrieved documents
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        top_k = top_k or self.config.top_k_retrieval

        # Encode query
        query_embedding = self.embedding_model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")

        # Search
        scores, indices = self.index.search(query_embedding, top_k)

        # Return retrieved documents
        retrieved = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.corpus):
                doc = self.corpus[idx].copy()
                doc["retrieval_score"] = float(score)
                retrieved.append(doc)

        return retrieved

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
        # Preprocess text
        text = NERDataLoader.preprocess_text(text)

        # Retrieve relevant documents
        retrieved_docs = self.retrieve(text)

        # Create prompt with context
        prompt = self.create_rag_prompt(text, retrieved_docs)

        # Generate response
        response = self._query_model(prompt)

        # Parse response
        entities = self._parse_response(response)

        return entities

    def _query_model(self, prompt: str) -> str:
        """Query LLM with prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.config.max_length)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        return response

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

        iterator = tqdm(texts, desc="Extracting entities (RAG)") if show_progress else texts

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
