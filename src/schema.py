"""Schema definitions for NER extraction."""

import json
from enum import Enum
from typing import Dict, List

from pydantic import BaseModel, Field


class ExtractionMode(Enum):
    """Extraction mode for NER."""
    RAW = "raw"  # Standard generation without structured output
    STRUCTURED_OUTPUT = "structured_output"  # vLLM structured output
    OUTLINES = "outlines"  # Outlines structured generation


class NEREntities(BaseModel):
    """Pydantic model for NER entities."""
    person: List[str] = Field(description="Names of people")
    organizations: List[str] = Field(description="Names of companies, institutions, organizations")
    address: List[str] = Field(description="Locations, places, addresses, geographical names")

    @classmethod
    def get_json_schema(cls) -> Dict:
        """
        Get JSON schema for structured output (used with vLLM).

        Returns:
            JSON schema for NER entities
        """
        return {
            "type": "object",
            "properties": {
                "person": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Names of people"
                },
                "organizations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Names of companies, institutions, organizations"
                },
                "address": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Locations, places, addresses, geographical names"
                }
            },
            "required": ["person", "organizations", "address"]
        }

    @classmethod
    def get_system_instruction(cls, add_schema: bool = False) -> str:
        """
        Get system instruction for NER extraction.

        This is the core instruction used in both training and inference.
        Ensures consistency across all extraction methods.

        Args:
            add_schema: If True, includes JSON schema in the instruction

        Returns:
            System instruction text
        """
        base_instruction = """You are an expert named entity recognition system. Extract named entities from the given text.

Extract the following entity types:
- person: Names of people
- organizations: Names of companies, institutions, organizations
- address: Locations, places, addresses, geographical names

Return ONLY a valid JSON object with these exact keys: "person", "organizations", "address"
Each key should have a list of extracted entities. If no entities found for a type, use an empty list."""

        if add_schema:
            schema = cls.get_json_schema()
            schema_str = json.dumps(schema, indent=2, ensure_ascii=False)
            return f"""{base_instruction}

JSON Schema:
```json
{schema_str}
```"""
        else:
            return base_instruction
