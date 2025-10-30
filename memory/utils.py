"""
Utility functions for memory system.
"""
from datetime import datetime
from typing import List, Dict
from .llm_client import extract_facts_from_text


async def extract_facts(text: str, event_date: datetime, context: str = "") -> List[Dict[str, str]]:
    """
    Extract semantic facts from text using LLM.

    Uses LLM for intelligent fact extraction that:
    - Filters out social pleasantries and filler words
    - Creates self-contained statements with absolute dates
    - Handles conversational text well
    - Resolves relative time expressions to absolute dates

    Args:
        text: Input text (conversation, article, etc.)
        event_date: Reference date for resolving relative times
        context: Context about the conversation/document

    Returns:
        List of fact dictionaries with keys: 'fact' (text) and 'date' (ISO string)

    Raises:
        Exception: If LLM fact extraction fails
    """
    if not text or not text.strip():
        return []

    fact_dicts = await extract_facts_from_text(text, event_date, context)

    if not fact_dicts:
        raise Exception(f"LLM extracted 0 facts from text of length {len(text)}. This may indicate the text contains no meaningful information, or the LLM failed to extract facts.")

    return fact_dicts


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Similarity score between 0 and 1
    """
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have same dimension")

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sum(a * a for a in vec1) ** 0.5
    magnitude2 = sum(b * b for b in vec2) ** 0.5

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


def calculate_recency_weight(days_since: float, decay_rate: float = 0.1) -> float:
    """
    Calculate recency weight with exponential decay.

    Recent memories are weighted higher. The decay rate controls
    how quickly old memories fade.

    Args:
        days_since: Number of days since the memory was created
        decay_rate: How quickly memories fade (higher = faster decay)

    Returns:
        Weight between 0 and 1
    """
    import math
    return math.exp(-decay_rate * days_since)


def calculate_frequency_weight(access_count: int, max_boost: float = 2.0) -> float:
    """
    Calculate frequency weight based on access count.

    Frequently accessed memories are weighted higher.
    Uses logarithmic scaling to avoid over-weighting.

    Args:
        access_count: Number of times the memory was accessed
        max_boost: Maximum multiplier for frequently accessed memories

    Returns:
        Weight between 1.0 and max_boost
    """
    import math
    if access_count <= 0:
        return 1.0

    # Logarithmic scaling: log(access_count + 1) / log(10)
    # This gives: 0 accesses = 1.0, 9 accesses ~= 1.5, 99 accesses ~= 2.0
    normalized = math.log(access_count + 1) / math.log(10)
    return 1.0 + min(normalized, max_boost - 1.0)
