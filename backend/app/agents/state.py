"""
LexFind LangGraph State Definition.
Provides the TypedDict for passing state between graph nodes.
"""
from typing import Optional, TypedDict


class LexFindState(TypedDict):
    session_id: str
    user_id: str
    question: str
    history: list[dict]
    explicit_mode: Optional[str]
    document_ids: list[str]
    
    is_legal: bool
    intent: str
    
    retrieved_chunks: list[dict]
    citations: list[dict]
    
    answer: str
    error: Optional[str]
