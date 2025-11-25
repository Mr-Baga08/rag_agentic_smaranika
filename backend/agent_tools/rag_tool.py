"""
RAG Tool - Wrapper for existing LightRAG system
"""

from langchain.tools import BaseTool
from pydantic import Field
from typing import Optional, Dict, Any
import traceback


class RAGTool(BaseTool):
    """
    Tool for querying the RAG system for domain-specific information.
    """

    name: str = "rag_query"
    description: str = """
    Query the Retrieval-Augmented Generation (RAG) system for information from uploaded documents.
    Useful for:
    - Answering questions from domain-specific documents
    - Finding information in medical, legal, financial, technical, or academic documents
    - Retrieving context and sources from the knowledge base
    - Getting verified, cited answers

    Input should be a natural language question about the documents.

    Available domains:
    - medical: Medical and healthcare documents
    - legal: Legal and compliance documents
    - financial: Financial reports and analysis
    - technical: Technical documentation and APIs
    - academic: Research papers and publications

    Example: "What are the symptoms of hypertension according to the medical documents?"
    """

    rag_instance: Any = Field(default=None, description="Reference to RAG instance")
    domain: str = Field(default="general", description="Current domain")
    llm_model: Any = Field(default=None, description="LLM model for generation")

    class Config:
        arbitrary_types_allowed = True

    def _run(self, query: str, domain: Optional[str] = None) -> str:
        """Query the RAG system"""
        try:
            if not self.rag_instance:
                return "❌ RAG system not initialized. Please upload documents first."

            # Use specified domain or default
            search_domain = domain if domain else self.domain

            # Query the RAG system
            # Note: This will be connected to the actual RAG instance
            # For now, return a placeholder that will be replaced with actual integration

            result = f"""🔍 **RAG Query Executed**

**Query:** {query}
**Domain:** {search_domain}

**Note:** This will be connected to the actual RAG system in the backend.
The RAG system will retrieve relevant context from your uploaded documents
and generate a comprehensive answer with sources.

Use the main /query endpoint for full RAG functionality.
"""

            return result

        except Exception as e:
            return f"❌ RAG query error: {str(e)}\n\n**Traceback:**\n```\n{traceback.format_exc()}\n```"

    async def _arun(self, query: str, domain: Optional[str] = None) -> str:
        """Async version"""
        return self._run(query, domain)

    def set_rag_instance(self, rag_instance: Any):
        """Set the RAG instance reference"""
        self.rag_instance = rag_instance

    def set_domain(self, domain: str):
        """Set the current domain"""
        self.domain = domain
