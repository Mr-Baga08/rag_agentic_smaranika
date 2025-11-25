"""
Shared Context Manager - Bridge between RAG and Agent Tools

This module provides a shared context that allows agent tools to access
files uploaded to the RAG system and share data between tools.
"""

import os
from typing import Dict, Any, Optional, List
from pathlib import Path
import threading


class SharedContext:
    """
    Singleton class to manage shared context between RAG and agent tools.
    Provides file registry, dataframe cache, and RAG context.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # File registry: maps friendly names to actual file paths
        self.file_registry: Dict[str, str] = {}

        # Dataframe cache: shared between data analysis and visualization tools
        self.dataframes: Dict[str, Any] = {}

        # Current active dataframe name
        self.current_df: str = "df"

        # RAG context: stores recent RAG query results for context
        self.rag_context: List[Dict[str, Any]] = []

        # Upload directories to search
        self.upload_dirs = [
            "/mnt/Local_D/thetruthschool_agentic/the_truth_school_rag/uploads",
            "/mnt/Local_D/thetruthschool_agentic/the_truth_school_rag/storage",
        ]

        self._initialized = True

    def register_file(self, friendly_name: str, file_path: str) -> None:
        """Register a file with a friendly name"""
        self.file_registry[friendly_name.lower()] = file_path

    def get_file_path(self, name: str) -> Optional[str]:
        """Get file path by friendly name or search in uploads"""
        name_lower = name.lower()

        # Check registry first
        if name_lower in self.file_registry:
            return self.file_registry[name_lower]

        # Search in upload directories
        for upload_dir in self.upload_dirs:
            if not os.path.exists(upload_dir):
                continue

            # Search recursively
            for root, dirs, files in os.walk(upload_dir):
                for file in files:
                    # Match by filename (case-insensitive)
                    if file.lower() == name_lower or file.lower().startswith(name_lower):
                        full_path = os.path.join(root, file)
                        # Auto-register for future use
                        self.register_file(name, full_path)
                        return full_path

        return None

    def search_files(self, pattern: str) -> List[str]:
        """Search for files matching a pattern"""
        import fnmatch
        matches = []

        for upload_dir in self.upload_dirs:
            if not os.path.exists(upload_dir):
                continue

            for root, dirs, files in os.walk(upload_dir):
                for file in files:
                    if fnmatch.fnmatch(file.lower(), pattern.lower()):
                        matches.append(os.path.join(root, file))

        return matches

    def store_dataframe(self, name: str, df: Any) -> None:
        """Store a dataframe in shared cache"""
        self.dataframes[name] = df
        self.current_df = name

    def get_dataframe(self, name: Optional[str] = None) -> Optional[Any]:
        """Get a dataframe from shared cache"""
        df_name = name or self.current_df
        return self.dataframes.get(df_name)

    def list_dataframes(self) -> List[str]:
        """List all available dataframes"""
        return list(self.dataframes.keys())

    def add_rag_context(self, query: str, result: str, metadata: Optional[Dict] = None) -> None:
        """Add RAG query result to context"""
        context_entry = {
            "query": query,
            "result": result,
            "metadata": metadata or {},
        }
        self.rag_context.append(context_entry)

        # Keep only last 5 RAG contexts
        if len(self.rag_context) > 5:
            self.rag_context.pop(0)

    def get_recent_rag_context(self, limit: int = 3) -> List[Dict[str, Any]]:
        """Get recent RAG context"""
        return self.rag_context[-limit:]

    def clear_rag_context(self) -> None:
        """Clear RAG context"""
        self.rag_context.clear()

    def auto_discover_files(self) -> Dict[str, str]:
        """Auto-discover all files in upload directories"""
        discovered = {}

        for upload_dir in self.upload_dirs:
            if not os.path.exists(upload_dir):
                continue

            for root, dirs, files in os.walk(upload_dir):
                for file in files:
                    # Register by filename (without UUID prefix if present)
                    clean_name = file
                    if '_' in file:
                        # Remove UUID prefix like "ddffd7ea-3e73-400b-b967-12fad3fb9571_"
                        parts = file.split('_', 1)
                        if len(parts) > 1 and len(parts[0]) == 36:  # UUID length
                            clean_name = parts[1]

                    full_path = os.path.join(root, file)
                    discovered[clean_name.lower()] = full_path
                    self.register_file(clean_name, full_path)

        return discovered

    def get_context_summary(self) -> str:
        """Get a summary of current context"""
        summary = []

        summary.append("### 📁 Registered Files")
        if self.file_registry:
            for name, path in self.file_registry.items():
                file_size = os.path.getsize(path) if os.path.exists(path) else 0
                summary.append(f"- **{name}**: `{path}` ({file_size:,} bytes)")
        else:
            summary.append("- No files registered")

        summary.append("\n### 📊 Cached Dataframes")
        if self.dataframes:
            for name, df in self.dataframes.items():
                active = " *(active)*" if name == self.current_df else ""
                try:
                    shape = df.shape
                    summary.append(f"- **{name}**{active}: {shape[0]} rows × {shape[1]} columns")
                except:
                    summary.append(f"- **{name}**{active}")
        else:
            summary.append("- No dataframes loaded")

        summary.append("\n### 🔍 Recent RAG Context")
        if self.rag_context:
            for i, ctx in enumerate(self.rag_context[-3:], 1):
                summary.append(f"{i}. Query: *{ctx['query'][:50]}...*")
        else:
            summary.append("- No RAG context available")

        return "\n".join(summary)


# Global singleton instance
_shared_context = SharedContext()


def get_shared_context() -> SharedContext:
    """Get the global shared context instance"""
    return _shared_context
