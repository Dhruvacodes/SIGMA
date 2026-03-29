"""
SIGMA Vector Store.
Uses ChromaDB Cloud for vector storage.
"""

from typing import Any

from config import settings


class SigmaVectorStore:
    """
    Vector store using ChromaDB Cloud for SIGMA's RAG pipeline.

    Collections:
    - historical_patterns: Signal outcome data
    - management_commentary: Earnings call excerpts
    - sector_context: Sector-level data
    """

    def __init__(self, persist_directory: str | None = None, use_cloud: bool | None = None):
        """
        Initialize the vector store.

        Args:
            persist_directory: Directory to persist ChromaDB data (local fallback).
            use_cloud: Whether to use ChromaDB Cloud. Defaults to settings.USE_CHROMA_CLOUD.
        """
        try:
            import chromadb
        except ImportError:
            raise ImportError("chromadb package required. Install with: pip install chromadb")

        use_cloud = use_cloud if use_cloud is not None else settings.USE_CHROMA_CLOUD

        if use_cloud and settings.CHROMA_API_KEY:
            # Use ChromaDB Cloud
            self.client = chromadb.CloudClient(
                api_key=settings.CHROMA_API_KEY,
                tenant=settings.CHROMA_TENANT,
                database=settings.CHROMA_DATABASE,
            )
            self._is_cloud = True
        else:
            # Fallback to local persistent client
            persist_dir = persist_directory or settings.CHROMA_PERSIST_DIR
            self.client = chromadb.PersistentClient(path=persist_dir)
            self._is_cloud = False

        # Create or get collections
        self.collections = {
            "historical_patterns": self.client.get_or_create_collection(
                name="historical_patterns",
                metadata={"description": "Signal outcome data"},
            ),
            "management_commentary": self.client.get_or_create_collection(
                name="management_commentary",
                metadata={"description": "Earnings call excerpts"},
            ),
            "sector_context": self.client.get_or_create_collection(
                name="sector_context",
                metadata={"description": "Sector-level data"},
            ),
        }

    def upsert_document(
        self, collection_name: str, doc_id: str, text: str, metadata: dict
    ) -> None:
        """
        Upsert a document into a collection.

        Args:
            collection_name: Name of the collection.
            doc_id: Unique document ID.
            text: Document text to embed and store.
            metadata: Metadata dict (must include ticker, signal_type, date, source).
        """
        if collection_name not in self.collections:
            raise ValueError(f"Unknown collection: {collection_name}")

        from rag.embeddings import embed_text_list

        embedding = embed_text_list(text)

        # Ensure required metadata fields
        required_fields = ["ticker", "signal_type", "date", "source"]
        for field in required_fields:
            if field not in metadata:
                metadata[field] = "unknown"

        # Convert any non-string values to strings (ChromaDB requirement)
        clean_metadata = {}
        for k, v in metadata.items():
            if isinstance(v, bool):
                clean_metadata[k] = str(v).lower()
            elif isinstance(v, (int, float)):
                clean_metadata[k] = v  # ChromaDB supports numeric types
            elif isinstance(v, list):
                clean_metadata[k] = ",".join(str(x) for x in v)
            else:
                clean_metadata[k] = str(v) if v is not None else ""

        self.collections[collection_name].upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[clean_metadata],
        )

    def query(
        self,
        collection_name: str,
        query_text: str,
        n_results: int = 5,
        where: dict | None = None,
    ) -> list[dict[str, Any]]:
        """
        Query a collection for similar documents.

        Args:
            collection_name: Name of the collection to query.
            query_text: Query text to find similar documents.
            n_results: Number of results to return.
            where: Optional metadata filter.

        Returns:
            List of {id, text, metadata, distance} dicts.
        """
        if collection_name not in self.collections:
            raise ValueError(f"Unknown collection: {collection_name}")

        from rag.embeddings import embed_text_list

        query_embedding = embed_text_list(query_text)

        # Build query kwargs
        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
        }
        if where:
            query_kwargs["where"] = where

        results = self.collections[collection_name].query(**query_kwargs)

        # Format results
        formatted = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                formatted.append(
                    {
                        "id": doc_id,
                        "text": results["documents"][0][i] if results["documents"] else "",
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "distance": results["distances"][0][i] if results["distances"] else 0.0,
                    }
                )

        return formatted

    def get_collection_count(self, collection_name: str) -> int:
        """Get the number of documents in a collection."""
        if collection_name not in self.collections:
            raise ValueError(f"Unknown collection: {collection_name}")
        return self.collections[collection_name].count()

    def is_empty(self, collection_name: str) -> bool:
        """Check if a collection is empty."""
        return self.get_collection_count(collection_name) == 0
