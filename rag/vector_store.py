"""
SIGMA Vector Store.
Lightweight in-memory implementation for serverless deployment.
Optionally uses ChromaDB Cloud if available.
"""

from typing import Any

from config import settings


class SigmaVectorStore:
    """
    Vector store for SIGMA's RAG pipeline.
    Uses in-memory storage for serverless, ChromaDB Cloud if configured.

    Collections:
    - historical_patterns: Signal outcome data
    - management_commentary: Earnings call excerpts
    - sector_context: Sector-level data
    """

    def __init__(self, persist_directory: str | None = None, use_cloud: bool | None = None):
        """
        Initialize the vector store.

        Args:
            persist_directory: Directory to persist data (ignored in serverless mode).
            use_cloud: Whether to use ChromaDB Cloud.
        """
        self._chromadb_client = None
        self._is_cloud = False
        self._use_memory = True

        # In-memory fallback storage
        self._memory_store: dict[str, list[dict]] = {
            "historical_patterns": [],
            "management_commentary": [],
            "sector_context": [],
        }

        # Try ChromaDB Cloud if configured
        use_cloud = use_cloud if use_cloud is not None else settings.USE_CHROMA_CLOUD
        if use_cloud and settings.CHROMA_API_KEY:
            try:
                import chromadb
                self._chromadb_client = chromadb.CloudClient(
                    api_key=settings.CHROMA_API_KEY,
                    tenant=settings.CHROMA_TENANT,
                    database=settings.CHROMA_DATABASE,
                )
                self._is_cloud = True
                self._use_memory = False
                self.collections = {
                    "historical_patterns": self._chromadb_client.get_or_create_collection(
                        name="historical_patterns",
                        metadata={"description": "Signal outcome data"},
                    ),
                    "management_commentary": self._chromadb_client.get_or_create_collection(
                        name="management_commentary",
                        metadata={"description": "Earnings call excerpts"},
                    ),
                    "sector_context": self._chromadb_client.get_or_create_collection(
                        name="sector_context",
                        metadata={"description": "Sector-level data"},
                    ),
                }
            except ImportError:
                pass  # Fall back to in-memory

    def upsert_document(
        self, collection_name: str, doc_id: str, text: str, metadata: dict
    ) -> None:
        """
        Upsert a document into a collection.

        Args:
            collection_name: Name of the collection.
            doc_id: Unique document ID.
            text: Document text to embed and store.
            metadata: Metadata dict.
        """
        if collection_name not in self._memory_store:
            raise ValueError(f"Unknown collection: {collection_name}")

        # Ensure required metadata fields
        required_fields = ["ticker", "signal_type", "date", "source"]
        for field in required_fields:
            if field not in metadata:
                metadata[field] = "unknown"

        if self._use_memory:
            # In-memory storage
            doc = {
                "id": doc_id,
                "text": text,
                "metadata": metadata,
            }
            # Update or insert
            existing = [i for i, d in enumerate(self._memory_store[collection_name]) if d["id"] == doc_id]
            if existing:
                self._memory_store[collection_name][existing[0]] = doc
            else:
                self._memory_store[collection_name].append(doc)
        else:
            # ChromaDB Cloud
            from rag.embeddings import embed_text_list
            embedding = embed_text_list(text)

            clean_metadata = {}
            for k, v in metadata.items():
                if isinstance(v, bool):
                    clean_metadata[k] = str(v).lower()
                elif isinstance(v, (int, float)):
                    clean_metadata[k] = v
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
        if collection_name not in self._memory_store:
            raise ValueError(f"Unknown collection: {collection_name}")

        if self._use_memory:
            # Simple keyword-based search for in-memory mode
            results = []
            query_lower = query_text.lower()
            query_words = set(query_lower.split())

            for doc in self._memory_store[collection_name]:
                # Apply metadata filter
                if where:
                    match = True
                    for k, v in where.items():
                        if doc["metadata"].get(k) != v:
                            match = False
                            break
                    if not match:
                        continue

                # Score by keyword overlap
                doc_text_lower = doc["text"].lower()
                doc_words = set(doc_text_lower.split())
                overlap = len(query_words & doc_words)

                if overlap > 0 or not query_words:
                    results.append({
                        "id": doc["id"],
                        "text": doc["text"],
                        "metadata": doc["metadata"],
                        "distance": 1.0 / (1 + overlap),  # Lower is better
                    })

            # Sort by distance and limit
            results.sort(key=lambda x: x["distance"])
            return results[:n_results]
        else:
            # ChromaDB Cloud query
            from rag.embeddings import embed_text_list
            query_embedding = embed_text_list(query_text)

            query_kwargs = {
                "query_embeddings": [query_embedding],
                "n_results": n_results,
            }
            if where:
                query_kwargs["where"] = where

            results = self.collections[collection_name].query(**query_kwargs)

            formatted = []
            if results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    formatted.append({
                        "id": doc_id,
                        "text": results["documents"][0][i] if results["documents"] else "",
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "distance": results["distances"][0][i] if results["distances"] else 0.0,
                    })

            return formatted

    def get_collection_count(self, collection_name: str) -> int:
        """Get the number of documents in a collection."""
        if collection_name not in self._memory_store:
            raise ValueError(f"Unknown collection: {collection_name}")

        if self._use_memory:
            return len(self._memory_store[collection_name])
        else:
            return self.collections[collection_name].count()

    def is_empty(self, collection_name: str) -> bool:
        """Check if a collection is empty."""
        return self.get_collection_count(collection_name) == 0
