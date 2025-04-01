import json
import base64
from datetime import datetime, timezone
from google.cloud import bigquery
from typing import Iterable, Tuple
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_community.bq_storage_vectorstores.bigquery import BigQueryVectorStore
from langmem import create_manage_memory_tool, create_search_memory_tool
from src.langgraph_slack.config import (
    PROJECT_ID, DATASET_ID, LOCATION,
    SEMANTIC_TABLE, EPISODIC_TABLE, PROCEDURAL_TABLE,
    CREDENTIALS
)

# Initialize the embedding model
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize BigQuery client
bq_client = bigquery.Client(credentials=CREDENTIALS, project=PROJECT_ID, location=LOCATION)
dataset_ref = bigquery.DatasetReference(PROJECT_ID, DATASET_ID)
dataset = bigquery.Dataset(dataset_ref)
dataset.location = LOCATION
bq_client.create_dataset(dataset, exists_ok=True)

from typing import Any, Optional, Union, Literal, Tuple, List, Dict
from langchain_core.documents import Document
from langchain_google_community.bq_storage_vectorstores.bigquery import BigQueryVectorStore
from langgraph.store.base import BaseStore, Item, SearchItem


class BigQueryMemoryStore(BigQueryVectorStore, BaseStore):
    async def aput(
        self,
        namespace: Tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: Optional[Union[Literal[False], list[str]]] = None,
        *,
        ttl: Optional[float] = None,
    ) -> None:
        value = dict(value)  # Ensure mutable

        now = datetime.now(timezone.utc)
        value.setdefault("user_timestamp", now)  # Allow override if upstream provides it
        value["namespace"] = ".".join(namespace)
        value["doc_id"] = key

        document = Document(page_content=value.get("content", ""), metadata=value)
        await self.aadd_documents([document])

    async def aget(
        self,
        namespace: Tuple[str, ...],
        key: str,
        *,
        refresh_ttl: Optional[bool] = None,
    ) -> Optional[Item]:
        docs = await self.aget_by_ids([key])
        if not docs:
            return None

        doc = docs[0]
        return Item(
            namespace=tuple(doc.metadata.get("namespace", "").split(".")),
            key=key,
            value=doc.metadata,
            created_at=None,
            updated_at=None,
        )

    async def adelete(
        self,
        namespace: Tuple[str, ...],
        key: str,
    ) -> None:
        await self.adelete(ids=[key])

    async def asearch(
        self,
        namespace_prefix: Tuple[str, ...],
        *,
        query: Optional[str] = None,
        filter: Optional[dict[str, Any]] = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: Optional[bool] = None,
    ) -> List[SearchItem]:
        if query is None:
            return []

        docs = await self.asimilarity_search(query, k=limit)
        return [
            SearchItem(
                namespace=tuple(doc.metadata.get("namespace", "").split(".")),
                key=doc.metadata.get("doc_id", ""),
                score=None,
                value=doc.metadata,
            )
            for doc in docs
        ]

    async def abatch(
        self,
        operations: Iterable[Tuple[str, Tuple[str, ...], str, Optional[dict]]],
    ) -> None:
        for op, namespace, key, value in operations:
            if op == "put":
                await self.aput(namespace, key, value or {})
            elif op == "delete":
                await self.adelete(namespace, key)

    def batch(
        self,
        operations: Iterable[Tuple[str, Tuple[str, ...], str, Optional[dict]]],
    ) -> None:
        import asyncio
        asyncio.run(self.abatch(operations))

# Function to initialize a BigQueryMemoryStore
def initialize_vector_store(table_name: str) -> BigQueryMemoryStore:
    return BigQueryMemoryStore(
        project_id=PROJECT_ID,
        dataset_name=DATASET_ID,
        table_name=table_name,
        location=LOCATION,
        embedding=embedding,
        credentials=CREDENTIALS
    )

# Initialize vector stores for each memory type
semantic_memory_store = initialize_vector_store(SEMANTIC_TABLE)
episodic_memory_store = initialize_vector_store(EPISODIC_TABLE)
procedural_memory_store = initialize_vector_store(PROCEDURAL_TABLE)

# Create memory tools with explicit names for registration in langgraph.json
semantic_manage_tool = create_manage_memory_tool(
    namespace=("semantic_memories", "{langgraph_auth_user_id}"),
    store=semantic_memory_store,
    name="manage_semantic_memory"
)

episodic_manage_tool = create_manage_memory_tool(
    namespace=("episodic_memories", "{langgraph_auth_user_id}"),
    store=episodic_memory_store,
    name="manage_episodic_memory"
)

procedural_manage_tool = create_manage_memory_tool(
    namespace=("procedural_memories", "{langgraph_auth_user_id}"),
    store=procedural_memory_store,
    name="manage_procedural_memory"
)

semantic_search_tool = create_search_memory_tool(
    namespace=("semantic_memories", "{langgraph_auth_user_id}"),
    store=semantic_memory_store,
    name="search_semantic_memory"
)

episodic_search_tool = create_search_memory_tool(
    namespace=("episodic_memories", "{langgraph_auth_user_id}"),
    store=episodic_memory_store,
    name="search_episodic_memory"
)

procedural_search_tool = create_search_memory_tool(
    namespace=("procedural_memories", "{langgraph_auth_user_id}"),
    store=procedural_memory_store,
    name="search_procedural_memory"
)

# Expose memory tools for use in agent.py
MEMORY_TOOLS = [
    semantic_manage_tool, semantic_search_tool,
    episodic_manage_tool, episodic_search_tool,
    procedural_manage_tool, procedural_search_tool
]