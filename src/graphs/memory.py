from __future__ import annotations
import src.langgraph_slack.patch_typing  # must run before any Pydantic model loading
import logging
import json
import asyncio
from datetime import datetime, timezone
from typing import Any, Type, Optional, Union, Literal, Tuple, List, Dict, Iterable
from typing_extensions import TypedDict

from pydantic import BaseModel
from google.cloud import bigquery
from google.cloud.exceptions import NotFound

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_community.bq_storage_vectorstores.bigquery import BigQueryVectorStore
from langgraph.store.base import Item, SearchItem
from langgraph.store.base.batch import AsyncBatchedBaseStore, Op, PutOp, GetOp, SearchOp, ListNamespacesOp, Result
from langmem import create_manage_memory_tool, create_search_memory_tool

from src.langgraph_slack.config import (
    PROJECT_ID, DATASET_ID, LOCATION,
    SEMANTIC_TABLE, EPISODIC_TABLE, PROCEDURAL_TABLE,
    CREDENTIALS
)

logger = logging.getLogger(__name__)

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

try:
    bq_client = bigquery.Client(credentials=CREDENTIALS, project=PROJECT_ID, location=LOCATION)
    logger.info("BigQuery client initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize BigQuery client: {e}")

CONTENT_FIELDS = {
    SEMANTIC_TABLE: "fact",
    EPISODIC_TABLE: "event_details",
    PROCEDURAL_TABLE: "procedure",
}

class Fact(BaseModel):
    content: str
    importance: Optional[str] = None
    category: Optional[str] = None

class Episode(BaseModel):
    observation: Optional[str] = None
    thoughts: Optional[str] = None
    action: Optional[str] = None
    result: Optional[str] = None

class Procedure(BaseModel):
    name: Optional[str] = None
    conditions: Optional[str] = None
    steps: Optional[str] = None
    notes: Optional[str] = None

PYDANTIC_MODELS = {
    "fact": Fact,
    "event_details": Episode,
    "procedure": Procedure,
}

SCHEMAS = {
    SEMANTIC_TABLE: [...],  # Redacted for brevity
    EPISODIC_TABLE: [...],
    PROCEDURAL_TABLE: [...],
}

class BigQueryMemoryStore(AsyncBatchedBaseStore):
    def __init__(
        self,
        vectorstore: BigQueryVectorStore,
        content_field: str,
        content_model: Optional[Type[BaseModel]] = None,
        schema: Optional[List[bigquery.SchemaField]] = None,
    ):
        super().__init__()
        self.vectorstore = vectorstore
        self.content_field = content_field
        self.content_model = content_model
        self.schema = schema

    @classmethod
    def from_client(
        cls,
        bq_client: bigquery.Client,
        dataset_name: str,
        table_name: str,
        embedding: Embeddings,
        content_field: str,
        content_model: Optional[Type[BaseModel]] = None,
        schema: Optional[List[bigquery.SchemaField]] = None,
        **kwargs,
    ) -> BigQueryMemoryStore:
        vectorstore = BigQueryVectorStore(
            embedding=embedding,
            project_id=bq_client.project,
            dataset_name=dataset_name,
            table_name=table_name,
            content_field=content_field,
            credentials=bq_client._credentials,
            location=bq_client.location,
            **kwargs,
        )
        object.__setattr__(vectorstore, "_bq_client", bq_client)
        return cls(
            vectorstore=vectorstore,
            content_field=content_field,
            content_model=content_model,
            schema=schema,
        )

    def _normalize_structured_field(self, raw: Any) -> str:
        if isinstance(raw, dict) and self.content_model:
            logger.debug(f"Raw content before wrapping: {raw}")
            return self.content_model(**raw)
        elif isinstance(raw, str) and self.content_model:
            return self.content_model(content=raw)
        elif isinstance(raw, self.content_model):
            return raw
        raise ValueError(
            f"{self.content_field} must be a dict, str, or {self.content_model.__name__} — got {type(raw)}"
        )

    async def aput(self, namespace: Tuple[str, ...], key: str, value: dict[str, Any], index: Optional[Union[Literal[False], list[str]]] = None, *, ttl: Optional[float] = None) -> None:
        logger.info(f"[aput] Inserting doc_id={key} into namespace={namespace}")
        
        value = dict(value)
        value["namespace"] = ".".join(namespace)
        value["doc_id"] = key

        # Get the raw content and ensure it's wrapped in the correct model (e.g., Fact)
        raw_content = value.get(self.content_field)
        
        if raw_content is None:
            logger.error(f"Content for {self.content_field} is None. This is not expected.")
            return

        # Ensure the content is wrapped into the content model (e.g., Fact)
        text = self._normalize_structured_field(raw_content)

        # Insert the document into BigQuery (wrapped content)
        doc = Document(page_content=text, metadata=value)
        self.vectorstore.add_documents([doc])

    async def aget(self, namespace: Tuple[str, ...], key: str, *, refresh_ttl: Optional[bool] = None) -> Optional[Item]:
        logger.info(f"[aget] Retrieving doc_id={key} from namespace={namespace}")
        docs = self.vectorstore.get_documents(ids=[key])
        if not docs:
            return None
        doc = docs[0]
        metadata = doc.metadata
        ns = tuple(metadata.get("namespace", "").split("."))
        content_val = metadata.get(self.content_field)
        if isinstance(content_val, str) and self.content_model:
            try:
                metadata[self.content_field] = self.content_model(**json.loads(content_val))
            except Exception:
                pass
        return Item(namespace=ns, key=key, value=metadata, created_at=None, updated_at=None)

    async def asearch(self, namespace_prefix: Tuple[str, ...], *, query: Optional[str] = None, filter: Optional[dict[str, Any]] = None, limit: int = 10, offset: int = 0, refresh_ttl: Optional[bool] = None) -> List[SearchItem]:
        logger.info(f"[asearch] Searching in namespace_prefix={namespace_prefix} for query='{query}'")
        if not query:
            return []
        results = self.vectorstore.similarity_search_with_score(
            query=query,
            filter=filter,
            k=limit + offset
        )[offset:]
        items = []
        for doc, score in results:
            ns = tuple(doc.metadata.get("namespace", "").split("."))
            key = doc.metadata.get("doc_id", "")
            items.append(SearchItem(namespace=ns, key=key, value=doc.metadata, score=score))
        return items

    async def adelete(self, namespace: Tuple[str, ...], key: str) -> None:
        logger.info(f"[adelete] Deleting doc_id={key} from namespace={namespace}")
        await self.vectorstore.adelete(ids=[key])

    def mget(self, keys: Sequence[str]) -> List[Optional[dict]]:
        return self.vectorstore.get_documents(ids=keys)
    
    async def amget(self, keys: Sequence[str]) -> List[Optional[dict]]:
        return await self.vectorstore.get_documents(ids=keys)
    
    def mset(self, key_value_pairs: Sequence[Tuple[str, dict]]) -> None:
        documents = [
            Document(page_content=value.get(self.content_field), metadata={**value, "doc_id": key})
            for key, value in key_value_pairs
        ]
        self.vectorstore.add_documents(documents)

    async def amset(self, key_value_pairs: Sequence[Tuple[str, dict]]) -> None:
        documents = [
            Document(page_content=value.get(self.content_field), metadata={**value, "doc_id": key})
            for key, value in key_value_pairs
        ]
        await self.vectorstore.add_documents(documents)

    def mdelete(self, keys: Sequence[str]) -> None:
        self.vectorstore.adelete(ids=keys)

    async def amdelete(self, keys: Sequence[str]) -> None:
        await self.vectorstore.adelete(ids=keys)

    def yield_keys(self, prefix: Optional[str] = None) -> Iterator[str]:
        return self.vectorstore.yield_keys(prefix=prefix)
    
    async def ayield_keys(self, prefix: Optional[str] = None) -> AsyncIterator[str]:
        async for key in self.vectorstore.yield_keys(prefix=prefix):
            yield key
    
    async def abatch(self, operations: Iterable[Op]) -> List[Result]:
        logger.info(f"[abatch] Executing {len(list(operations))} batch operations")
        results = []
        for op in operations:
            if isinstance(op, PutOp):
                await self.aput(op.namespace, op.key, op.value, index=op.index, ttl=op.ttl)
                results.append(None)
            elif isinstance(op, GetOp):
                item = await self.aget(op.namespace, op.key, refresh_ttl=op.refresh_ttl)
                results.append(item)
            elif isinstance(op, SearchOp):
                items = await self.asearch(
                    op.namespace_prefix,
                    query=op.query,
                    filter=op.filter,
                    limit=op.limit,
                    offset=op.offset,
                    refresh_ttl=op.refresh_ttl,
                )
                results.append(items)
            elif isinstance(op, ListNamespacesOp):
                results.append([])
            else:
                raise NotImplementedError(f"Unsupported op: {op}")
        return results


CONTENT_FIELDS = {
    SEMANTIC_TABLE: "fact",
    EPISODIC_TABLE: "event_details",
    PROCEDURAL_TABLE: "procedure",
}



#----------------------SCHEMAS------------------------------------------------
# Define the desired schemas for each table
# Define the desired schemas for each table
SCHEMAS = {
    SEMANTIC_TABLE: [
        bigquery.SchemaField("doc_id", "STRING", mode="REQUIRED", description="Unique identifier for the semantic memory."),
        bigquery.SchemaField("fact", "RECORD", mode="REQUIRED", fields=[
            bigquery.SchemaField("content", "STRING", description="The factual statement or knowledge."),
            bigquery.SchemaField("importance", "STRING", description="Importance level e.g., HIGH, MEDIUM, LOW."),
            bigquery.SchemaField("category", "STRING", description="Type of fact e.g., BACKGROUND, HOBBY, SKILL."),
        ], description="Structured representation of factual knowledge."),
        bigquery.SchemaField("source", "STRING", mode="NULLABLE", description="Where the fact came from (user input, document, etc)."),
        bigquery.SchemaField("namespace", "STRING", mode="NULLABLE", description="Namespace for logical grouping."),
        bigquery.SchemaField("embedding", "FLOAT", mode="REPEATED", description="Embedding for vector similarity."),
        bigquery.SchemaField("stored_at", "TIMESTAMP", mode="NULLABLE", default_value_expression='CURRENT_TIMESTAMP()', description="When the record was stored."),
    ],
    EPISODIC_TABLE: [
        bigquery.SchemaField("doc_id", "STRING", mode="REQUIRED", description="Unique identifier for the episode."),
        bigquery.SchemaField("episode", "RECORD", mode="REQUIRED", fields=[
            bigquery.SchemaField("observation", "STRING", description="Context or setup of the interaction."),
            bigquery.SchemaField("thoughts", "STRING", description="Reasoning or internal monologue."),
            bigquery.SchemaField("action", "STRING", description="What was done in response."),
            bigquery.SchemaField("result", "STRING", description="Outcome and what made it successful."),
        ], description="Structured record of a specific past interaction."),
        bigquery.SchemaField("participants", "STRING", mode="REPEATED", description="Users or agents involved in the episode."),
        bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED", description="When the episode occurred."),
        bigquery.SchemaField("namespace", "STRING", mode="NULLABLE", description="Namespace for logical grouping."),
        bigquery.SchemaField("embedding", "FLOAT", mode="REPEATED", description="Vector embedding for similarity search."),
        bigquery.SchemaField("stored_at", "TIMESTAMP", mode="NULLABLE", default_value_expression='CURRENT_TIMESTAMP()', description="When the record was stored."),
    ],
    PROCEDURAL_TABLE: [
        bigquery.SchemaField("doc_id", "STRING", mode="REQUIRED", description="Unique identifier for the procedure."),
        bigquery.SchemaField("procedure", "RECORD", mode="REQUIRED", fields=[
            bigquery.SchemaField("name", "STRING", description="Name or title of the procedure."),
            bigquery.SchemaField("conditions", "STRING", description="When or under what conditions to execute."),
            bigquery.SchemaField("steps", "STRING", description="Step-by-step instructions."),
            bigquery.SchemaField("notes", "STRING", description="Additional notes or guidelines."),
        ], description="Structured behavioral rule or policy."),
        bigquery.SchemaField("version", "STRING", mode="NULLABLE", description="Version identifier for procedural logic."),
        bigquery.SchemaField("namespace", "STRING", mode="NULLABLE", description="Namespace for logical grouping."),
        bigquery.SchemaField("embedding", "FLOAT", mode="REPEATED", description="Embedding for vector-based retrieval."),
        bigquery.SchemaField("stored_at", "TIMESTAMP", mode="NULLABLE", default_value_expression='CURRENT_TIMESTAMP()', description="When the record was stored."),
]
}

#---------------------------------get memory tools--------------------------------
async def get_memory_tools() -> List:
    semantic_memory_store = BigQueryMemoryStore.from_client(
        bq_client=bq_client,
        dataset_name=DATASET_ID, 
        table_name=SEMANTIC_TABLE, 
        embedding=embedding,
        content_field=CONTENT_FIELDS[SEMANTIC_TABLE],
        content_model=PYDANTIC_MODELS[CONTENT_FIELDS[SEMANTIC_TABLE]],
    )
    

    episodic_memory_store = BigQueryMemoryStore.from_client(
        bq_client=bq_client,
        dataset_name=DATASET_ID, 
        table_name=EPISODIC_TABLE, 
        embedding=embedding,
        content_field=CONTENT_FIELDS[EPISODIC_TABLE],
        content_model=PYDANTIC_MODELS[CONTENT_FIELDS[EPISODIC_TABLE]],
    )
    

    procedural_memory_store = BigQueryMemoryStore.from_client(
        bq_client=bq_client,
        dataset_name=DATASET_ID, 
        table_name=PROCEDURAL_TABLE,  
        embedding=embedding,
        content_field=CONTENT_FIELDS[PROCEDURAL_TABLE],
        content_model=PYDANTIC_MODELS[CONTENT_FIELDS[PROCEDURAL_TABLE]],
    )
    
    semantic_manage_tool = create_manage_memory_tool(
        namespace=("semantic_memories", "{langgraph_auth_user_id}"),
        store=semantic_memory_store,
        name="manage_semantic_memory",
        schema=Fact
    )
    episodic_manage_tool = create_manage_memory_tool(
        namespace=("episodic_memories", "{langgraph_auth_user_id}"),
        store=episodic_memory_store,
        name="manage_episodic_memory",
        schema=Episode
    )
    procedural_manage_tool = create_manage_memory_tool(
        namespace=("procedural_memories", "{langgraph_auth_user_id}"),
        store=procedural_memory_store,
        name="manage_procedural_memory",
        schema=Procedure
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

    return [
        semantic_manage_tool, semantic_search_tool,
        episodic_manage_tool, episodic_search_tool,
        procedural_manage_tool, procedural_search_tool
    ]