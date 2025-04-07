import src.langgraph_slack.patch_typing  # must run before any Pydantic model loading
import json
import base64
from datetime import datetime, timezone
from google.cloud import bigquery
from typing import Any, Optional, Union, Literal, Tuple, List, Dict, Iterable
from typing_extensions import TypedDict
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

from langchain_core.documents import Document
from langchain_google_community.bq_storage_vectorstores.bigquery import BigQueryVectorStore
from langgraph.store.base import BaseStore, Item, SearchItem


class BigQueryMemoryStore(BigQueryVectorStore, BaseStore):
    __pydantic_config__ = {'arbitrary_types_allowed': True}
    __pydantic_model__ = None
    def __init__(self, project_id: str, dataset_name: str, table_name: str, location: str, embedding, credentials, schema: list[bigquery.SchemaField]):
        super().__init__(
            project_id=project_id,
            dataset_name=dataset_name,
            table_name=table_name,
            location=location,
            embedding=embedding,
            credentials=credentials,
            extra_fields={field.name: field.field_type for field in schema if field.name not in ['doc_id', 'content', 'embedding']}
        )
        self.schema = schema
        self._ensure_table_exists()

    def _ensure_table_exists(self):
        client = bigquery.Client(credentials=self.credentials, project=self.project_id, location=self.location)
        dataset_ref = client.dataset(self.dataset_name)
        table_ref = dataset_ref.table(self.table_name)
        try:
            client.get_table(table_ref)
        except NotFound:
            table = bigquery.Table(table_ref, schema=self.schema)
            client.create_table(table)

    #--------------------------------------------------------------------------------
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
        # Generate embedding for the content
        content = value.get("content", "")
        embedding_vector = self.embedding.embed(content)
        value["embedding"] = embedding_vector

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
        metadata = doc.metadata
        namespace_tuple = tuple(metadata.get("namespace", "").split("."))
        user_timestamp = metadata.get("user_timestamp")

        # Parse user_timestamp safely
        if isinstance(user_timestamp, str):
            try:
                user_timestamp = datetime.fromisoformat(user_timestamp.replace("Z", "+00:00"))
            except ValueError:
                user_timestamp = None

        return Item(
            namespace=namespace_tuple,
            key=key,
            value=metadata,
            created_at=user_timestamp,
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

        # Generate embedding for the query
        query_embedding = self.embedding.embed(query)

        # Perform similarity search using the query embedding
        docs = await self.asimilarity_search_by_vector(query_embedding, k=limit + offset)
        results = []
        for doc in docs[offset:]:
            metadata = doc.metadata
            namespace_tuple = tuple(metadata.get("namespace", "").split("."))
            doc_id = metadata.get("doc_id", "")
            user_timestamp = metadata.get("user_timestamp")

            # Parse user_timestamp safely
            if isinstance(user_timestamp, str):
                try:
                    user_timestamp = datetime.fromisoformat(user_timestamp.replace("Z", "+00:00"))
                except ValueError:
                    user_timestamp = None

            results.append(SearchItem(
                namespace=namespace_tuple,
                key=doc_id,
                score=None,  # Populate with actual similarity score if available
                value=metadata,
                created_at=user_timestamp,
                updated_at=None
            ))
        return results

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
def initialize_vector_store(table_name: str, schema: List[bigquery.SchemaField]) -> BigQueryMemoryStore:
    return BigQueryMemoryStore(
        project_id=PROJECT_ID,
        dataset_name=DATASET_ID,
        table_name=table_name,
        location=LOCATION,
        embedding=embedding,
        credentials=CREDENTIALS,
        schema=schema
    )

#----------------------SCHEMAS------------------------------------------------
# Define the desired schemas for each table
SCHEMAS = {
    SEMANTIC_TABLE: [
        bigquery.SchemaField(
        'doc_id', 'STRING', mode='REQUIRED',
        description='A unique identifier for the knowledge entry.'
    ),
    bigquery.SchemaField(
        'fact', 'JSON', mode='NULLABLE',
        description='A JSON object representing the fact or piece of knowledge.'
    ),
    bigquery.SchemaField(
        'embedding', 'FLOAT', mode='REPEATED',
        description='A vector representation of the fact, aiding in semantic searches.'
    ),
    bigquery.SchemaField(
        'namespace', 'STRING', mode='NULLABLE',
        description='A category or domain to which the fact belongs.'
    ),
    bigquery.SchemaField(
        'stored_at', 'TIMESTAMP', mode='NULLABLE', default_value_expression='CURRENT_TIMESTAMP()',
        description='The timestamp when the fact was stored.'
    ),
    bigquery.SchemaField(
        'source', 'STRING', mode='NULLABLE',
        description='The origin of the fact, which could be a document, user input, or external knowledge base.'
    ),
    ],
    EPISODIC_TABLE: [
    bigquery.SchemaField(
        'doc_id', 'STRING', mode='REQUIRED',
        description='A unique identifier for the memory entry.'
    ),
    bigquery.SchemaField(
        'event_details', 'JSON', mode='NULLABLE',
        description='A JSON object containing detailed information about the event, such as participants, location, and actions.'
    ),
    bigquery.SchemaField(
        'embedding', 'FLOAT', mode='REPEATED',
        description='A vector representation of the memory, useful for similarity searches.'
    ),
    bigquery.SchemaField(
        'namespace', 'STRING', mode='NULLABLE',
        description='A category or domain to which the memory belongs, aiding in organization and retrieval.'
    ), 
    bigquery.SchemaField(
        'user_timestamp', 'TIMESTAMP', mode='NULLABLE',
        description='The timestamp when the user entered her prompt.'
    ),
    bigquery.SchemaField(
        'stored_at', 'TIMESTAMP', mode='NULLABLE', default_value_expression='CURRENT_TIMESTAMP()',
        description='The timestamp when the memory was stored.'
    ),
    bigquery.SchemaField(
        'user_id', 'STRING', mode='NULLABLE',
        description='The identifier of the user associated with this memory.'
    ),
    bigquery.SchemaField(
        'channel_id', 'STRING', mode='NULLABLE',
        description='The identifier of the communication channel where the event occurred.'
    ),

    ],
    PROCEDURAL_TABLE: [
        bigquery.SchemaField(
        'doc_id', 'STRING', mode='REQUIRED',
        description='A unique identifier for the procedural rule or behavior.'
    ),
    bigquery.SchemaField(
        'procedure', 'JSON', mode='NULLABLE',
        description='A JSON object detailing the procedure or behavior, including conditions and actions.'
    ),
    bigquery.SchemaField(
        'embedding', 'FLOAT', mode='REPEATED',
        description='A vector representation of the procedure, facilitating similarity assessments.'
    ),
    bigquery.SchemaField(
        'namespace', 'STRING', mode='NULLABLE',
        description='A category or domain to which the procedure belongs.'
    ),
    bigquery.SchemaField(
        'stored_at', 'TIMESTAMP', mode='NULLABLE', default_value_expression='CURRENT_TIMESTAMP()',
        description='The timestamp when the procedure was stored.'
    ),
    bigquery.SchemaField(
        'version', 'STRING', mode='NULLABLE',
        description='The version of the procedure, useful for tracking updates and changes.'
    ),
]

}

#---------------------------------get memory tools--------------------------------
def get_memory_tools() -> List:
    semantic_memory_store = initialize_vector_store(SEMANTIC_TABLE, SCHEMAS[SEMANTIC_TABLE])
    episodic_memory_store = initialize_vector_store(EPISODIC_TABLE, SCHEMAS[EPISODIC_TABLE])
    procedural_memory_store = initialize_vector_store(PROCEDURAL_TABLE, SCHEMAS[PROCEDURAL_TABLE])

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

    return [
        semantic_manage_tool, semantic_search_tool,
        episodic_manage_tool, episodic_search_tool,
        procedural_manage_tool, procedural_search_tool
    ]