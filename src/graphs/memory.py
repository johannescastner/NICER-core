# src/graphs/memory.py
"""
This is where we define the BigQuery memory store,
which connects the langmem longrun memory system to BigQuery.
"""
from __future__ import annotations
from functools import lru_cache
import asyncio
import concurrent.futures
from collections.abc import Iterable as IterableABC
import uuid
import logging
import json
from datetime import datetime, timezone
from typing import (
    Any,
    Type,
    Optional,
    Union,
    Literal,
    Tuple,
    List,
    Dict,
    Iterable,
    Sequence,
    AsyncIterator,
    Iterator,
    overload
)
from pydantic import BaseModel
import google
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_google_community.bq_storage_vectorstores.bigquery import BigQueryVectorStore
from langchain_google_community.bq_storage_vectorstores.utils import validate_column_in_bq_schema
from langgraph.store.base import (
    BaseStore,
    Item,
    SearchItem,
    NamespacePath,
    NOT_PROVIDED,
    NotProvided,
    _validate_namespace,
)
from langgraph.store.base.batch import (
    AsyncBatchedBaseStore,
    Op,
    PutOp,
    GetOp,
    SearchOp,
    ListNamespacesOp,
    Result
)
from langmem import create_manage_memory_tool, create_search_memory_tool


import src.langgraph_slack.patch_typing  # must run before any Pydantic model loading

from src.langgraph_slack.config import (
    PROJECT_ID, DATASET_ID, LOCATION,
    SEMANTIC_TABLE, EPISODIC_TABLE, PROCEDURAL_TABLE,
    CREDENTIALS
)

NamespaceTemplate = Tuple[str, ...]

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class ModalEmbeddings(Embeddings):
    """
    LangChain-compatible embeddings wrapper that calls Modal endpoints.
    
    Replaces HuggingFaceEmbeddings to eliminate local model loading.
    Uses bge-base-en-v1.5 (768 dims) for best retrieval quality.
    """
    
    def __init__(self, model: str = "bge-base"):
        """
        Args:
            model: "bge-base" (768 dims, best quality) or "minilm" (384 dims, legacy)
        """
        self.model = model
        self._client = None
    
    def _get_client(self):
        """Lazy-load the inference client."""
        if self._client is None:
            from pro.ml_inference.client import get_inference_client
            self._client = get_inference_client()
        return self._client
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents (sync, for LangChain compatibility)."""
        import asyncio
        
        async def _embed_batch():
            client = self._get_client()
            results = await client.embed_batch(texts, model=self.model)
            return [r.embedding for r in results]
        
        # Handle running in existing event loop vs new loop
        try:
            loop = asyncio.get_running_loop()
            # Already in async context - run in thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, _embed_batch())
                return future.result()
        except RuntimeError:
            # No running loop - safe to create one
            return asyncio.run(_embed_batch())
    
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query (sync, for LangChain compatibility)."""
        import asyncio
        
        async def _embed():
            client = self._get_client()
            result = await client.embed(text, model=self.model)
            return result.embedding
        
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, _embed())
                return future.result()
        except RuntimeError:
            return asyncio.run(_embed())
    
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Async embed documents."""
        client = self._get_client()
        results = await client.embed_batch(texts, model=self.model)
        return [r.embedding for r in results]
    
    async def aembed_query(self, text: str) -> list[float]:
        """Async embed query."""
        client = self._get_client()
        result = await client.embed(text, model=self.model)
        return result.embedding


@lru_cache(maxsize=1)
def get_embedding():
    """
    Lazy embedding factory.
    
    Returns a Modal-based embeddings object that's compatible with LangChain
    but doesn't load any local models.
    
    Uses bge-base-en-v1.5 (768 dims) for 22% better retrieval than MiniLM.
    """
    return ModalEmbeddings(model="bge-base")

@lru_cache(maxsize=1)
def get_bq_client() -> bigquery.Client:
    """
    Lazy BigQuery client factory.
    Avoid doing API/client setup at import time.
    """
    try:
        client = bigquery.Client(credentials=CREDENTIALS, project=PROJECT_ID, location=LOCATION)
        logger.info("BigQuery client initialized successfully.")
        return client
    except Exception as e:
        logger.error("Failed to initialize BigQuery client: %s", e, exc_info=True)
        raise

CONTENT_FIELDS = {
    SEMANTIC_TABLE: "fact",
    EPISODIC_TABLE: "episode",
    PROCEDURAL_TABLE: "procedure",
}

def parse_fqn(fqn: str, default_project: str | None = None) -> Tuple[str, ...]:
    """
    Accepts 'dataset.table' or 'project.dataset.table'.
    Returns (project, dataset, table).
    """
    parts = fqn.split(".")
    if len(parts) == 2:
        if not default_project:
            raise ValueError("Default project required for <dataset>.<table> form")
        project, dataset, table = default_project, parts[0], parts[1]
    elif len(parts) == 3:
        project, dataset, table = parts
    else:
        raise ValueError("FQN must be <dataset>.<table> or <project>.<dataset>.<table>")
    return (project, dataset, table)

def ns_sem_dataset(project: str, dataset: str) -> Tuple[str, ...]:
    return ("metadata", project, dataset)

def ns_sem_table(project: str, dataset: str, table: str) -> Tuple[str, ...]:
    return ("metadata", project, dataset, table)

def ns_sem_column(project: str, dataset: str, table: str, column: str) -> Tuple[str, ...]:
    return ("metadata", project, dataset, table, column)

def ns_join(parts: Tuple[str, ...]) -> str:
    """Serialize for storage (STRING namespace column in BQ)."""
    return ".".join(parts)

def ns_with_bucket(ns: Tuple[str, ...], bucket: str, anchor: str = "episodic") -> Tuple[str, ...]:
    """
    Insert a priority bucket *right after* the 'anchor' segment (e.g., 'episodic').
    Returns a tuple, regardless of input.
    """
    try:
        i = ns.index(anchor)
        return (*ns[:i+1], bucket, *ns[i+1:])
    except ValueError:
        # If anchor isn't present, prepend it (keeps behavior predictable)
        return (anchor, bucket, *ns)

def ns_sem_for_target(target: str, default_project: str | None = PROJECT_ID) -> tuple[str, ...]:
    """
    Accepts: "dataset", "dataset.table", "dataset.table.column",
             or "project.dataset.table[.column]".
    Returns the correct semantic namespace tuple.
    """
    parts = [p for p in (target or "").split(".") if p]
    if not parts:
        raise ValueError("empty target")

    if len(parts) == 1:                    # dataset
        return ns_sem_dataset(default_project, parts[0])
    if len(parts) == 2:                    # dataset.table
        return ns_sem_table(default_project, parts[0], parts[1])
    if len(parts) == 3:                    # dataset.table.column
        return ns_sem_column(default_project, parts[0], parts[1], parts[2])
    if len(parts) == 4:                    # project.dataset.table.column
        return ns_sem_column(parts[0], parts[1], parts[2], parts[3])

    raise ValueError(f"Unrecognized target format: {target!r}")

class Fact(BaseModel):
    """
    satisfying the fact type
    """
    content: str
    importance: Optional[str] = None
    category: Optional[str] = None

class Episode(BaseModel):
    """
    satisfying the episode type
    """
    observation: Optional[str] = None
    thoughts: Optional[str] = None
    action: Optional[str] = None
    result: Optional[str] = None

class Procedure(BaseModel):
    """
    satisfying the procedure type
    """
    name: Optional[str] = None
    conditions: Optional[str] = None
    steps: Optional[str] = None
    notes: Optional[str] = None

PYDANTIC_MODELS = {
    "fact": Fact,
    "episode": Episode,
    "procedure": Procedure,
}


class PatchedBigQueryVectorStore(BigQueryVectorStore):
    """
    The original BigQueryVectorStore has to be patched
    so that it allows for struct types
    """
    def _normalize_page_content(self, content: Any) -> str:
        if isinstance(content, dict):
            return json.dumps(content)
        return str(content)

    def _create_langchain_documents(
        self,
        search_results: List[List[Any]],
        k: int,
        num_queries: int,
        with_embeddings: bool = False,
    ) -> List[List[List[Any]]]:
        if len(search_results) == 0:
            return [[]]

        result_fields = list(search_results[0].keys())  # type: ignore[attr-defined]
        metadata_fields = [
            x
            for x in result_fields
            if x not in [self.embedding_field, self.content_field, "row_num"]
        ]
        documents = []
        for result in search_results:
            metadata = {
                metadata_field: result[metadata_field]
                for metadata_field in metadata_fields
            }
            document = Document(
                page_content=self._normalize_page_content(
                    result[self.content_field]
                ),
                metadata=metadata,
            )
            if with_embeddings:
                document_record = [
                    document,
                    metadata["score"],
                    result[self.embedding_field],  # type: ignore
                ]
            else:
                document_record = [document, metadata["score"]]
            documents.append(document_record)

        results_docs = [documents[i * k : (i + 1) * k] for i in range(num_queries)]
        return results_docs

    def get_documents(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[Union[Dict[str, Any], str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Search documents by their ids or metadata values.

        Args:
            ids: List of ids of documents to retrieve from the vectorstore.
            filter: (Optional) A dictionary or a string specifying filter criteria.
                - If a dictionary is provided, it should map column names to their
                corresponding values. The method will generate SQL expressions based
                on the data types defined in `self.table_schema`. The value is enclosed
                in single quotes unless the column is of type "INTEGER" or "FLOAT", in
                which case the value is used directly. E.g., `{"str_property": "foo",
                "int_property": 123}`.
                - If a string is provided, it is assumed to be a valid SQL WHERE clause.
        Returns:
            List of ids from adding the texts into the vectorstore.
        """

        if ids and len(ids) > 0:
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ArrayQueryParameter("ids", "STRING", ids),
                ]
            )
            id_expr = f"{self.doc_id_field} IN UNNEST(@ids)"
        else:
            job_config = None
            id_expr = "TRUE"

        where_filter_expr = self._create_filters(filter)

        job = self._bq_client.query(  # type: ignore[union-attr]
            f"""
                    SELECT * FROM `{self.full_table_id}`
                    WHERE {id_expr} AND {where_filter_expr}
                    """,
            job_config=job_config,
        )
        docs: List[Document] = []
        for row in job:
            metadata = {}
            for field in row.keys():
                if field not in [
                    self.embedding_field,
                    self.content_field,
                ]:
                    metadata[field] = row[field]
            metadata["__id"] = row[self.doc_id_field]
            doc = Document(
                page_content=self._normalize_page_content(row[self.content_field]),
                metadata=metadata
            )
            docs.append(doc)
        return docs

    def add_texts_with_embeddings(
        self,
        texts: List[Union[str, dict]],
        embs: List[List[float]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        # Generate IDs if not provided
        if ids is None:
            ids = [uuid.uuid4().hex for _ in texts]
        if metadatas is None:
            metadatas = [{} for _ in texts]

        values_dict: List[Dict[str, Any]] = []
        for idx, text, emb, metadata_dict in zip(ids, texts, embs, metadatas):
            record = {
                self.doc_id_field: idx,
                self.embedding_field: emb,
            }

            if self._is_record_column():
                # Use the structured data from metadata_dict for RECORD fields
                if isinstance(metadata_dict.get(self.content_field), dict):
                    record[self.content_field] = metadata_dict[self.content_field]
                else:
                    raise ValueError(
                        "Expected dict for RECORD column %s, got: %s" % (
                            self.content_field,
                            type(metadata_dict.get(self.content_field))
                        )
                    )
            else:
                # For non-RECORD fields, handle as before
                if isinstance(text, dict):
                    record[self.content_field] = json.dumps(text)
                elif isinstance(text, str):
                    record[self.content_field] = text
                else:
                    raise ValueError(
                        f"Expected str or dict for `{self.content_field}`, got: {type(text)}"
                    )

            record.update(metadata_dict)
            values_dict.append(record)

        logger.debug("Type of 'fact' field: %s", type(record[self.content_field]))
        logger.debug("Content of 'fact' field: %s", record[self.content_field])
        table = self._bq_client.get_table(self.full_table_id)
        try:

            # Schema enforcement happens at table definition time; load just sends rows.
            job = self._bq_client.load_table_from_json(values_dict, table)
            logger.debug("Loaded %d row(s) into %s", len(values_dict), self.full_table_id)
            job.result()  # Wait for the job to complete
        except google.api_core.exceptions.GoogleAPIError as e:
            # Handle errors returned by the Google API
            print("Google API error occurred: %s", e)
            logger.debug("Google API error occurred: %s", e)
            raise
        except ValueError as e:
            # Handle value errors, such as schema mismatches
            print("Value error: %s", e)
            logger.debug("Value error: %s", e)
            raise
        except Exception as e:
            # Handle other unexpected exceptions
            print("An unexpected error occurred: %s", e)
            logger.debug("An unexpected error occurred: %s", e)
            raise
        self._validate_bq_table()
        self._logger.debug("Stored %s records in BigQuery.", len(ids))
        self.sync_data()
        return ids

    def _is_record_column(self) -> bool:
        """Check whether the content_field is a RECORD in the BigQuery schema."""
        table = self._bq_client.get_table(self.full_table_id)
        for field in table.schema:
            if field.name == self.content_field:
                return field.field_type.upper() == "RECORD"
        return False

    def _validate_bq_table(self) -> Any:

        table_ref = bigquery.TableReference.from_string(self.full_table_id)

        try:
            # Attempt to retrieve the table information
            table = self._bq_client.get_table(table_ref)
        except NotFound:
            self._logger.debug("Couldn't find table %s", self.full_table_id)
            return

        schema = table.schema.copy()
        if schema:  # Check if table has a schema
            self.table_schema = {field.name: field.field_type for field in schema}
            columns = {c.name: c for c in schema}

            # Validate doc_id_field
            validate_column_in_bq_schema(
                column_name=self.doc_id_field,
                columns=columns,
                expected_types=["STRING"],
                expected_modes=["NULLABLE", "REQUIRED"],
            )

            # Validate content_field based on actual schema
            if self.content_field in columns:
                content_field_type = columns[self.content_field].field_type.upper()
                expected_types = [content_field_type]  # Accept the actual type
                validate_column_in_bq_schema(
                    column_name=self.content_field,
                    columns=columns,
                    expected_types=expected_types,
                    expected_modes=["NULLABLE", "REQUIRED"],
                )
            else:
                raise ValueError(f"Column '{self.content_field}' not found in the table schema.")

            # Validate embedding_field
            validate_column_in_bq_schema(
                column_name=self.embedding_field,
                columns=columns,
                expected_types=["FLOAT", "FLOAT64"],
                expected_modes=["REPEATED"],
            )

            # Validate extra_fields if provided
            if self.extra_fields is None:
                extra_fields = {}
                for column in schema:
                    if column.name not in [
                        self.doc_id_field,
                        self.content_field,
                        self.embedding_field,
                    ]:
                        # Check for unsupported REPEATED mode
                        if column.mode == "REPEATED":
                            raise ValueError(
                                f"Column '{column.name}' is REPEATED. "
                                f"REPEATED fields are not supported in this context."
                            )
                        extra_fields[column.name] = column.field_type
                self.extra_fields = extra_fields
            else:
                for field, type in self.extra_fields.items():
                    validate_column_in_bq_schema(
                        column_name=field,
                        columns=columns,
                        expected_types=[type],
                        expected_modes=["NULLABLE", "REQUIRED"],
                    )

            self._logger.debug("Table %s validated", self.full_table_id)
        return table_ref

class BigQueryMemoryStore(AsyncBatchedBaseStore):
    """
    This gives persistence of memory directly into Bigquery.
    """
    # We don't currently implement TTL semantics end-to-end in BigQuery.
    # Keep this False so BaseStore.put enforces correctness.
    supports_ttl: bool = False

    def __init__(
        self,
        vectorstore: PatchedBigQueryVectorStore,
        content_field: str,
        content_model: Optional[Type[BaseModel]] = None,
        schema: Optional[List[bigquery.SchemaField]] = None,
    ):
        # AsyncBatchedBaseStore expects to be constructed inside a running
        # event loop. When we build this store in a worker thread (via
        # asyncio.to_thread) there is no running loop, so `super().__init__()`
        # would raise `RuntimeError: no running event loop`.
        #
        # We catch that here and provide the attributes that the base class'
        # destructor expects so shutdown is clean. Our implementation of
        # `abatch` does not rely on the base-class batching machinery.
        try:
            super().__init__()
        except RuntimeError:
            # We intentionally do not rely on AsyncBatchedBaseStore's
            # background queue here. This store may be constructed in a
            # worker thread with no running loop.
            self._loop = None  # type: ignore[assignment]
            self._task = None  # type: ignore[assignment]
            self._aqueue = None  # type: ignore[assignment]

        self.vectorstore = vectorstore
        self.content_field = content_field
        self.content_model = content_model
        self.schema = schema

    @classmethod
    def from_client(
        cls,
        dataset_name: str,
        table_name: str,
        content_field: str,
        content_model: Optional[Type[BaseModel]] = None,
        schema: Optional[List[bigquery.SchemaField]] = None,
        **kwargs,
    ) -> "BigQueryMemoryStore":
        bq_client = get_bq_client()
        vectorstore = PatchedBigQueryVectorStore(
            embedding=get_embedding(),
            project_id=bq_client.project,
            doc_id_field="doc_id",
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

    def _normalize_structured_field(self, raw: Any) -> dict:
        if isinstance(raw, dict) and self.content_model:
            # Wrap into the content model (e.g., Fact)
            return raw  # Return as a dictionary
        elif isinstance(raw, str) and self.content_model:
            # If it's a string, wrap it into the content model
            return {"content":raw}
        elif self.content_model is not None and isinstance(raw, self.content_model):
            return raw.dict()  # Already in the correct format
        else:
            raise ValueError(
                "%s must be a dict, str, or %s — got: %s"
                % (
                    self.content_field,
                    self.content_model.__name__ if self.content_model else "BaseModel",
                    type(raw),
                )
            )

    async def aput(
        self,
        namespace: Tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: Optional[Union[Literal[False], list[str]]] = None,
        *,
        ttl: float | None | NotProvided = NOT_PROVIDED,
    ) -> None:
        # Validate namespace consistently with BaseStore expectations
        _validate_namespace(namespace)

        logger.info(
            "[aput] Inserting doc_id=%s into namespace=%s",
            key,
            ".".join(namespace),
        )
        data = {"namespace": ".".join(namespace), "doc_id": key}
        logger.debug("[aput] initial data: %s", json.dumps(data, indent=2))
        logger.debug("[aput] Using content field: %s", self.content_field)

        raw_content = value.get("content")
        logger.debug("[aput] Raw content retrieved: %s", raw_content)

        if raw_content is None:
            logger.error("Content for %s is None. This is not expected.", self.content_field)
            return

        actual_content = raw_content

        # ─── EPISODIC-SPECIFIC HANDLING ───
        # Only episodic table has a timestamp column (REQUIRED).
        # Semantic and procedural tables do NOT have timestamp.
        if self.content_field == "episode":
            # Handle wrapped structure: {"episode": {...}, "timestamp": ...}
            # This happens when caller sends: {"episode": Episode(...), "timestamp": datetime}
            if isinstance(raw_content, dict) and "episode" in raw_content:
                actual_content = raw_content["episode"]  # Unwrap to get Episode fields only
                logger.debug("[aput] Unwrapped episode content from nested structure")
                
                # Extract timestamp from content level if present
                if "timestamp" in raw_content:
                    ts = raw_content["timestamp"]
                    if isinstance(ts, datetime):
                        data["timestamp"] = ts.isoformat()
                    elif isinstance(ts, str):
                        data["timestamp"] = ts
                    logger.debug("[aput] Extracted timestamp from raw_content: %s", data.get("timestamp"))
            
            # Fallback: check value level (in case timestamp passed at invoke level)
            if "timestamp" not in data and "timestamp" in value:
                ts = value["timestamp"]
                if isinstance(ts, datetime):
                    data["timestamp"] = ts.isoformat()
                elif isinstance(ts, str):
                    data["timestamp"] = ts
                logger.debug("[aput] Extracted timestamp from value: %s", data.get("timestamp"))
            
            # Final fallback: generate timestamp if still missing (field is REQUIRED in schema)
            if "timestamp" not in data:
                data["timestamp"] = datetime.now(timezone.utc).isoformat()
                logger.debug("[aput] Generated fallback timestamp: %s", data["timestamp"])
        # ─── END EPISODIC-SPECIFIC HANDLING ───
        # For semantic/procedural: actual_content = raw_content unchanged, no timestamp added

        text = self._normalize_structured_field(actual_content)
        embedding_content = json.dumps(text)
        logger.debug("[aput] Normalized content: %s", text)
        data[self.content_field] = text
        logger.debug("[aput] Value being inserted: %s", json.dumps(data, indent=2, default=str))
        logger.debug("[aput] the type(data[self.content_field]): %s", type(data[self.content_field]))

        doc = Document(page_content=embedding_content, metadata=data, id=key)

        # IMPORTANT: offload the blocking BigQuery call to a worker thread
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            self.vectorstore.add_documents,
            [doc],
        )

    async def aget(
            self,
            namespace: Tuple[str, ...],
            key: str,
            *,
            refresh_ttl: Optional[bool] = None
    ) -> Optional[Item]:
        logger.debug("[aget] Retrieving doc_id=%s from namespace=%s", key, namespace)

        loop = asyncio.get_running_loop()
        docs = await loop.run_in_executor(
            None,
            self.vectorstore.get_documents,
            [key],
        )
        if not docs:
            return None
        doc = docs[0]
        metadata = doc.metadata
        ns = tuple(metadata.get("namespace", "").split("."))
        content_val = doc.page_content
        if isinstance(content_val, str) and self.content_model:
            try:
                metadata[self.content_field] = self.content_model(**json.loads(content_val))
            except Exception:
                pass

        raw_created = doc.metadata.get("created_at")
        raw_updated = doc.metadata.get("updated_at")

        def to_dt(x):
            if isinstance(x, datetime):
                return x
            if isinstance(x, str):
                return datetime.fromisoformat(x)
            return datetime.now(timezone.utc)

        created_at = to_dt(raw_created)
        updated_at = to_dt(raw_updated)

        return Item(
            namespace=ns,
            key=key,
            value=metadata,
            created_at=created_at,
            updated_at=updated_at
        )
    # --- Overloads to satisfy both older & newer langgraph versions ---
    @overload
    async def asearch(  # current shape (with refresh_ttl)
        self,
        namespace_prefix: tuple[str, ...], /,
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: bool | None = None,
    ) -> list[SearchItem]: ...
    @overload
    async def asearch(  # older shape (without refresh_ttl)
        self,
        namespace_prefix: tuple[str, ...], /,
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[SearchItem]: ...

    # --- Implementation matches the *newer* signature; extra kw is optional ---
    async def asearch(
        self,
        namespace_prefix: tuple[str, ...], /,
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: bool | None = None,   # accepted; not used here
    ) -> list[SearchItem]:
        logger.info(
            "[asearch] ns_prefix=%s, query=%r, limit=%d, offset=%d",
            namespace_prefix, query, limit, offset
        )

        if not query:
            return []

        # If caller gave a dict (or you have a content_model), preserve its shape
        if isinstance(query, dict) or self.content_model:
            try:
                model_cls = self.content_model
                if model_cls and isinstance(query, dict):
                    query = json.dumps(model_cls(**query).dict())
                elif isinstance(query, dict):
                    query = json.dumps(query)
            except Exception:
                query = str(query)

        loop = asyncio.get_running_loop()
        # Offload vectorstore similarity search to avoid blocking
        hits = await loop.run_in_executor(
            None,
            lambda: self.vectorstore.similarity_search_with_score(
                query=query, filter=filter, k=limit + offset
            ),
        )
        hits = hits[offset:]

        out: list[SearchItem] = []
        for doc, score in hits:
            ns = tuple((doc.metadata.get("namespace") or "").split("."))
            key = doc.metadata.get("doc_id") or ""

            # Deserialize page_content if it was JSON
            raw_page = doc.page_content
            try:
                content = json.loads(raw_page)
            except Exception:
                content = raw_page

            md = dict(doc.metadata)
            md[self.content_field] = content

            # Normalize timestamps defensively
            def _to_dt(x):
                if isinstance(x, datetime):
                    return x
                if isinstance(x, str):
                    try:
                        return datetime.fromisoformat(x)
                    except Exception:
                        pass
                return datetime.now(timezone.utc)

            created_at = _to_dt(doc.metadata.get("created_at"))
            updated_at = _to_dt(doc.metadata.get("updated_at"))

            out.append(
                SearchItem(
                    namespace=ns,
                    key=key,
                    value=md,
                    score=float(score),
                    created_at=created_at,
                    updated_at=updated_at,
                )
            )
        return out

    def search(
        self,
        namespace_prefix: tuple[str, ...], /,
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
        # accept refresh_ttl if the base class ever passes it through
        refresh_ttl: bool | None = None,
    ) -> list[SearchItem]:
        """
        Synchronous search wrapper.

        Some callers (e.g. sync tool paths) expect `search` to be *sync*.
        We resolve the async `asearch` coroutine here so nobody ever ends up
        with an un-awaited `BigQueryMemoryStore.asearch` object.
        """
        async def _runner() -> list[SearchItem]:
            return await self.asearch(
                namespace_prefix,
                query=query,
                filter=filter,
                limit=limit,
                offset=offset,
                refresh_ttl=refresh_ttl,
            )

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop: safe to create one and run to completion
            return asyncio.run(_runner())

        # Already in an event loop: hop to a worker thread with its own loop
        def _thread_runner() -> list[SearchItem]:
            new_loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(new_loop)
                return new_loop.run_until_complete(_runner())
            finally:
                new_loop.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(_thread_runner).result()

    async def adelete(self, namespace: Tuple[str, ...], key: str) -> None:
        logger.info("[adelete] Deleting doc_id=%s from namespace=%s", key, namespace)

        adelete = getattr(self.vectorstore, "adelete", None)
        if callable(adelete):
            await adelete(ids=[key])  # type: ignore[misc]
            return

        delete = getattr(self.vectorstore, "delete", None)
        if not callable(delete):
            raise NotImplementedError("Vector store does not support delete/adelete")

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, delete, [key])

    def mget(self, keys: Sequence[str]) -> List[Optional[dict]]:
        """
        Synchronous multi-get. BigQueryVectorStore only exposes a synchronous
        get_documents() API, so we call it directly here.
        """
        return self.vectorstore.get_documents(ids=list(keys))

    async def amget(self, keys: Sequence[str]) -> List[Optional[dict]]:
        """
        Async-friendly multi-get.

        The underlying BigQueryVectorStore is synchronous, so we offload the
        blocking call to a worker thread when running inside an event loop.
        This keeps the Store API consistent for callers and avoids any
        un-awaited coroutine warnings.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self.vectorstore.get_documents, list(keys)
        )

    def mset(self, key_value_pairs: Sequence[Tuple[str, dict]]) -> None:
        documents = [
            Document(page_content=value.get(self.content_field), metadata={**value, "doc_id": key})
            for key, value in key_value_pairs
        ]
        self.vectorstore.add_documents(documents)

    async def amset(
            self,
            key_value_pairs: Sequence[Tuple[str, dict]]
    ) -> None:
        """
        Async-friendly multi-set.

        BigQueryVectorStore.add_documents is synchronous, so we offload it to
        a worker thread to avoid blocking the event loop.
        """
        documents = [
            Document(
                page_content=value.get(self.content_field),
                metadata={**value, "doc_id": key}
            )
            for key, value in key_value_pairs
        ]
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            self.vectorstore.add_documents,
            documents
        )


    def mdelete(self, keys: Sequence[str]) -> None:
        """
        Synchronous delete.

        Prefer a synchronous `delete()` method on the underlying vectorstore.
        If only an async `adelete()` is available, run it to completion in a
        one-shot event loop so callers do not get a `coroutine was never awaited`
        warning.
        """
        delete = getattr(self.vectorstore, "delete", None)
        if callable(delete):
            delete(ids=list(keys))
            return

        adelete = getattr(self.vectorstore, "adelete", None)
        if adelete is None:
            raise NotImplementedError("Vector store does not support delete/adelete")

        # If we're in a plain sync context, just run the coroutine to completion.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(adelete(ids=list(keys)))
            return

        # If we're *already* in an event loop, hop to a worker thread that owns
        # its own loop so we don't try to nest event loops and explode.
        def _runner() -> None:
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                loop.run_until_complete(adelete(ids=list(keys)))
            finally:
                loop.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            ex.submit(_runner).result()

    async def amdelete(self, keys: Sequence[str]) -> None:
        """
        Async delete wrapper.

        If the underlying vectorstore exposes `adelete`, await it directly;
        otherwise fall back to running the sync `delete()` in a worker thread.
        """
        adelete = getattr(self.vectorstore, "adelete", None)
        if callable(adelete):
            await adelete(ids=list(keys))  # type: ignore[misc]
            return

        delete = getattr(self.vectorstore, "delete", None)
        if not callable(delete):
            raise NotImplementedError("Vector store does not support delete/adelete")

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, delete, list(keys))

    def yield_keys(self, prefix: Optional[str] = None) -> Iterator[str]:
        "yields keys"
        return self.vectorstore.yield_keys(prefix=prefix)

    async def ayield_keys(self, prefix: Optional[str] = None) -> AsyncIterator[str]:
        """
        Async adapter for yield_keys.
        BigQueryVectorStore.yield_keys is synchronous; offload collection.
        """
        loop = asyncio.get_running_loop()
        keys = await loop.run_in_executor(
            None, lambda: list(self.vectorstore.yield_keys(prefix=prefix))
        )
        for key in keys:
            yield key

    # ---------------------------------------------------------------------
    # Sync API: override AsyncBatchedBaseStore wrappers.
    #
    # We cannot rely on AsyncBatchedBaseStore._loop because this store may be
    # constructed in a worker thread (no running loop). These overrides ensure
    # callers never hit run_coroutine_threadsafe(..., None).
    # ---------------------------------------------------------------------

    def batch(self, ops: IterableABC[Op]) -> list[Result]:
        async def _runner() -> list[Result]:
            return await self.abatch(ops)

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_runner())

        def _thread_runner() -> list[Result]:
            new_loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(new_loop)
                return new_loop.run_until_complete(_runner())
            finally:
                new_loop.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(_thread_runner).result()

    def get(
        self,
        namespace: tuple[str, ...],
        key: str,
        *,
        refresh_ttl: bool | None = None,
    ) -> Item | None:
        return BaseStore.get(self, namespace, key, refresh_ttl=refresh_ttl)

    def put(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: Literal[False] | list[str] | None = None,
        *,
        ttl: float | None | NotProvided = NOT_PROVIDED,
    ) -> None:
        return BaseStore.put(self, namespace, key, value, index=index, ttl=ttl)

    def delete(self, namespace: tuple[str, ...], key: str) -> None:
        return BaseStore.delete(self, namespace, key)

    def list_namespaces(
        self,
        *,
        prefix: NamespacePath | None = None,
        suffix: NamespacePath | None = None,
        max_depth: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        return BaseStore.list_namespaces(
            self,
            prefix=prefix,
            suffix=suffix,
            max_depth=max_depth,
            limit=limit,
            offset=offset,
        )

    async def abatch(self, ops: Iterable[Op]) -> List[Result]:
        ops_list = list(ops)
        logger.info("[abatch] Executing %s batch operations", len(ops_list))
        results: List[Result] = []
        for op in ops_list:
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

async def _build_store_in_thread(
    *,
    table_name: str,
    content_field: str,
    model: Type[BaseModel],
    schema: List[bigquery.SchemaField]
) -> "BigQueryMemoryStore":
    """
    Build a BigQueryMemoryStore in a worker thread so that the synchronous
    BigQuery dataset/table validation runs off the main event loop (and
    doesn't trip LangGraph dev's blocking-call detector).
    """

    def _builder() -> "BigQueryMemoryStore":
        return BigQueryMemoryStore.from_client(
            dataset_name=DATASET_ID,
            table_name=table_name,
            content_field=content_field,
            content_model=model,
            schema=schema,
        )

    # run in default ThreadPoolExecutor
    return await asyncio.to_thread(_builder)

#---------------------------------get memory tools--------------------------------
async def get_memory_tools(
    namespace_templates: Dict[str, NamespaceTemplate]
) -> List:
    """
    Create pairs of (manage, search) memory tools for each namespace template.

    Args:
      namespace_templates: a dict where
         - key = tool‐base‐name (e.g. "semantic", "episodic", "procedural", or anything you choose)
         - value = a NamespaceTemplate tuple, e.g.
             ("metadata", "{object_type}", "{langgraph_auth_user_id}", "{object_name}")

    Returns:
      A list of all created tools (both manage_* and search_*).
    """
    tools: List = []

    # Build all three stores in worker threads so that the synchronous
    # BigQuery API calls (dataset/table validation) do not block the
    # main asyncio event loop under LangGraph dev.
    (
        semantic_memory_store,
        episodic_memory_store,
        procedural_memory_store,
    ) = await asyncio.gather(
        _build_store_in_thread(
            table_name=SEMANTIC_TABLE,
            content_field=CONTENT_FIELDS[SEMANTIC_TABLE],
            model=PYDANTIC_MODELS[CONTENT_FIELDS[SEMANTIC_TABLE]],
            schema=SCHEMAS[SEMANTIC_TABLE],
        ),
        _build_store_in_thread(
            table_name=EPISODIC_TABLE,
            content_field=CONTENT_FIELDS[EPISODIC_TABLE],
            model=PYDANTIC_MODELS[CONTENT_FIELDS[EPISODIC_TABLE]],
            schema=SCHEMAS[EPISODIC_TABLE],
        ),
        _build_store_in_thread(
            table_name=PROCEDURAL_TABLE,
            content_field=CONTENT_FIELDS[PROCEDURAL_TABLE],
            model=PYDANTIC_MODELS[CONTENT_FIELDS[PROCEDURAL_TABLE]],
            schema=SCHEMAS[PROCEDURAL_TABLE],
        ),
    )
    # 2. For each namespace_template, build both a manage‐tool and a search‐tool
    for base_key, ns_template in namespace_templates.items():
        # Determine which store to hook up:
        if base_key == "semantic":
            store = semantic_memory_store
            schema_model = Fact
        elif base_key == "episodic":
            store = episodic_memory_store
            schema_model = PYDANTIC_MODELS[CONTENT_FIELDS[EPISODIC_TABLE]]
        elif base_key == "procedural":
            store = procedural_memory_store
            schema_model = PYDANTIC_MODELS[CONTENT_FIELDS[PROCEDURAL_TABLE]]
        else:
            # If you have some other base_key (e.g. "www", "dbt_model"), choose
            # to either re‐use semantic_store or create your own. For now:
            store = semantic_memory_store
            schema_model = Fact

        # 2a. Create the “manage” tool
        manage_tool = create_manage_memory_tool(
            namespace=ns_template,
            store=store,
            name=f"manage_{base_key}_memory",
            schema=schema_model
        )
        tools.append(manage_tool)

        # 2b. Create the corresponding “search” tool
        search_tool = create_search_memory_tool(
            namespace=ns_template,
            store=store,
            name=f"search_{base_key}_memory"
        )
        tools.append(search_tool)

    return tools
