import json
import base64
from google.cloud import bigquery
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

# Function to initialize a BigQueryVectorStore
def initialize_vector_store(table_name: str) -> BigQueryVectorStore:
    return BigQueryVectorStore(
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

# Create memory tools
semantic_manage_tool = create_manage_memory_tool(namespace="semantic_memories", store=semantic_memory_store)
episodic_manage_tool = create_manage_memory_tool(namespace="episodic_memories", store=episodic_memory_store)
procedural_manage_tool = create_manage_memory_tool(namespace="procedural_memories", store=procedural_memory_store)

semantic_search_tool = create_search_memory_tool(namespace="semantic_memories", store=semantic_memory_store)
episodic_search_tool = create_search_memory_tool(namespace="episodic_memories", store=episodic_memory_store)
procedural_search_tool = create_search_memory_tool(namespace="procedural_memories", store=procedural_memory_store)

# Expose memory tools for use in agent.py
MEMORY_TOOLS = [
    semantic_manage_tool, semantic_search_tool,
    episodic_manage_tool, episodic_search_tool,
    procedural_manage_tool, procedural_search_tool
]