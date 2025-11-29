#!/usr/bin/env python3
"""
Memory Operations Logger

Tracks all long-term memory operations (create/retrieve) with correlation
to conversations, turns, and LangSmith traces. This enables comprehensive
analysis of memory usage patterns and performance.
"""

import logging
import json
import uuid
import time
import threading
import collections
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from google.cloud import bigquery
from google.api_core.exceptions import TooManyRequests
import backoff
from src.langgraph_slack.config import PROJECT_ID, CREDENTIALS, LOCATION

logger = logging.getLogger(__name__)

# Global throttling mechanism to prevent BigQuery rate limiting
# Track last load time per table to enforce 2s minimum delay
_last_load_timestamps = collections.defaultdict(lambda: 0.0)
_load_lock = threading.Lock()

@backoff.on_exception(backoff.expo, TooManyRequests, max_time=300)
def _safe_streaming_insert(values: List[Dict], table_ref: str, client: bigquery.Client):
    """Safely stream JSON data to BigQuery with retry on rate limits."""
    table = client.get_table(table_ref)
    errors = client.insert_rows_json(table, values)
    if errors:
        raise RuntimeError(f"BigQuery streaming insert errors: {errors}")

def _stream_with_throttling(values: List[Dict], table_ref: str, client: bigquery.Client):
    """Stream data with minimal throttling (streaming inserts have higher quotas)."""
    # Streaming inserts don't have the same rate limits as load jobs
    # But we still add minimal throttling to be safe
    with _load_lock:
        # Much shorter delay for streaming inserts (0.1s vs 2s)
        delta = time.time() - _last_load_timestamps[table_ref]
        if delta < 0.1:
            sleep_time = 0.1 - delta
            logger.debug(f"Minimal throttling for streaming insert: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)

        # Perform the streaming insert with retry
        _safe_streaming_insert(values, table_ref, client)
        _last_load_timestamps[table_ref] = time.time()

class MemoryOperationsLogger:
    """Logger for tracking memory operations with conversation correlation."""

    def __init__(self):
        """Initialize the memory operations logger."""
        self.client = bigquery.Client(credentials=CREDENTIALS, project=PROJECT_ID, location=LOCATION)
        self.table_id = f"{PROJECT_ID}.agent_system_memory.long_term_memory_operations"
        
    def log_memory_operation(
        self,
        doc_id: str,
        memory_store: str,  # "semantic_memory", "episodic_memory", "procedural_memory"
        memory_type: str,  # "semantic", "episodic", "procedural"
        operation_type: str,  # "CREATE" or "RETRIEVE"
        success: bool,
        namespace: Optional[str] = None,
        conversation_id: Optional[str] = None,
        turn_number: Optional[int] = None,
        thread_id: Optional[str] = None,
        creator_process: Optional[str] = None,
        retriever_process: Optional[str] = None,
        langsmith_trace_id: Optional[str] = None,
        operation_context: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        processing_time_ms: Optional[int] = None,
    ) -> str:
        """
        Log a memory operation to BigQuery.
        
        Args:
            doc_id: Document ID from the memory table
            memory_store: Which memory store (semantic_memory, episodic_memory, procedural_memory)
            memory_type: Type of memory (semantic, episodic, procedural)
            operation_type: Type of operation (CREATE or RETRIEVE)
            success: Whether the operation completed successfully
            namespace: Memory namespace for logical grouping
            conversation_id: Conversation ID for correlation
            turn_number: Turn number within the conversation
            thread_id: LangGraph thread ID for correlation
            creator_process: Process that created the memory
            retriever_process: Process that retrieved the memory
            langsmith_trace_id: LangSmith trace ID for detailed tracking
            operation_context: Additional context about the operation
            error_message: Error message if operation failed
            processing_time_ms: Time taken to complete the operation
            
        Returns:
            operation_id: Unique identifier for this operation
        """
        operation_id = str(uuid.uuid4())
        
        # Prepare the record
        record = {
            "operation_id": operation_id,
            "doc_id": doc_id,
            "memory_store": memory_store,
            "memory_type": memory_type,
            "operation_type": operation_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "namespace": namespace,
            "conversation_id": conversation_id,
            "turn_number": turn_number,
            "thread_id": thread_id,
            "creator_process": creator_process,
            "retriever_process": retriever_process,
            "langsmith_trace_id": langsmith_trace_id,
            "operation_context": json.dumps(operation_context) if operation_context else None,
            "success": success,
            "error_message": error_message,
            "processing_time_ms": processing_time_ms,
        }
        
        try:
            # Insert the record into BigQuery using streaming inserts (no rate limits)
            _stream_with_throttling([record], self.table_id, self.client)

            logger.debug(f"✅ Logged {operation_type} operation for {memory_type} memory: {doc_id}")

        except Exception as e:
            logger.error(f"❌ Failed to log memory operation: {e}")
            logger.error(f"   Operation: {operation_type} {memory_type} {doc_id}")
            # Don't raise - logging failures shouldn't break the main operation
            
        return operation_id
    
    def log_create_operation(
        self,
        doc_id: str,
        memory_store: str,
        memory_type: str,
        success: bool,
        namespace: Optional[str] = None,
        conversation_id: Optional[str] = None,
        turn_number: Optional[int] = None,
        thread_id: Optional[str] = None,
        creator_process: Optional[str] = None,
        langsmith_trace_id: Optional[str] = None,
        operation_context: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        processing_time_ms: Optional[int] = None,
    ) -> str:
        """Log a memory CREATE operation."""
        return self.log_memory_operation(
            doc_id=doc_id,
            memory_store=memory_store,
            memory_type=memory_type,
            operation_type="CREATE",
            success=success,
            namespace=namespace,
            conversation_id=conversation_id,
            turn_number=turn_number,
            thread_id=thread_id,
            creator_process=creator_process,
            langsmith_trace_id=langsmith_trace_id,
            operation_context=operation_context,
            error_message=error_message,
            processing_time_ms=processing_time_ms,
        )
    
    def log_retrieve_operation(
        self,
        doc_id: str,
        memory_store: str,
        memory_type: str,
        success: bool,
        namespace: Optional[str] = None,
        conversation_id: Optional[str] = None,
        turn_number: Optional[int] = None,
        thread_id: Optional[str] = None,
        retriever_process: Optional[str] = None,
        langsmith_trace_id: Optional[str] = None,
        operation_context: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        processing_time_ms: Optional[int] = None,
    ) -> str:
        """Log a memory RETRIEVE operation."""
        return self.log_memory_operation(
            doc_id=doc_id,
            memory_store=memory_store,
            memory_type=memory_type,
            operation_type="RETRIEVE",
            success=success,
            namespace=namespace,
            conversation_id=conversation_id,
            turn_number=turn_number,
            thread_id=thread_id,
            retriever_process=retriever_process,
            langsmith_trace_id=langsmith_trace_id,
            operation_context=operation_context,
            error_message=error_message,
            processing_time_ms=processing_time_ms,
        )

# Global instance for easy access
_memory_operations_logger = None

def get_memory_operations_logger() -> MemoryOperationsLogger:
    """Get the global memory operations logger instance."""
    global _memory_operations_logger
    if _memory_operations_logger is None:
        _memory_operations_logger = MemoryOperationsLogger()
    return _memory_operations_logger

# Convenience functions for common operations
def log_memory_create(
    doc_id: str,
    memory_store: str,
    memory_type: str,
    success: bool,
    **kwargs
) -> str:
    """Log a memory CREATE operation using the global logger."""
    return get_memory_operations_logger().log_create_operation(
        doc_id=doc_id,
        memory_store=memory_store,
        memory_type=memory_type,
        success=success,
        **kwargs
    )

def log_memory_retrieve(
    doc_id: str,
    memory_store: str,
    memory_type: str,
    success: bool,
    **kwargs
) -> str:
    """Log a memory RETRIEVE operation using the global logger."""
    return get_memory_operations_logger().log_retrieve_operation(
        doc_id=doc_id,
        memory_store=memory_store,
        memory_type=memory_type,
        success=success,
        **kwargs
    )

# Context manager for timing operations
class MemoryOperationTimer:
    """Context manager for timing memory operations."""
    
    def __init__(self, doc_id: str, memory_store: str, memory_type: str, operation_type: str, **kwargs):
        self.doc_id = doc_id
        self.memory_store = memory_store
        self.memory_type = memory_type
        self.operation_type = operation_type
        self.kwargs = kwargs
        self.start_time = None
        self.success = False
        self.error_message = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        processing_time_ms = int((time.time() - self.start_time) * 1000)
        
        if exc_type is None:
            self.success = True
        else:
            self.success = False
            self.error_message = str(exc_val)
        
        # Log the operation
        get_memory_operations_logger().log_memory_operation(
            doc_id=self.doc_id,
            memory_store=self.memory_store,
            memory_type=self.memory_type,
            operation_type=self.operation_type,
            success=self.success,
            error_message=self.error_message,
            processing_time_ms=processing_time_ms,
            **self.kwargs
        )
        
        # Don't suppress exceptions
        return False
