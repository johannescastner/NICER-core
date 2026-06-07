"""Batch-LOAD writes for canonical_regions / canonical_decomposition_edges.

Historical context — the streaming-buffer DML blackout (2026-04-14/15):
  The ingest scripts (HRE, Glottography, ArchaeoGLOBE, etc.) wrote
  ``canonical_regions`` and ``canonical_decomposition_edges`` via
  ``bq.insert_rows_json`` — the legacy ``tabledata.insertAll`` streaming
  API.  Streaming writes sit in a per-table streaming buffer for up to
  ~90 minutes before they migrate to managed storage.  While *any* row
  in the table is in the buffer, BigQuery rejects UPDATE / DELETE /
  MERGE statements on that table (the planner conservatively fails
  DML when it cannot prove that the predicate excludes buffered rows).

  That created a 90-minute DML blackout after every ingest — including
  an inability to correct an ingested row, or to idempotently
  DELETE-then-INSERT from a second ingest.  One Glottography run
  locked canonical_regions for 90 minutes and blocked the follow-up
  HRE QID correction we did via BQ verification.

Fix: use batch LOAD jobs instead of streaming inserts.  Batch loads
write directly to managed storage.  Zero buffer, DML works immediately.

This module exposes a single helper ``batch_load_json_rows`` that
takes the same input shape as ``insert_rows_json`` but routes through
``load_table_from_file`` with NEWLINE_DELIMITED_JSON.
"""
from __future__ import annotations

import io
import json
import time

import sqlglot.expressions as exp
from google.cloud import bigquery


def batch_load_json_rows(
    bq: bigquery.Client,
    table_fqn: str,
    rows: list[dict],
    *,
    write_disposition: str = bigquery.WriteDisposition.WRITE_APPEND,
) -> None:
    """Append JSON rows to a table via a batch LOAD job (no streaming buffer).

    Args:
        bq: BigQuery client.
        table_fqn: fully-qualified table name ``project.dataset.table``.
        rows: list of dicts; each dict becomes one row.  Keys must match
            the target table's column names.  NULL values can be passed
            as Python ``None``.
        write_disposition: default ``WRITE_APPEND`` (matches
            ``insert_rows_json`` semantics).  Pass ``WRITE_TRUNCATE`` to
            replace the whole table.  DML alternatives (DELETE WHERE ...
            before INSERT) work correctly with batch-loaded rows because
            they never enter a streaming buffer.
    """
    if not rows:
        return
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        write_disposition=write_disposition,
    )
    data = "\n".join(json.dumps(r, default=str) for r in rows).encode("utf-8")
    job = bq.load_table_from_file(
        io.BytesIO(data), table_fqn, job_config=job_config,
    )
    job.result()


def merge_upsert_json_rows(
    bq: bigquery.Client,
    target_table_fqn: str,
    rows: list[dict],
    *,
    merge_keys: list[str],
    delete_unmatched: bool = False,
) -> None:
    """Idempotent upsert: batch-LOAD ``rows`` into a temp staging table,
    then ``MERGE`` into ``target_table_fqn`` matching on ``merge_keys``.

    Non-destructive by design — rows in target that are absent from
    source are preserved.  Re-running with the same inputs converges
    on the same final state without duplication, without DELETE, and
    without fighting the streaming buffer.

    Why not ``DELETE + INSERT``: a failed re-run mid-flight leaves the
    table empty for that source; it also requires DML on the target,
    which fights the streaming buffer; and it gives up the ability to
    *correct* existing rows in place.

    Args:
        bq: BigQuery client.
        target_table_fqn: ``project.dataset.table``.
        rows: list of dicts.  Keys must match target-table column names.
            Every key other than those in ``merge_keys`` will be
            ``UPDATE``d on match and ``INSERT``ed on no-match.
        merge_keys: composite natural key used to match source rows to
            target rows.  For ``canonical_regions``: ``["region_id"]``.
            For ``canonical_decomposition_edges``:
            ``["parent_region_id", "child_region_id", "set_name"]``.
        delete_unmatched: opt-in, default ``False``. When ``True``, target
            rows NOT present in ``rows`` are DELETED in the SAME atomic MERGE
            (``WHEN NOT MATCHED BY SOURCE THEN DELETE``) — for a FULL re-seed
            that must mirror the source exactly (e.g. the warehouse
            ``schema_embeddings`` seed, where a wide table's old whole-table
            doc must be removed once it's replaced by chunked column-group
            docs). Leave ``False`` for partial/single-source merges, which
            must preserve other sources' rows (the default contract above).

    Notes:
      * The staging table is batch-LOADed (no streaming buffer).  Once
        all writers to the target table use batch LOAD + MERGE, the
        target never accrues a streaming buffer, and MERGE always
        succeeds.
      * Staging table name carries a millisecond timestamp suffix so
        concurrent ingests don't collide.  Dropped on success.
    """
    if not rows:
        return
    cols = list(rows[0].keys())
    assert merge_keys, "merge_keys must be non-empty"
    for k in merge_keys:
        assert k in cols, f"merge_key {k!r} missing from row keys {cols}"

    # Derive staging schema from target to avoid type-inference surprise.
    # The INFORMATION_SCHEMA table reference is a structured BQ identifier;
    # the only user-supplied literal is the table name, which we pass via
    # a query parameter rather than string substitution.
    project, dataset, table_name = target_table_fqn.split(".")
    schema_q = (
        "SELECT column_name, data_type FROM "
        f"`{project}.{dataset}.INFORMATION_SCHEMA.COLUMNS` "
        "WHERE table_name = @table_name"
    )
    schema_job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("table_name", "STRING", table_name),
        ]
    )
    target_schema = {
        r.column_name: r.data_type
        for r in bq.query(schema_q, job_config=schema_job_config).result()
    }
    missing = [c for c in cols if c not in target_schema]
    if missing:
        raise ValueError(
            f"rows contain columns not in target {target_table_fqn}: {missing}"
        )

    staging_fqn = f"{target_table_fqn}_merge_staging_{int(time.time() * 1000)}"

    # CREATE OR REPLACE TABLE staging via sqlglot AST.
    create_stmt = exp.Create(
        this=exp.Schema(
            this=exp.to_table(staging_fqn, dialect="bigquery"),
            expressions=[
                exp.ColumnDef(
                    this=exp.to_identifier(c, quoted=True),
                    kind=exp.DataType.build(
                        target_schema[c], dialect="bigquery"
                    ),
                )
                for c in cols
            ],
        ),
        kind="TABLE",
        replace=True,
    )
    bq.query(create_stmt.sql(dialect="bigquery")).result()
    try:
        batch_load_json_rows(bq, staging_fqn, rows)

        # MERGE INTO target USING staging — built via sqlglot AST.
        update_cols = [c for c in cols if c not in merge_keys]
        target_alias, source_alias = "T", "S"
        target_tbl = exp.alias_(
            exp.to_table(target_table_fqn, dialect="bigquery"),
            alias=target_alias,
            table=True,
        )
        source_tbl = exp.alias_(
            exp.to_table(staging_fqn, dialect="bigquery"),
            alias=source_alias,
            table=True,
        )
        on_clause = exp.and_(
            *[
                exp.EQ(
                    this=exp.column(k, table=target_alias, quoted=True),
                    expression=exp.column(k, table=source_alias, quoted=True),
                )
                for k in merge_keys
            ]
        )
        when_clauses: list[exp.When] = []
        if update_cols:
            when_clauses.append(
                exp.When(
                    matched=True,
                    then=exp.Update(
                        expressions=[
                            exp.EQ(
                                this=exp.column(
                                    c, table=target_alias, quoted=True
                                ),
                                expression=exp.column(
                                    c, table=source_alias, quoted=True
                                ),
                            )
                            for c in update_cols
                        ]
                    ),
                )
            )
        when_clauses.append(
            exp.When(
                matched=False,
                then=exp.Insert(
                    this=exp.Tuple(
                        expressions=[
                            exp.to_identifier(c, quoted=True) for c in cols
                        ]
                    ),
                    expression=exp.Tuple(
                        expressions=[
                            exp.column(c, table=source_alias, quoted=True)
                            for c in cols
                        ]
                    ),
                ),
            )
        )
        if delete_unmatched:
            # Opt-in (full re-seed only): target rows absent from source are
            # stale — delete them atomically in the SAME MERGE. Default off,
            # so partial/single-source callers keep preserve-unmatched.
            when_clauses.append(
                exp.When(matched=False, source=True, then=exp.Delete())
            )
        merge_stmt = exp.Merge(
            this=target_tbl,
            using=source_tbl,
            on=on_clause,
            whens=exp.Whens(expressions=when_clauses),
        )
        bq.query(merge_stmt.sql(dialect="bigquery")).result()
    finally:
        # DROP TABLE IF EXISTS staging via sqlglot AST.
        drop_stmt = exp.Drop(
            this=exp.to_table(staging_fqn, dialect="bigquery"),
            kind="TABLE",
            exists=True,
        )
        bq.query(drop_stmt.sql(dialect="bigquery")).result()
