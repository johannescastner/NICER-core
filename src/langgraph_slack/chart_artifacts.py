"""Private previews of PNGs attested by saved chart tool receipts."""
from urllib.parse import urlsplit
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.messages.utils import convert_to_messages
from pro.tools.superset.ops import ToolEnvelope

MAX_PNG_BYTES = 16 * 1024 * 1024
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def chart_object(uri: str, bucket_name: str) -> str:
    parsed = urlsplit(uri)
    if (parsed.scheme != "gs" or parsed.netloc != bucket_name
            or parsed.query or parsed.fragment
            or not parsed.path.startswith("/dashboards/")
            or not parsed.path.endswith(".png")):
        raise ValueError("Invalid chart artifact")
    return parsed.path.lstrip("/")


def approved_chart_uris(messages, bucket_name: str) -> list[str]:
    """Only actual successful chart-tool results authorize storage reads."""
    calls = {}
    approved = []
    for raw in messages:
        try:
            message = convert_to_messages([raw])[0]
        except (ValueError, TypeError, KeyError):
            continue
        if isinstance(message, AIMessage):
            calls.update({call["id"]: call["name"] for call in message.tool_calls})
        if not isinstance(message, ToolMessage):
            continue
        if calls.get(message.tool_call_id) != "superset_tool":
            continue
        try:
            receipt = ToolEnvelope.model_validate_json(message.text())
            uri = receipt.superset_receipt.links.get("chart_png_uri")
            if not receipt.ok or not uri:
                continue
            chart_object(uri, bucket_name)
        except (ValueError, TypeError):
            continue
        if uri not in approved:
            approved.append(uri)
    return approved


def read_chart_png(uri: str, bucket_name: str, storage_client) -> bytes:
    blob = storage_client.bucket(bucket_name).blob(chart_object(uri, bucket_name))
    blob.reload()
    if blob.size is None or blob.size > MAX_PNG_BYTES:
        raise ValueError("Chart artifact is too large")
    data = blob.download_as_bytes(if_generation_match=blob.generation, end=MAX_PNG_BYTES)
    if len(data) > MAX_PNG_BYTES or not data.startswith(PNG_SIGNATURE):
        raise ValueError("Chart artifact is not a PNG")
    return data
