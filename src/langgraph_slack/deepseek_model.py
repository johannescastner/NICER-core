"""Thin compatibility seam for the pinned DeepSeek SDK's outgoing reasoning.

The maintained SDK records reasoning_content on AIMessage but its inherited
OpenAI request serializer drops it. Preserve the provider's original value;
synthetic framework assistant messages have no reasoning and use an empty field.
No reasoning is generated, logged or exposed as citizen-facing content here.
"""
from langchain_deepseek import ChatDeepSeek


class DeepSeekWithReasoning(ChatDeepSeek):
    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        messages = self._convert_input(input_).to_messages()
        payload = super()._get_request_payload(messages, stop=stop, **kwargs)
        # The SDK has already applied invocation overrides to instance defaults.
        # Explicit non-thinking requests must retain its unmodified serializer.
        extra = payload.get("extra_body")
        thinking = extra.get("thinking") if isinstance(extra, dict) else None
        if isinstance(thinking, dict) and thinking.get("type") == "disabled":
            return payload
        if len(payload["messages"]) != len(messages):
            raise RuntimeError("DeepSeek message serializer changed; inspect the SDK contract")
        for outgoing, original in zip(payload["messages"], messages):
            if outgoing.get("role") == "assistant":
                reasoning = (getattr(original, "additional_kwargs", None) or {}).get("reasoning_content")
                outgoing["reasoning_content"] = reasoning if isinstance(reasoning, str) else ""
        return payload
