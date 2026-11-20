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
        choice = payload.get("tool_choice")
        forced = choice == "required" or (
            isinstance(choice, dict) and choice.get("type") == "function"
        )
        if payload.get("tools") and forced:
            # DeepSeek rejects required/named tool choices in thinking mode.
            # Keep the caller's forced-tool contract, using the provider's native
            # non-thinking mode for this invocation only. Never mutate defaults
            # on the shared model or weaken a clarification-only tool registry.
            extra = dict(extra or {})
            extra["thinking"] = {"type": "disabled"}
            payload["extra_body"] = extra
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
