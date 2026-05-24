from rlm_paged.client.base import GenerationResult, LLMClient

__all__ = [
    "GenerationResult",
    "LLMClient",
    "build_client",
]


def build_client(spec: str, **kwargs) -> LLMClient:
    """Construct a client from a spec string like 'anthropic:claude-opus-4-7'.

    Kwargs are passed to the concrete client constructor (e.g. thinking_budget).
    """
    if ":" not in spec:
        raise ValueError(f"client spec must be 'provider:model', got: {spec!r}")
    provider, model = spec.split(":", 1)
    if provider == "anthropic":
        from rlm_paged.client.anthropic import AnthropicClient
        return AnthropicClient(model, **kwargs)
    if provider == "openai":
        from rlm_paged.client.openai import OpenAIClient
        return OpenAIClient(model, **kwargs)
    if provider == "gemini":
        from rlm_paged.client.gemini import GeminiClient
        return GeminiClient(model, **kwargs)
    if provider == "together":
        from rlm_paged.client.together import TogetherClient
        return TogetherClient(model, **kwargs)
    raise ValueError(f"unknown provider: {provider!r}")
