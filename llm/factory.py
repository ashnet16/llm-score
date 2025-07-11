from typing import Any

from llm.base import BaseLLM
from llm.gpt.llm import ChatGPTLLM


class LLMFactory:

    @staticmethod
    def create(provider: str, connection_info: Any) -> BaseLLM:
        provider = provider.lower()

        if provider == "openai" or provider == "chatgpt" or provider == "chatgpt-llm":
            client = ChatGPTLLM()
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

        client.connect(connection_info)
        return client
