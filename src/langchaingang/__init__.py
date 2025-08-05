from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from . import provider

__version__ = "0.1.0"


@provider.register
def openai() -> provider.Info:
    from langchain_openai import ChatOpenAI

    return provider.Info(provider_name="openai", model_type=ChatOpenAI)


@provider.register
def azure_openai() -> provider.Info:
    from langchain_openai import AzureChatOpenAI

    return provider.Info(provider_name="azure_openai", model_type=AzureChatOpenAI)


@provider.register
def bedrock() -> provider.Info:
    from langchain_aws import ChatBedrock

    return provider.Info(provider_name="bedrock", model_type=ChatBedrock)


@provider.register
def vertex() -> provider.Info:
    from langchain_google_vertexai import ChatVertexAI

    return provider.Info(provider_name="vertex", model_type=ChatVertexAI)


@provider.register
def gemini() -> provider.Info:
    from langchain_google_genai import ChatGoogleGenerativeAI

    return provider.Info(provider_name="gemini", model_type=ChatGoogleGenerativeAI)


@provider.register
def anthropic() -> provider.Info:
    from langchain_anthropic import ChatAnthropic

    return provider.Info(provider_name="anthropic", model_type=ChatAnthropic)


@provider.register
def ollama() -> provider.Info:
    from langchain_ollama import ChatOllama

    return provider.Info(provider_name="ollama", model_type=ChatOllama)


def get_provider_list() -> list[str]:
    """
    Get a list of available provider names (e.g., 'openai', 'azure_openai',
    'bedrock', 'vertex', 'gemini', 'anthropic', 'ollama').
    """

    return provider.get_list()


def get_chat_model(provider_name: str, **kwargs: Any) -> BaseChatModel:
    """
    Factory function to return a LangChain-compatible chat model
    based on the provider name and configuration kwargs.

    Example:

    >>> get_chat_model(provider_name="openai", model="gpt-4o-mini")
    <langchain_openai.ChatOpenAI object at 0x7f8b4c177790>

    >>> get_chat_model(provider_name="azure_openai", model="gpt-4o-mini")
    <langchain_openai.AzureChatOpenAI object at 0x7f8b4c177790>

    >>> get_chat_model(
    ...     provider_name="bedrock", model="meta.llama3-2-70b-instruct-v1:0"
    ... )
    <langchain_aws.ChatBedrock object at 0x7f8b4c177790>

    >>> get_chat_model(provider_name="vertex", model="gemini-2.0-flash-001")
    <langchain_google_vertexai.ChatVertexAI object at 0x7f8b4c177790>

    >>> get_chat_model(provider_name="gemini", model="gemini-2.0-flash-001")
    <langchain_google_genai.ChatGoogleGenerativeAI object at 0x7f8b4c177790>

    >>> get_chat_model(provider_name="anthropic", model="claude-sonnet-4-0")
    <langchain_anthropic.ChatAnthropic object at 0x7f8b4c177790>

    >>> get_chat_model(provider_name="ollama", model="llama3")
    <langchain_ollama.ChatOllama object at 0x7f8b4c177790>
    """

    chat_model_class = provider.get_chat_model_class(provider_name)
    if chat_model_class is None:
        raise ValueError(f"Unsupported provider: {provider_name}")

    # Inject provider-specific defaults
    if provider_name == "bedrock":
        # AWS Bedrock uses model_id instead of model
        if "model" in kwargs:
            kwargs["model_id"] = kwargs.pop("model")

    if provider_name == "vertex":
        # Vertex AI uses model_name instead of model
        if "model" in kwargs:
            kwargs["model_name"] = kwargs.pop("model")

    return chat_model_class(**kwargs)
