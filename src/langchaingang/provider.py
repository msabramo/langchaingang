from dataclasses import dataclass
from typing import Callable

from langchain_core.language_models.chat_models import BaseChatModel


@dataclass
class Info:
    provider_name: str
    model_type: type[BaseChatModel]


_provider_info_funcs_list: list[Callable[[], Info]] = []
_provider_dict: dict[str, type[BaseChatModel]] = {}


def register(func: Callable[[], Info]) -> Callable[[], Info]:
    """
    Decorator to register a provider

    Example:

    >>> @register
    ... def openai() -> Info:
    ...     from langchain_openai import ChatOpenAI
    ...     return Info(provider_name="openai", model_type=ChatOpenAI)

    """

    _provider_info_funcs_list.append(func)
    return func


def is_supported(provider_name: str) -> bool:
    """
    Check if a provider is supported.
    """

    return provider_name in _get_dict()


def get_chat_model_class(provider_name: str) -> type[BaseChatModel]:
    """
    Get the LangChain chat model class for a given provider name.
    """

    return _get_dict()[provider_name]


def get_list() -> list[str]:
    return list(_get_dict().keys())


def _get_dict() -> dict[str, type[BaseChatModel]]:
    """
    Get a dictionary that maps provider names to the corresponding types of
    their LangChain chat model classes.
    """

    if _provider_dict:
        return _provider_dict

    # Iterate through all the provider info functions
    for provider_info_func in _provider_info_funcs_list:
        try:
            provider_info = provider_info_func()
            _provider_dict[provider_info.provider_name] = provider_info.model_type
        except ImportError:
            pass

    return _provider_dict
