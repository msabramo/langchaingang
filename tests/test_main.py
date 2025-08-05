"""Tests for the main module functionality."""

from unittest.mock import Mock, patch

import pytest

import langchaingang


def test_version():
    """Test that version is properly defined."""
    assert hasattr(langchaingang, "__version__")
    assert langchaingang.__version__ == "0.1.0"


def test_get_provider_list():
    """Test the get_provider_list function."""
    # This will use the actual registered providers
    provider_list = langchaingang.get_provider_list()
    assert isinstance(provider_list, list)
    # We expect at least some providers to be registered
    expected_providers = {
        "openai",
        "azure_openai",
        "bedrock",
        "vertex",
        "gemini",
        "anthropic",
        "ollama",
    }
    # Some might be missing due to import errors, but the list should be a
    # subset
    assert set(provider_list).issubset(expected_providers)


def test_get_chat_model_unsupported_provider():
    """Test error handling for unsupported provider."""
    with pytest.raises(ValueError, match="Unsupported provider: nonexistent"):
        langchaingang.get_chat_model("nonexistent")


@patch("langchaingang.provider.get_chat_model_class")
@patch("langchaingang.provider.is_supported")
def test_get_chat_model_bedrock_parameter_conversion(mock_is_supported, mock_get_class):
    """Test that Bedrock model parameter is converted to model_id."""
    mock_is_supported.return_value = True
    mock_chat_model = Mock()
    mock_get_class.return_value = mock_chat_model

    langchaingang.get_chat_model("bedrock", model="test-model", other_param="value")

    # Verify the mock was called with model_id instead of model
    mock_chat_model.assert_called_once_with(model_id="test-model", other_param="value")


@patch("langchaingang.provider.get_chat_model_class")
@patch("langchaingang.provider.is_supported")
def test_get_chat_model_vertex_parameter_conversion(mock_is_supported, mock_get_class):
    """Test that Vertex AI model parameter is converted to model_name."""
    mock_is_supported.return_value = True
    mock_chat_model = Mock()
    mock_get_class.return_value = mock_chat_model

    langchaingang.get_chat_model("vertex", model="test-model", other_param="value")

    # Verify the mock was called with model_name instead of model
    mock_chat_model.assert_called_once_with(
        model_name="test-model", other_param="value"
    )


@patch("langchaingang.provider.get_chat_model_class")
@patch("langchaingang.provider.is_supported")
def test_get_chat_model_standard_parameters(mock_is_supported, mock_get_class):
    """Test that other providers use standard parameters."""
    mock_is_supported.return_value = True
    mock_chat_model = Mock()
    mock_get_class.return_value = mock_chat_model

    langchaingang.get_chat_model("openai", model="test-model", api_key="key")

    # Verify the mock was called with original parameters
    mock_chat_model.assert_called_once_with(model="test-model", api_key="key")


@patch("langchaingang.provider.get_chat_model_class")
@patch("langchaingang.provider.is_supported")
def test_get_chat_model_ollama_parameters(mock_is_supported, mock_get_class):
    """Test that Ollama provider uses standard parameters."""
    mock_is_supported.return_value = True
    mock_chat_model = Mock()
    mock_get_class.return_value = mock_chat_model

    langchaingang.get_chat_model(
        "ollama", model="llama3", base_url="http://localhost:11434"
    )

    # Verify the mock was called with original parameters (no conversion)
    mock_chat_model.assert_called_once_with(
        model="llama3", base_url="http://localhost:11434"
    )
