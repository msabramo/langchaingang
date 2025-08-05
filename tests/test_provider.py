"""Tests for the provider module."""

from unittest.mock import Mock

from langchaingang import provider


def test_info_dataclass():
    """Test the Info dataclass."""
    mock_model_type = Mock()
    info = provider.Info(provider_name="test", model_type=mock_model_type)
    
    assert info.provider_name == "test"
    assert info.model_type == mock_model_type


def test_register_decorator():
    """Test the register decorator."""
    # Clear any existing registrations for test isolation
    provider._provider_info_funcs_list.clear()
    provider._provider_dict.clear()
    
    @provider.register
    def test_provider():
        return provider.Info(provider_name="test_provider", model_type=Mock())
    
    assert len(provider._provider_info_funcs_list) == 1
    assert provider._provider_info_funcs_list[0] == test_provider
    assert provider.get_list() == ["test_provider"]


def test_get_list():
    """Test getting the list of available providers."""
    # Clear any existing registrations for test isolation
    provider._provider_info_funcs_list.clear()
    provider._provider_dict.clear()
    
    @provider.register
    def test_provider1():
        return provider.Info(provider_name="test1", model_type=Mock())
    
    @provider.register
    def test_provider2():
        return provider.Info(provider_name="test2", model_type=Mock())
    
    provider_list = provider.get_list()
    assert "test1" in provider_list
    assert "test2" in provider_list
    assert len(provider_list) == 2


def test_is_supported():
    """Test checking if a provider is supported."""
    # Clear any existing registrations for test isolation
    provider._provider_info_funcs_list.clear()
    provider._provider_dict.clear()
    
    @provider.register
    def test_provider():
        return provider.Info(provider_name="supported_provider", model_type=Mock())
    
    assert provider.is_supported("supported_provider") is True
    assert provider.is_supported("unsupported_provider") is False


def test_get_chat_model_class():
    """Test getting a chat model class."""
    # Clear any existing registrations for test isolation
    provider._provider_info_funcs_list.clear()
    provider._provider_dict.clear()
    
    mock_model_type = Mock()
    
    @provider.register
    def test_provider():
        return provider.Info(provider_name="test_provider", model_type=mock_model_type)
    
    result = provider.get_chat_model_class("test_provider")
    assert result == mock_model_type


def test_import_error_handling():
    """Test that ImportError is handled gracefully."""
    # Clear any existing registrations for test isolation
    provider._provider_info_funcs_list.clear()
    provider._provider_dict.clear()
    
    @provider.register
    def failing_provider():
        raise ImportError("Mock import failure")
    
    @provider.register
    def working_provider():
        return provider.Info(provider_name="working", model_type=Mock())
    
    # The failing provider should not appear in the list
    provider_list = provider.get_list()
    assert "working" in provider_list
    assert len(provider_list) == 1