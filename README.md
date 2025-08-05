# LangChainGang

[![Tests](https://github.com/msabramo/langchaingang/workflows/Tests/badge.svg)](https://github.com/msabramo/langchaingang/actions?query=workflow%3ATests)

A unified interface for LangChain chat model providers that simplifies working with multiple LLM providers.

## Features

- **Unified API**: Single interface for multiple LangChain chat model providers
- **Lazy Loading**: Providers are only imported when needed
- **Graceful Fallbacks**: Missing dependencies don't break the entire package
- **Provider Discovery**: Easily list available providers
- **Standardized Parameters**: Consistent parameter handling across providers

## Supported Providers

- **OpenAI** (`openai`) - `ChatOpenAI`
- **Azure OpenAI** (`azure_openai`) - `AzureChatOpenAI`
- **AWS Bedrock** (`bedrock`) - `ChatBedrock`
- **Google Vertex AI** (`vertex`) - `ChatVertexAI`
- **Google Gemini** (`gemini`) - `ChatGoogleGenerativeAI`
- **Anthropic** (`anthropic`) - `ChatAnthropic`
- **Ollama** (`ollama`) - `ChatOllama`

## Installation

Install the base package:

```bash
uv add langchaingang
```

Install with specific provider support:

```bash
# OpenAI (and Azure OpenAI) support
uv add "langchaingang[openai]"

# AWS Bedrock support
uv add "langchaingang[aws]"

# Google Gemini and Vertex AI support
uv add "langchaingang[google]"

# Anthropic support
uv add "langchaingang[anthropic]"

# Ollama support
uv add "langchaingang[ollama]"

# Multiple providers
uv add "langchaingang[openai,anthropic,aws]"

# All providers
uv add "langchaingang[all]"
```

## Quick Start

```python
import langchaingang

# List available providers
providers = langchaingang.get_provider_list()
print(providers)  # ['openai', 'anthropic', 'ollama', ...]

# Get a chat model
model = langchaingang.get_chat_model(
    provider_name="openai", 
    model="gpt-4o-mini",
    api_key="your-api-key"
)

# Use the model
response = model.invoke("Hello, world!")
print(response.content)
```

## Usage Examples

### OpenAI

```python
model = langchaingang.get_chat_model(
    provider_name="openai",
    model="gpt-4o-mini",
    api_key="your-openai-key"
)
```

### Azure OpenAI

```python
model = langchaingang.get_chat_model(
    provider_name="azure_openai",
    model="gpt-4o-mini",
    azure_endpoint="https://your-resource.openai.azure.com/",
    api_key="your-azure-key",
    api_version="2024-02-01"
)
```

### AWS Bedrock

```python
model = langchaingang.get_chat_model(
    provider_name="bedrock",
    model="meta.llama3-2-70b-instruct-v1:0",  # Will be converted to model_id
    region_name="us-east-1"
)
```

### Google Vertex AI

```python
model = langchaingang.get_chat_model(
    provider_name="vertex",
    model="gemini-2.0-flash-001",  # Will be converted to model_name
    project="your-gcp-project"
)
```

### Anthropic

```python
model = langchaingang.get_chat_model(
    provider_name="anthropic",
    model="claude-sonnet-4-0",
    api_key="your-anthropic-key"
)
```

### Ollama

```python
model = langchaingang.get_chat_model(
    provider_name="ollama",
    model="llama3",
    base_url="http://localhost:11434"  # Optional, defaults to localhost:11434
)
```

## Parameter Handling

LangChainGang automatically handles provider-specific parameter differences:

- **Bedrock**: `model` parameter is converted to `model_id`
- **Vertex AI**: `model` parameter is converted to `model_name`
- **All others**: `model` parameter is passed through unchanged

## Error Handling

The package gracefully handles missing dependencies:

```python
# This won't fail even if langchain-openai isn't installed
providers = langchaingang.get_provider_list()

# This will raise ImportError if langchain-openai isn't installed
model = langchaingang.get_chat_model("openai", model="gpt-4o-mini")
```

## Development

Install development dependencies:

```bash
uv add "langchaingang[dev]"
```

Run tests:

```bash
pytest
```

Format code:

```bash
black langchaingang/
isort langchaingang/
```

Type checking:

```bash
mypy langchaingang/
```

## License

MIT License - see LICENSE file for details.