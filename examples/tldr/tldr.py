#!/usr/bin/env -S uv run --script
#
# Environment variables
# OpenAI:
# - OPENAI_BASE_URL
# - OPENAI_API_KEY
# Azure OpenAI:
# - AZURE_OPENAI_ENDPOINT
# - AZURE_OPENAI_API_KEY
# - AZURE_OPENAI_API_VERSION
# AWS Bedrock:
# - AWS_ACCESS_KEY_ID
# - AWS_SECRET_ACCESS_KEY
# - AWS_DEFAULT_REGION
# Google Vertex AI:
# - GOOGLE_APPLICATION_CREDENTIALS or default ADC
# - GOOGLE_CLOUD_PROJECT
# Google Gemini:
# - GOOGLE_API_KEY

import argparse
import urllib.request

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Import our chat model factory
from langchaingang import get_chat_model, get_provider_list

provider_list = get_provider_list()

parser = argparse.ArgumentParser(
    description="Summarize a document using various LLM providers via LangChain",
)
parser.add_argument(
    "--env-file",
    help="Read in a file of environment variables",
)
parser.add_argument(
    "filename_or_url",
    help="Path to the document file or URL to summarize",
)
parser.add_argument(
    "--model",
    default="gpt-4o-mini",
    help="Model to use (default: gpt-4o-mini)",
)
parser.add_argument(
    "--provider",
    choices=provider_list,
    default="openai",
    help=f"LLM provider to use (available: {', '.join(provider_list)})",
)
args = parser.parse_args()

if args.env_file:
    load_dotenv(args.env_file)

# Read document from file or URL
if args.filename_or_url.startswith(("http://", "https://")):
    # Read from URL
    with urllib.request.urlopen(args.filename_or_url) as response:
        document = response.read().decode('utf-8')
else:
    # Read from file
    with open(args.filename_or_url, "r") as f:
        document = f.read()

# Check if any providers are available
if not provider_list:
    raise RuntimeError("No LLM providers are available. Please check your dependencies.")

# Create LLM using the chat model factory
llm = get_chat_model(
    provider_name=args.provider,
    model=args.model,
    temperature=0,
)

# Create a prompt template
prompt = ChatPromptTemplate.from_messages([
    ("user", "Please summarize this document in a single sentence:\n\n{document}")
])

# Create the chain
chain = prompt | llm | StrOutputParser()

# Run the chain
response = chain.invoke({"document": document})
print(response)
