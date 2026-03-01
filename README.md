# langchain-litellm

<table>
<thead>
<tr>
<th align="center">📦 Distribution</th>
<th align="center">🔧 Project</th>
<th align="center">🚀 Activity</th>
</tr>
</thead>
<tbody>
<tr>
<td align="center">
<a href="https://pypi.org/project/langchain-litellm/">
<img src="https://img.shields.io/pypi/v/langchain-litellm?label=PyPI%20package&style=flat" alt="PyPI Package Version">
</a><br/>
<a href="https://pepy.tech/projects/langchain-litellm">
<img src="https://static.pepy.tech/personalized-badge/langchain-litellm?period=total&units=INTERNATIONAL_SYSTEM&left_color=GREY&right_color=BLUE&left_text=downloads" alt="PyPI Downloads">
</a><br/>
<a href="https://github.com/Akshay-Dongare/langchain-litellm/actions/workflows/pypi-release.yml">
</a>
<a href="https://opensource.org/licenses/MIT">
<img src="https://img.shields.io/badge/License-MIT-brightgreen.svg" alt="License: MIT">
</a>
</td>

<td align="center">
<img src="https://img.shields.io/badge/Platform-Linux%2C%20Windows%2C%20macOS-blue" alt="Platform">
<br>
<a href="https://www.python.org">
<img src="https://img.shields.io/badge/Python-3670A0?style=flat&logo=python&logoColor=ffdd54" alt="Python">
</a><br/>

<a href="https://github.com/astral-sh/uv">
<img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json" alt="uv">
</a>
</td>

<td align="center">
<img src="https://img.shields.io/github/issues-closed/Akshay-Dongare/langchain-litellm" alt="GitHub Issues Closed"><br/>
<img src="https://img.shields.io/github/issues/Akshay-Dongare/langchain-litellm" alt="GitHub Issues Open"><br/>
<img src="https://img.shields.io/github/issues-pr/Akshay-Dongare/langchain-litellm" alt="GitHub PRs Open"><br/>
<img src="https://img.shields.io/github/issues-pr-closed/Akshay-Dongare/langchain-litellm" alt="GitHub PRs Closed">
</td>
</tr>
</tbody>
</table>

## 🤔 What is this?

This package contains the LangChain integration with LiteLLM. [LiteLLM](https://github.com/BerriAI/litellm) is a library that simplifies calling Anthropic, Azure, Huggingface, Replicate, etc.

## 📖 Documentation

For conceptual guides, tutorials, and examples on using these classes, see the [LangChain Docs](https://docs.langchain.com/oss/python/integrations/providers/litellm).

### Advanced Features

<details>
<summary><strong>Embeddings</strong></summary>

Use `LiteLLMEmbeddings` to embed text across 100+ providers with a single, consistent interface. All configuration is explicit -- no environment variables required.

```python
from langchain_litellm import LiteLLMEmbeddings

embeddings = LiteLLMEmbeddings(
    model="openai/text-embedding-3-small",
    api_key="sk-...",
)

vectors = embeddings.embed_documents(["hello", "world"])
query_vector = embeddings.embed_query("hello")
```

Switch providers by changing `model` -- the interface stays the same:

```python
# Cohere
embeddings = LiteLLMEmbeddings(
    model="cohere/embed-english-v3.0",
    api_key="...",
    document_input_type="search_document",
    query_input_type="search_query",
)

# Azure OpenAI
embeddings = LiteLLMEmbeddings(
    model="azure/my-embedding-deployment",
    api_key="...",
    api_base="https://my-resource.openai.azure.com",
    api_version="2024-02-01",
)

# Bedrock
embeddings = LiteLLMEmbeddings(
    model="bedrock/amazon.titan-embed-text-v1",
)
```

For load-balancing across multiple deployments of the same model, use `LiteLLMEmbeddingsRouter`:

```python
from litellm import Router
from langchain_litellm import LiteLLMEmbeddingsRouter

router = Router(model_list=[
    {
        "model_name": "text-embedding-3-small",
        "litellm_params": {
            "model": "openai/text-embedding-3-small",
            "api_key": "sk-key1",
        },
    },
    {
        "model_name": "text-embedding-3-small",
        "litellm_params": {
            "model": "openai/text-embedding-3-small",
            "api_key": "sk-key2",
        },
    },
])

embeddings = LiteLLMEmbeddingsRouter(router=router)
```

</details>

<details>
<summary><strong>Vertex AI Grounding (Google Search)</strong></summary>

_Supported in v0.3.5+_

You can use Google Search grounding with Vertex AI models (e.g., `gemini-2.5-flash`). Citations and metadata are returned in `response_metadata` (Batch) or `additional_kwargs` (Streaming).

**Setup**

```python
import os
from langchain_litellm import ChatLiteLLM

os.environ["VERTEX_PROJECT"] = "your-project-id"
os.environ["VERTEX_LOCATION"] = "us-central1"

llm = ChatLiteLLM(model="vertex_ai/gemini-2.5-flash", temperature=0)
```

**Batch Usage**

```python
# Invoke with Google Search tool enabled
response = llm.invoke(
    "What is the current stock price of Google?",
    tools=[{"googleSearch": {}}]
)

# Access Citations & Metadata
provider_fields = response.response_metadata.get("provider_specific_fields")
if provider_fields:
    # Vertex returns a list; the first item contains the grounding info
    print(provider_fields[0])
```

**Streaming Usage**

```python
stream = llm.stream(
    "What is the current stock price of Google?",
    tools=[{"googleSearch": {}}]
)

for chunk in stream:
    print(chunk.content, end="", flush=True)
    # Metadata is injected into the chunk where it arrives
    if "provider_specific_fields" in chunk.additional_kwargs:
        print("\n[Metadata Found]:", chunk.additional_kwargs["provider_specific_fields"])
```

</details>

## 📕 Releases & Versioning

See our [Releases](https://docs.langchain.com/oss/python/release-policy) and [Versioning](https://docs.langchain.com/oss/python/versioning) policies.

## 💁 Contributing

As an open-source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infrastructure, or better documentation.

For detailed information on how to contribute, see the [Contributing Guide](https://docs.langchain.com/oss/python/contributing/overview).
