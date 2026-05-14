"""Integration tests for LiteLLMOCRLoader.

These tests require a running LiteLLM proxy with OCR configured.
Set LITELLM_PROXY_URL and optionally LITELLM_API_KEY environment variables.
"""

import os

import pytest

from langchain_litellm.document_loaders import LiteLLMOCRLoader

# Skip tests if proxy is not configured
PROXY_URL = os.getenv("LITELLM_PROXY_URL")
PROXY_API_KEY = os.getenv("LITELLM_API_KEY")
PROXY_MODEL = os.getenv("LITELLM_OCR_MODEL", "azure-document")

pytestmark = pytest.mark.skipif(
    not PROXY_URL, reason="LITELLM_PROXY_URL not set. Set it to run integration tests."
)


@pytest.mark.integration
class TestLiteLLMOCRLoaderIntegration:
    """Integration tests with real LiteLLM proxy."""

    def test_load_from_url(self) -> None:
        """Test loading document from URL."""
        # Using a public test PDF
        test_url = "https://arxiv.org/pdf/1706.03762"  # Attention Is All You Need paper

        loader = LiteLLMOCRLoader(
            proxy_base_url=PROXY_URL,  # type: ignore
            api_key=PROXY_API_KEY,
            url_path=test_url,
            model=PROXY_MODEL,
            mode="page",
        )

        documents = loader.load()

        # Verify we got documents
        assert len(documents) > 0

        # Verify first document has content
        assert len(documents[0].page_content) > 0

        # Verify metadata
        assert "page" in documents[0].metadata
        assert documents[0].metadata["source"] == test_url

        print(f"Loaded {len(documents)} pages from {test_url}")
        print(f"First page preview: {documents[0].page_content[:200]}...")

    def test_load_single_mode(self) -> None:
        """Test loading document in single mode."""
        test_url = "https://arxiv.org/pdf/1706.03762"

        loader = LiteLLMOCRLoader(
            proxy_base_url=PROXY_URL,  # type: ignore
            api_key=PROXY_API_KEY,
            url_path=test_url,
            model=PROXY_MODEL,
            mode="single",
        )

        documents = loader.load()

        # Should get exactly one document
        assert len(documents) == 1

        # Should have total_pages in metadata
        assert "total_pages" in documents[0].metadata
        assert documents[0].metadata["total_pages"] > 0

        print(
            f"Loaded single document with {documents[0].metadata['total_pages']} pages"
        )

    @pytest.mark.asyncio
    async def test_aload_from_url(self) -> None:
        """Test async loading document from URL."""
        test_url = "https://arxiv.org/pdf/1706.03762"

        loader = LiteLLMOCRLoader(
            proxy_base_url=PROXY_URL,  # type: ignore
            api_key=PROXY_API_KEY,
            url_path=test_url,
            model=PROXY_MODEL,
            mode="page",
        )

        documents = await loader.aload()

        # Verify we got documents
        assert len(documents) > 0
        assert len(documents[0].page_content) > 0

        print(f"Async loaded {len(documents)} pages")

    def test_lazy_load(self) -> None:
        """Test lazy loading documents."""
        test_url = "https://arxiv.org/pdf/1706.03762"

        loader = LiteLLMOCRLoader(
            proxy_base_url=PROXY_URL,  # type: ignore
            api_key=PROXY_API_KEY,
            url_path=test_url,
            model=PROXY_MODEL,
            mode="page",
        )

        # Consume iterator
        documents = []
        for doc in loader.lazy_load():
            documents.append(doc)
            # Verify each document as we iterate
            assert len(doc.page_content) > 0
            assert "page" in doc.metadata

        assert len(documents) > 0
        print(f"Lazy loaded {len(documents)} pages")


# Additional test to verify error handling with wrong proxy URL
@pytest.mark.integration
def test_load_with_invalid_proxy() -> None:
    """Test that invalid proxy URL raises appropriate error."""
    loader = LiteLLMOCRLoader(
        proxy_base_url="http://nonexistent-proxy.example.com:9999",
        url_path="https://example.com/doc.pdf",
        model="test-model",
    )

    with pytest.raises(RuntimeError, match="Failed to connect"):
        loader.load()
