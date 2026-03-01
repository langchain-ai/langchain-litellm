from importlib import metadata

from .chat_models import ChatLiteLLM, ChatLiteLLMRouter
from .document_loaders import LiteLLMOCRLoader
from .embeddings import LiteLLMEmbeddings, LiteLLMEmbeddingsRouter

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "ChatLiteLLM",
    "ChatLiteLLMRouter",
    "LiteLLMEmbeddings",
    "LiteLLMEmbeddingsRouter",
    "LiteLLMOCRLoader",
    "__version__",
]
