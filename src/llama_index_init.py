"""
Helper file to initialize the LLM and embedding model settings for LlamaIndex

:author: Jonathan Decker
"""
# -----------------------------------------------------------------------------
# Reproducibility / scientific reporting notes
# - This module defines the *global* LlamaIndex Settings (LLM, embedding model,
#   and node parser). Any changes here affect all benchmark results.
# - For comparable runs, record:
#     * LLM_MODEL_NAME, EMBEDDING_MODEL_NAME
#     * endpoint/base URL + headers
#     * chunk_size / chunk_overlap
#     * temperature + max_tokens (generation settings)
# - Determinism:
#     * temperature=0 reduces sampling randomness, but exact determinism still
#       depends on the backend implementation and potential server-side changes.
#
# GitHub hygiene
# - Do not hard-code API keys. Use env vars (see constants.py) and pass through init().
# -----------------------------------------------------------------------------
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai_like import OpenAILike

from constants import (
    LLM_MODEL_NAME,
    LLM_API_ENDPOINT,
    EMBEDDING_MODEL_NAME,
    LLM_API_KEY,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP
)

from llama_index.core import Settings

from vllm_embeddings import VLLMEmbedding


def init(api_key: str = LLM_API_KEY, end_point: str = LLM_API_ENDPOINT, llm_model_name: str = LLM_MODEL_NAME,
         embedding_model_name: str = EMBEDDING_MODEL_NAME, temperature: int = 0, max_tokens: int = 200) -> Settings:

    """
        Initializes the global LlamaIndex 'Settings' object with the configured
        LLM, Embedding Model, and Node Parser.

        This function should be called once at application startup (e.g., in run_bench.py).

        Args:
            api_key (str): The API key for both the LLM and Embedding service.
            end_point (str): The base URL for the API service (e.g., 'https://.../v1/').
            llm_model_name (str): The model name for answer generation (LLM).
            embedding_model_name (str): The model name for creating text embeddings.
            temperature (float): The sampling temperature for the LLM (0.0 for deterministic answers).
            max_tokens (int): The maximum number of new tokens the LLM can generate.

        Returns:
            Settings: The globally configured LlamaIndex Settings object.
        """

    llm = OpenAILike(
        model=llm_model_name,
        is_chat_model=True,
        temperature=temperature,
        max_new_tokens=max_tokens,
        api_key=api_key,
        api_base=end_point,
        max_retries=0, # because retries/rate-limiting logic is handled separately in benchmark.py.
        request_timeout=180,  # seconds
        default_headers={"inference-service": llm_model_name},
    )

    embed_model = VLLMEmbedding(
        model=embedding_model_name,
        api_key=api_key,
        api_base=end_point,
        batch_size=32,
        max_concurrency=4,
        max_retries = 0
    )

#    Settings.node_parser = None

    Settings.node_parser = SentenceSplitter(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    )

    Settings.llm = llm
    Settings.embed_model = embed_model

    return Settings