"""Batch embedding pipeline for Axon knowledge graphs.

Takes a :class:`KnowledgeGraph`, generates natural-language descriptions for
each embeddable symbol node, encodes them using the configured embedding
provider, and returns a list of :class:`NodeEmbedding` objects ready for storage.

Supports multiple embedding backends:
- FastEmbed (default): Local embedding using fastembed library
- Azure OpenAI via UAIS: Cloud-based embedding using UHG's UAIS authentication

Configure via environment variables:
- AXON_EMBEDDING_PROVIDER: "fastembed" (default) or "azure_openai"
- UAIS_CLIENT_ID / UAIS_CLIENT_SECRET: Required for azure_openai provider

Only code-level symbol nodes are embedded.  Structural nodes (Folder,
Community, Process) are deliberately skipped — they lack the semantic
richness that makes embedding worthwhile.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Iterator, Protocol

from axon.config.embeddings import EmbeddingProvider, get_embedding_config
from axon.core.embeddings.text import build_class_method_index, generate_text
from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import NodeLabel
from axon.core.storage.base import NodeEmbedding

if TYPE_CHECKING:
    from fastembed import TextEmbedding

    from axon.core.embeddings.azure_openai import AzureOpenAIEmbedder

logger = logging.getLogger(__name__)


class EmbedderProtocol(Protocol):
    """Protocol for embedding providers."""

    def embed(self, texts: list[str], batch_size: int = 64) -> Iterator:
        """Generate embeddings for texts."""
        ...


@lru_cache(maxsize=4)
def _get_fastembed_model(model_name: str) -> "TextEmbedding":
    """Get a cached FastEmbed model instance."""
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=model_name)


def _get_model(model_name: str | None = None) -> "TextEmbedding | AzureOpenAIEmbedder":
    """Get the appropriate embedding model based on configuration.

    Args:
        model_name: Optional model name override for FastEmbed.
            Ignored when using Azure OpenAI provider.

    Returns:
        An embedding model instance (FastEmbed or Azure OpenAI).
    """
    config = get_embedding_config()

    if config.provider == EmbeddingProvider.AZURE_OPENAI:
        from axon.core.embeddings.azure_openai import get_azure_embedder

        return get_azure_embedder(config)

    # Default to FastEmbed
    effective_model = model_name or config.fastembed_model
    return _get_fastembed_model(effective_model)

# Labels worth embedding — skip Folder, Community, Process (structural only).
EMBEDDABLE_LABELS: frozenset[NodeLabel] = frozenset(
    {
        NodeLabel.FILE,
        NodeLabel.FUNCTION,
        NodeLabel.CLASS,
        NodeLabel.METHOD,
        NodeLabel.INTERFACE,
        NodeLabel.TYPE_ALIAS,
        NodeLabel.ENUM,
    }
)

def _to_list(vector: Any) -> list[float]:
    """Convert embedding vector to a plain Python list.

    Handles both numpy arrays (from FastEmbed) and lists (from Azure OpenAI).

    Args:
        vector: Embedding vector (numpy array or list).

    Returns:
        Plain Python list of floats.
    """
    if hasattr(vector, "tolist"):
        return vector.tolist()
    return list(vector)


def embed_graph(
    graph: KnowledgeGraph,
    model_name: str = "BAAI/bge-small-en-v1.5",
    batch_size: int = 64,
) -> list[NodeEmbedding]:
    """Generate embeddings for all embeddable nodes in the graph.

    Uses the configured embedding provider (FastEmbed or Azure OpenAI).
    Each embeddable node is converted to a natural-language description
    via :func:`generate_text`, then embedded in batches.

    Args:
        graph: The knowledge graph whose nodes should be embedded.
        model_name: The fastembed model identifier (ignored for Azure OpenAI).
            Defaults to ``"BAAI/bge-small-en-v1.5"``.
        batch_size: Number of texts to encode per batch.  Defaults to 64.

    Returns:
        A list of :class:`NodeEmbedding` instances, one per embeddable node,
        each carrying the node's ID and its embedding vector as a plain
        Python ``list[float]``.
    """
    nodes = [n for n in graph.iter_nodes() if n.label in EMBEDDABLE_LABELS]

    if not nodes:
        return []

    class_method_idx = build_class_method_index(graph)
    texts = [generate_text(node, graph, class_method_idx) for node in nodes]

    model = _get_model(model_name)
    vectors = list(model.embed(texts, batch_size=batch_size))

    results: list[NodeEmbedding] = []
    for node, vector in zip(nodes, vectors):
        results.append(
            NodeEmbedding(
                node_id=node.id,
                embedding=_to_list(vector),
            )
        )

    return results


def embed_nodes(
    graph: KnowledgeGraph,
    node_ids: set[str],
    model_name: str = "BAAI/bge-small-en-v1.5",
    batch_size: int = 64,
) -> list[NodeEmbedding]:
    """Like :func:`embed_graph`, but only for the given *node_ids*."""
    if not node_ids:
        return []

    nodes = [graph.get_node(nid) for nid in node_ids]
    nodes = [n for n in nodes if n is not None and n.label in EMBEDDABLE_LABELS]

    if not nodes:
        return []

    class_method_idx = build_class_method_index(graph)
    model = _get_model(model_name)

    texts = [generate_text(n, graph, class_method_idx) for n in nodes]
    embeddings: list[NodeEmbedding] = []
    for node, vector in zip(nodes, model.embed(texts, batch_size=batch_size)):
        embeddings.append(
            NodeEmbedding(
                node_id=node.id,
                embedding=_to_list(vector),
            )
        )

    return embeddings
