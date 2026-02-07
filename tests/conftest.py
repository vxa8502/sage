"""Shared pytest fixtures for Sage tests."""

import pytest

from sage.core.models import ProductScore, RetrievedChunk


@pytest.fixture
def make_chunk():
    """Factory fixture for creating RetrievedChunk instances."""

    def _make_chunk(
        product_id: str = "P1",
        score: float = 0.85,
        rating: float = 4.5,
        text: str | None = None,
        review_id: str | None = None,
    ) -> RetrievedChunk:
        return RetrievedChunk(
            text=text or f"Review for {product_id}",
            score=score,
            product_id=product_id,
            rating=rating,
            review_id=review_id or f"rev_{product_id}",
        )

    return _make_chunk


@pytest.fixture
def make_product():
    """Factory fixture for creating ProductScore instances with evidence."""

    def _make_product(
        product_id: str = "P1",
        score: float = 0.85,
        n_chunks: int = 2,
        avg_rating: float = 4.5,
        text_len: int = 200,
    ) -> ProductScore:
        evidence = [
            RetrievedChunk(
                text="x" * text_len,
                score=score - i * 0.01,
                product_id=product_id,
                rating=avg_rating,
                review_id=f"rev_{i}",
            )
            for i in range(n_chunks)
        ]
        return ProductScore(
            product_id=product_id,
            score=score,
            chunk_count=n_chunks,
            avg_rating=avg_rating,
            evidence=evidence,
        )

    return _make_product


@pytest.fixture
def sample_chunk(make_chunk) -> RetrievedChunk:
    """A sample RetrievedChunk for simple tests."""
    return make_chunk(product_id="P1", score=0.9, rating=4.5, text="Good product")


@pytest.fixture
def sample_product(make_product) -> ProductScore:
    """A sample ProductScore for simple tests."""
    return make_product(product_id="P1", score=0.9, n_chunks=2, avg_rating=4.5)
