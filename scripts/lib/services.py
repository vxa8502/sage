"""Shared service initialization for scripts."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sage.adapters.hhem import HallucinationDetector
    from sage.services.explanation import Explainer


def get_explanation_services() -> tuple[Explainer, HallucinationDetector]:
    """Initialize Explainer and HallucinationDetector.

    Centralizes the common pattern of creating both services together.
    Import is deferred to avoid loading heavy models until needed.

    Returns:
        Tuple of (Explainer, HallucinationDetector) instances.
    """
    from sage.adapters.hhem import HallucinationDetector
    from sage.services.explanation import Explainer

    return Explainer(), HallucinationDetector()
