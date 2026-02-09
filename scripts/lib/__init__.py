"""Shared utilities for scripts."""

# Re-export from sage.services for backwards compatibility
from sage.services import get_explanation_services

__all__ = ["get_explanation_services"]
