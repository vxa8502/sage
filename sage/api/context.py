"""
Request context management using contextvars.

Provides thread-safe request context propagation for logging and tracing.
Request ID set in middleware is accessible throughout the request lifecycle.
"""

from contextvars import ContextVar

# Request ID for correlation across logs
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


def get_request_id() -> str:
    """Get the current request ID from context."""
    return request_id_var.get()


def set_request_id(request_id: str) -> None:
    """Set the request ID in context."""
    request_id_var.set(request_id)
