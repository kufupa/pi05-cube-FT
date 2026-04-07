"""Compatibility shim for runtime environments with legacy typing_extensions metadata."""

try:
    import typing_extensions
except Exception:
    typing_extensions = None

if typing_extensions is not None and not hasattr(typing_extensions, "__version__"):
    typing_extensions.__version__ = "0.0.0"

