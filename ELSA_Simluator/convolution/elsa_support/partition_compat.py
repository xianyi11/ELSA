"""
Legacy pickle expects a top-level module ``partition`` (Node / Edge).
Re-export from ``mapping_exploration.partition``; see ``elsa_support.__init__``.
"""
from mapping_exploration.partition import Edge, Node  # noqa: F401

__all__ = ["Edge", "Node"]
