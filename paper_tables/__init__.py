from importlib import import_module

from paper_tables.common import Context, Panel, render

__all__ = ["TABLES", "Context", "Panel", "load", "render"]
TABLES = {"t1": "table_1", "t2": "table_2", **{f"ed{i}": f"ed_{i:02d}" for i in range(1, 29)}}


def load(key):
    return import_module(f"paper_tables.{TABLES[key]}")
