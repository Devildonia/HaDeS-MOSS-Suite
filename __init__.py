from . import moss_merger_nodes, moss_saver_nodes

__version__ = "0.1.0"

NODE_CLASS_MAPPINGS = {**moss_merger_nodes.NODE_CLASS_MAPPINGS, **moss_saver_nodes.NODE_CLASS_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**moss_merger_nodes.NODE_DISPLAY_NAME_MAPPINGS, **moss_saver_nodes.NODE_DISPLAY_NAME_MAPPINGS}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
