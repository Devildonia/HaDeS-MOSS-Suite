from . import moss_lora_nodes, moss_merger_nodes, moss_saver_nodes

__version__ = "0.1.0"

_MODULES = (moss_merger_nodes, moss_saver_nodes, moss_lora_nodes)
NODE_CLASS_MAPPINGS = {k: v for m in _MODULES for k, v in m.NODE_CLASS_MAPPINGS.items()}
NODE_DISPLAY_NAME_MAPPINGS = {k: v for m in _MODULES for k, v in m.NODE_DISPLAY_NAME_MAPPINGS.items()}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
