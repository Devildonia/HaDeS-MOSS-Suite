from . import moss_control_nodes, moss_enhancer_nodes, moss_image_saver_nodes, moss_loader_nodes, moss_lora_nodes, moss_merger_nodes, moss_metadata_nodes, moss_pipeline_nodes, moss_sampler_nodes, moss_saver_nodes, moss_telemetry_nodes

__version__ = "0.7.3"

# the Dev Tools (measurement nodes) live in the git repository only; the Comfy Registry package leaves them out
try:
    from . import moss_dev_nodes
except ImportError:
    moss_dev_nodes = None

_MODULES = (moss_loader_nodes, moss_merger_nodes, moss_saver_nodes, moss_lora_nodes, moss_metadata_nodes, moss_pipeline_nodes, moss_control_nodes, moss_enhancer_nodes, moss_sampler_nodes, moss_image_saver_nodes, moss_telemetry_nodes) + ((moss_dev_nodes,) if moss_dev_nodes else ())
NODE_CLASS_MAPPINGS = {k: v for m in _MODULES for k, v in m.NODE_CLASS_MAPPINGS.items()}
NODE_DISPLAY_NAME_MAPPINGS = {k: v for m in _MODULES for k, v in m.NODE_DISPLAY_NAME_MAPPINGS.items()}
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
