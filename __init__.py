from . import moss_enhancer_nodes, moss_image_saver_nodes, moss_loader_nodes, moss_lora_nodes, moss_merger_nodes, moss_metadata_nodes, moss_pipeline_nodes, moss_sampler_nodes, moss_saver_nodes, moss_telemetry_nodes

__version__ = "0.4.1"

_MODULES = (moss_loader_nodes, moss_merger_nodes, moss_saver_nodes, moss_lora_nodes, moss_metadata_nodes, moss_pipeline_nodes, moss_enhancer_nodes, moss_sampler_nodes, moss_image_saver_nodes, moss_telemetry_nodes)
NODE_CLASS_MAPPINGS = {k: v for m in _MODULES for k, v in m.NODE_CLASS_MAPPINGS.items()}
NODE_DISPLAY_NAME_MAPPINGS = {k: v for m in _MODULES for k, v in m.NODE_DISPLAY_NAME_MAPPINGS.items()}
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
