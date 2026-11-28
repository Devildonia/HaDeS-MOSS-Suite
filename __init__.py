from . import moss_compare_nodes, moss_control_nodes, moss_depth_nodes, moss_enhancer_nodes, moss_image_saver_nodes, moss_loader_nodes, moss_lora_nodes, moss_merger_nodes, moss_metadata_nodes, moss_pipeline_nodes, moss_sampler_nodes, moss_saver_nodes, moss_telemetry_nodes, moss_textcheck_nodes

__version__ = "1.0.0"

# the Dev Tools (measurement nodes) live in the git repository only; the Comfy Registry package leaves them out
import logging
import os

# absent in the Registry package (expected, silent); present but broken in a git checkout: logged with its traceback (audit 0.8.1)
# (a name bound in the package first would make `from . import moss_dev_nodes` return it instead of importing)
_dev_nodes = None
if os.path.exists(os.path.join(os.path.dirname(__file__), "moss_dev_nodes.py")):
    try:
        from . import moss_dev_nodes as _dev_nodes
    except Exception:  # the rest of the pack still loads; the log says why the two Dev Tools are missing
        logging.getLogger(__name__).exception("HaDeS MOSS Suite: the Dev Tools (moss_dev_nodes.py) failed to import")

_MODULES = (moss_loader_nodes, moss_merger_nodes, moss_saver_nodes, moss_lora_nodes, moss_metadata_nodes, moss_pipeline_nodes, moss_control_nodes, moss_depth_nodes, moss_enhancer_nodes, moss_sampler_nodes, moss_image_saver_nodes, moss_compare_nodes, moss_textcheck_nodes, moss_telemetry_nodes) + ((_dev_nodes,) if _dev_nodes else ())
NODE_CLASS_MAPPINGS = {k: v for m in _MODULES for k, v in m.NODE_CLASS_MAPPINGS.items()}
NODE_DISPLAY_NAME_MAPPINGS = {k: v for m in _MODULES for k, v in m.NODE_DISPLAY_NAME_MAPPINGS.items()}
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
