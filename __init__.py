# AnyTile 图像切分放大插件 (bofengttp)
from .anytile_nodes import (
    AnyTileSplit,
    AnyTileBatchSampler,
    AnyTileBatchSamplerCustom,
    AnyTileMerge,
    AnyTilePerTileTextEncode,
    AnyTileSeamDebugPreview,
    NODE_CLASS_MAPPINGS as ANYTILE_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as ANYTILE_DISPLAY,
)

NODE_CLASS_MAPPINGS = {
    **ANYTILE_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **ANYTILE_DISPLAY,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
