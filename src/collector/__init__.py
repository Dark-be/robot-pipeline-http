from .mcap_collector import ActMcapCollector

# 注：HDF5 collector（act_hdf5）已移除（h5py 依赖一并移除），当前仅支持 act_mcap。

COLLECTOR_REGISTRY = {
    "act_mcap": ActMcapCollector,
}


def get_collector(base_cfg):
    """Factory: build a collector from config using the type registry.

    入参支持两种形式：
    - 完整 base 配置（含 ``collector`` 子段），或
    - 直接传入 collector 子配置（含 ``type``）。
    无论哪种，都要求配置里有 ``type``。

    Example::

        collector:
          type: act_mcap
          save_dir: ./data/test/dual_alicia_teleop
          image_format: jpeg

    注：``act_hdf5``（HDF5 collector）暂不可用，当前仅支持 ``act_mcap``。
    """
    if isinstance(base_cfg, dict) and "collector" in base_cfg:
        cfg = base_cfg["collector"]
    else:
        cfg = base_cfg or {}
    collector_type = cfg.get("type")

    if collector_type not in COLLECTOR_REGISTRY:
        available = list(COLLECTOR_REGISTRY.keys())
        raise ValueError(
            f"Unknown collector type '{collector_type}'. "
            f"Available: {available}"
        )

    cls = COLLECTOR_REGISTRY[collector_type]
    return cls(cfg)