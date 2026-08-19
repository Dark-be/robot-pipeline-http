from .hdf5_collector import ActHDF5Collector

COLLECTOR_REGISTRY = {
    "act_hdf5": ActHDF5Collector,
}


def get_collector(base_cfg):
    """Factory: build a collector from config using the type registry.

    Config expects a ``collector`` section with at least ``type``.
    Example::

        collector:
          type: act_hdf5
          save_dir: ./data/test/dual_alicia_teleop
          image_format: jpeg
    """
    cfg = base_cfg.get("collector")
    collector_type = cfg.get("type")

    if collector_type not in COLLECTOR_REGISTRY:
        available = list(COLLECTOR_REGISTRY.keys())
        raise ValueError(
            f"Unknown collector type '{collector_type}'. "
            f"Available: {available}"
        )

    cls = COLLECTOR_REGISTRY[collector_type]
    return cls(cfg)