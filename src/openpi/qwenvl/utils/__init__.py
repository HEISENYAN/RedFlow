"""
Utility modules for qwenvl package.
"""

__all__ = [
    "HDF5FeatureSchema",
    "ValueTokenizer",
    "convert_robotwin_run_to_lerobot",
    "convert_task_hdf5_to_lerobot",
    "infer_hdf5_feature_schema",
    "scan_task_hdf5_dirs",
]


def __getattr__(name: str):
    if name == "ValueTokenizer":
        from openpi.qwenvl.utils.value_tokenizer import ValueTokenizer

        return ValueTokenizer
    if name in {
        "HDF5FeatureSchema",
        "convert_robotwin_run_to_lerobot",
        "convert_task_hdf5_to_lerobot",
        "infer_hdf5_feature_schema",
        "scan_task_hdf5_dirs",
    }:
        from openpi.qwenvl.utils import convert_hdf5_to_lerobot as module

        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
