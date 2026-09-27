"""QwenVL value-function package for openpi."""

__all__ = ["ValueFunction"]


def __getattr__(name: str):
    if name == "ValueFunction":
        from openpi.qwenvl.value_function import ValueFunction

        return ValueFunction
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
