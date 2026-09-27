from __future__ import annotations

from openpi.qwenvl.data.data_processor_pair_map import make_supervised_pair_map_data_module
from openpi.qwenvl.data.data_processor_pi06_map import make_supervised_pi06_map_data_module


def build_training_data_module(
    processor,
    data_args,
    *,
    model_args=None,
    value_tokenizer=None,
    return_stats: bool = False,
):
    if getattr(data_args, "pair_mode", False):
        return make_supervised_pair_map_data_module(
            processor,
            data_args=data_args,
            model_args=model_args,
            return_stats=return_stats,
        )
    return make_supervised_pi06_map_data_module(
        processor,
        data_args=data_args,
        model_args=model_args,
        value_tokenizer=value_tokenizer,
        return_stats=return_stats,
    )
