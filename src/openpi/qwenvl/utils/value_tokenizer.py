import os
from typing import List
from typing import Union

import numpy as np
from transformers import AutoTokenizer


class ValueTokenizer:
    def __init__(
        self, llm_path: str,
        bins: int = 201, min_value: float = -1.0, max_value: float = 0.0
    ) -> None:
        """
        Discretizes continuous values into N bins per dimension and maps to the least used tokens.

        NOTE =>> by default, assumes a BPE-style tokenizer akin to the LlamaTokenizer, where *the least used tokens*
                 appear at the end of the vocabulary!

        :param llm_path: LLM path to find tokenizer.
        :param bins: Number of bins for each continuous value; we'll adopt a uniform binning strategy.
        :param min_value: Minimum value (for clipping, setting lower bound on bin interval).
        :param max_value: Maximum value (for clipping, setting upper bound on bin interval).
        """
        # Load the tokenizer locally when available, avoiding a network download.
        local_files_only = os.path.exists(llm_path) if os.path.isabs(llm_path) or not llm_path.startswith(('http://', 'https://')) else False
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_path,
            local_files_only=local_files_only
        )  # Use AutoTokenizer for Qwen2.5-VL compatibility.
        self.n_bins, self.min_value, self.max_value = bins, min_value, max_value

        # Create uniform bins and compute bin centers.
        self.bin_edges = np.linspace(min_value, max_value, self.n_bins + 1)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2.0

        self.vocab_size = len(self.tokenizer)

        self.extra_id_token_ids = []
        for i in range(self.n_bins):
            token_name = f"<extra_id_{i}>"
            tid = self.tokenizer.convert_tokens_to_ids(token_name)
            if tid is None or tid == self.tokenizer.unk_token_id:
                raise ValueError(f"Could not find token {token_name} in tokenizer. "
                               f"Make sure the model was trained with extra_id tokens added via add_token.py")
            self.extra_id_token_ids.append(int(tid))
        self.extra_id_token_ids = np.array(self.extra_id_token_ids)
        self.token_id_to_bin_idx = {int(token_id): int(bin_idx) for bin_idx, token_id in enumerate(self.extra_id_token_ids)}

    def __call__(self, value: np.ndarray) -> Union[str, List[str]]:
        """
        Clip and bin values into <extra_id_{i}> tokens.
        """
        value = np.asarray(value)
        value = np.clip(value, a_min=float(self.min_value), a_max=float(self.max_value))
        discretized_value = np.digitize(value, self.bin_edges)
        bin_indices = np.clip(discretized_value - 1, 0, self.n_bins - 1)
        token_ids = self.extra_id_token_ids[bin_indices]

        if value.ndim == 0 or (value.ndim == 1 and value.shape[0] == 1):
            token_id = int(token_ids.item() if token_ids.ndim > 0 else token_ids)
            return self.tokenizer.decode([token_id], skip_special_tokens=False)
        return self.tokenizer.batch_decode(token_ids.astype(int).tolist(), skip_special_tokens=False)

    def decode_token_ids_to_values(self, value_token_ids: np.ndarray) -> np.ndarray:
        """
        Returns continuous values for discrete value token IDs.

        Token IDs correspond to <extra_id_{i}> tokens where i is the bin index.
        We convert back to bin indices [0, n_bins-1] and then to bin centers.
        """
        bin_indices = []
        for tid in value_token_ids:
            token_id = int(tid)
            bin_idx = self.token_id_to_bin_idx.get(token_id)
            if bin_idx is None:
                token_str = self.tokenizer.decode([tid])
                raise ValueError(f"Token ID {tid} ({token_str}) is not a valid extra_id token. "
                               f"Expected one of: {self.extra_id_token_ids}")
            bin_indices.append(bin_idx)

        bin_indices = np.array(bin_indices)
        return self.bin_centers[bin_indices]
