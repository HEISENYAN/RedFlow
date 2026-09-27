from __future__ import annotations

import os

import torch


def resolve_dataset_paths(dataset_use: str) -> list[str]:
    dataset_paths = [path.strip() for path in dataset_use.split(",") if path.strip()]
    if not dataset_paths:
        raise ValueError("dataset_use is empty")
    resolved_paths = []
    for path in dataset_paths:
        if os.path.exists(path):
            resolved_paths.append(os.path.abspath(path))
            continue
        repo_relative = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../", path))
        if os.path.exists(repo_relative):
            resolved_paths.append(repo_relative)
            continue
        raise FileNotFoundError(f"Dataset path not found: {path}")
    return resolved_paths


def sync_min_length(local_len: int) -> int:
    if not torch.distributed.is_initialized():
        return int(local_len)
    backend = torch.distributed.get_backend()
    device = torch.device("cuda", torch.cuda.current_device()) if backend == "nccl" else torch.device("cpu")
    tensor = torch.tensor([int(local_len)], dtype=torch.long, device=device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MIN)
    return int(tensor.item())


def sync_min_length_list(local_lens: list[int]) -> list[int]:
    if not torch.distributed.is_initialized():
        return [int(item) for item in local_lens]
    backend = torch.distributed.get_backend()
    device = torch.device("cuda", torch.cuda.current_device()) if backend == "nccl" else torch.device("cpu")
    tensor = torch.tensor([int(item) for item in local_lens], dtype=torch.long, device=device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MIN)
    return [int(item) for item in tensor.tolist()]
