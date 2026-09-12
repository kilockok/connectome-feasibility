"""Leak-free, trajectory-disjoint input construction for latent_state_v1."""
import torch

MAX_K = 32


def windows(data, trajectory, end, k):
    """Last token (X[t], U[t]) predicts X[t+1]. Never reads data['z']."""
    dev = data['states'].device
    trajectory = torch.as_tensor(trajectory, device=dev)
    end = torch.as_tensor(end, device=dev)
    idx = end[:, None] + torch.arange(1-k, 1, device=dev)
    if k < 1 or bool((idx < 0).any()) or bool((end >= data['stimulus'].shape[1]).any()):
        raise ValueError('Window outside trajectory')
    x = torch.cat((data['states'][trajectory[:, None], idx],
                   data['stimulus'][trajectory[:, None], idx].unsqueeze(-1)), -1)
    return x, data['states'][trajectory, end+1]


def history_control(x, mode, generator=None):
    if mode == 'ordered':
        return x
    if mode == 'last':
        return x[:, -1:].expand_as(x)
    if mode != 'shuffle':
        raise ValueError(mode)
    # Preserve current state and pair ALL neurons/features under one temporal permutation.
    perm = torch.randperm(x.shape[1]-1, generator=generator).to(x.device)
    return torch.cat((x[:, perm], x[:, -1:]), 1)


def sample_indices(data, count, seed, min_end=MAX_K-1):
    g = torch.Generator().manual_seed(seed)
    b = torch.randint(len(data['states']), (count,), generator=g)
    t = torch.randint(min_end, data['stimulus'].shape[1], (count,), generator=g)
    return b, t
