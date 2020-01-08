import random
import numpy as np
import torch


__all__ = ["make_reproducible", "intgeomspace", "df_to_csv"]


def make_reproducible(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def intgeomspace(start, stop, num=50, endpoint=True, max_iterations=1000):
    endpoint_offset = 1 if endpoint else 0
    max_len = stop - start + endpoint_offset

    if num > max_len:
        # FIXME
        raise RuntimeError
    elif num == max_len:
        return np.arange(start, stop + endpoint_offset, dtype=np.int)

    iteration = num_offset = 0
    while iteration < max_iterations:
        x = np.geomspace(start, stop, num=num + num_offset, endpoint=endpoint)
        x = np.unique(np.round(x).astype(np.int))

        num_diff = num - len(x)
        if num_diff == 0:
            break

        num_offset += num_diff
        iteration += 1
    else:
        # FIXME error here
        print("Maximum number of iterations was reached")

    return x


def df_to_csv(df, file, header=True, float_format="%.4e", **kwargs):
    df.to_csv(file, header=header, float_format=float_format, **kwargs)
