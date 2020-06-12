import numpy as np


def sample_tensor_indices(tensor, n_samples):
    ind_1d = tensor.float().flatten().multinomial(n_samples)
    samples = np.unravel_index(ind_1d, tensor.size())
    samples = np.array(samples)
    samples = [tuple(r) for r in samples.T]
    return samples
