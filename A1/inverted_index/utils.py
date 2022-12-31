import itertools
from typing import Union

import numpy as np
import numba as nb


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def dict_to_str(dct, sep='_'):
    return sep.join([f'{k}={v}' for k, v in dct.items()])


def logn(x, base: Union[int, float] = np.e):
    assert base > 0
    if base == 2:
        return np.log2(x, out=np.zeros_like(x, dtype=np.float64), where=(x != 0))
    elif base == 10:
        return np.log10(x, out=np.zeros_like(x, dtype=np.float64), where=(x != 0))
    elif base == np.e:
        return np.log(x, out=np.zeros_like(x, dtype=np.float64), where=(x != 0))
    else:
        return np.log(x, out=np.zeros_like(x, dtype=np.float64), where=(x != 0)) / np.log(base)


@nb.njit(cache=True)
def divide_rows_csr(data, indices, indptr, divisors):
    m = indptr.shape[0] - 1

    result = data.copy().astype(np.float32)
    for i in nb.prange(m):
        result[indptr[i]: indptr[i + 1]] = result[indptr[i]: indptr[i + 1]] / divisors[i]

    return result


@nb.njit(cache=True)
def top_k_csr(data, indices, indptr, K):
    m = indptr.shape[0] - 1
    k = min(K, data.shape[0])
    max_indices = np.zeros((m, k), dtype=indices.dtype)
    max_values = np.zeros((m, k), dtype=data.dtype)

    for i in nb.prange(m):
        top_inds = np.argsort(data[indptr[i]: indptr[i + 1]])[::-1][:k]
        max_indices[i] = indices[indptr[i]: indptr[i + 1]][top_inds]
        max_values[i] = data[indptr[i]: indptr[i + 1]][top_inds]

    return max_indices, max_values


def generate_gs_schemes(match_query=False):
    _available_schemes = [['b', 'n', 'a', 'l', 'L', 'd'],
                          ['n', 'f', 't', 'p'],
                          ['n', 'c', 'u', 'b']]
    if match_query:
        return [f'{"".join(scm)}.{"".join(scm)}' for scm in itertools.product(*_available_schemes)]
    else:
        return [f'{"".join(scm1)}.{"".join(scm2)}'
                for scm1 in itertools.product(*_available_schemes)
                for scm2 in itertools.product(*_available_schemes)]
