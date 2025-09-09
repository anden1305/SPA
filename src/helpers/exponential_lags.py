from __future__ import annotations

from typing import List

__all__ = ["exponential_lags"]

def exponential_lags(max_lag: int | None = None, num: int | None = None) -> List[int]:
    """Return exponential lag list: 1,2,4,... up to constraint.

    Specify either ``max_lag`` (include powers-of-two <= max_lag) or ``num`` (number
    of lags). If both provided, stop when either condition reached.
    """
    if max_lag is None and num is None:
        raise ValueError("Provide max_lag or num (or both).")
    lags: list[int] = []
    v = 1
    while True:
        if max_lag is not None and v > max_lag:
            break
        lags.append(v)
        if num is not None and len(lags) >= num:
            break
        v *= 2
    return lags
