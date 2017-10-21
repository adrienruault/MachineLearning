# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    try:
        D = x.shape[1]
    except IndexError:
        D = 1
        
    N = x.shape[0]
    # First column is offset column of 1
    tx = np.ones(N)
    for d in range(degree):
        tx = np.c_[tx, x**(d+1)]
    
    return tx