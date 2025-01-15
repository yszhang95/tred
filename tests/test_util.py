#!/usr/bin/env pytest

from tred.util import slice_first, slice_length

import torch
def test_util_slices():

    assert slice_first(slice(None, 2)) == 0
    assert slice_first(slice(2)) == 0
    assert slice_first(slice(1,2)) == 1
    assert slice_first(slice(None, None, 2)) == 0

    l = list(range(10))

    assert (slice_length(slice(1,3), 10) == 2)
    assert (slice_length(slice(0,20), 10) == 10)
    assert (slice_length(slice(0,10,3), 10) == 4)
    assert (slice_length(slice(0,20,3), 10) == 4)
