#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `staple` package."""

# import pytest

from click.testing import CliRunner

import guided_diffusion.staple as staple


# @pytest.fixture
def probabilities_small():
    import numpy as np
    result = 0, 0, 0, 0, 1, 1, 1, 1
    return np.array(result, dtype=np.float64)


def test_staple_small(probabilities_small):
    import numpy as np
    segmentations = (
        (0, 0, 0, 0, 0, 1, 1, 0),
        (0, 0, 0, 0, 1, 1, 1, 1),
        (1, 1, 1, 0, 1, 1, 1, 1),
        (0, 0, 0, 0, 1, 1, 1, 0),
        (0, 1, 0, 0, 0, 1, 1, 1),
        (0, 0, 1, 1, 0, 0, 1, 1),
    )
    arrays = [np.array(s, dtype=np.float64) for s in segmentations]
    print(arrays)

    print("shape",arrays[0])
    s = staple.STAPLE(arrays, convergence_threshold=0)


    result = s.run()
    # np.testing.assert_almost_equal(result, probabilities_small)
    print(result)

if __name__ == "__main__":
    test_staple_small(0.00000001)