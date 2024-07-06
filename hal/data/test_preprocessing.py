import numpy as np
from numpy.testing import assert_array_equal

from hal.data.preprocessing import convert_target_button_to_one_hot


def test_convert_target_to_one_hot_3d() -> None:
    # Test case 1: Basic scenario (keep the same as before)
    arr1 = np.array(
        [
            [
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 1, 0, 1],
                [0, 0, 1, 0, 1],
                [0, 0, 1, 0, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        ],
        dtype=np.int8,
    )
    expected1 = np.array(
        [
            [
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 1],
            ]
        ],
        dtype=np.int8,
    )
    assert_array_equal(convert_target_button_to_one_hot(arr1), expected1)

    # Test case 2: Basic scenario (keep the same as before)
    arr2 = np.array(
        [
            [
                [1, 0, 0, 1, 0],
                [1, 0, 0, 1, 0],
                [1, 0, 1, 0, 1],
                [0, 0, 1, 0, 1],
                [0, 0, 1, 0, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        ],
        dtype=np.int8,
    )
    expected2 = np.array(
        [
            [
                [1, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 1],
            ]
        ],
        dtype=np.int8,
    )
    assert_array_equal(convert_target_button_to_one_hot(arr2), expected2)

    print("All test cases passed!")
