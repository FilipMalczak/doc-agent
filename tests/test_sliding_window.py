import pytest

from docassist.agents.rag.tool import sliding_overlapping_window


@pytest.mark.parametrize(
    "data, step, max_len, expected",
    [
        # Trivial cases
        ([], 3, 5, []),
        ([1], 1, 5, [[1]]),
        ([1, 2, 3], 5, 5, [[1, 2, 3]]),

        # No overlap (step == max_len)
        (
            list(range(10)),
            3,
            3,
            [
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
                [9],
            ],
        ),

        # Overlap
        (
            list(range(10)),
            2,
            5,
            [
                [0, 1, 2, 3, 4],
                [2, 3, 4, 5, 6],
                [4, 5, 6, 7, 8],
                [6, 7, 8, 9],
                [8, 9],
            ],
        ),

        # Heavy overlap
        (
            list(range(7)),
            1,
            4,
            [
                [0, 1, 2, 3],
                [1, 2, 3, 4],
                [2, 3, 4, 5],
                [3, 4, 5, 6],
                [4, 5, 6],
                [5, 6],
                [6],
            ],
        ),

        # step > 1, max_len > remaining
        (
            list(range(5)),
            2,
            10,
            [
                [0, 1, 2, 3, 4],
                [2, 3, 4],
                [4],
            ],
        ),
    ],
)
def test_sliding_overlapping_window(data, step, max_len, expected):
    # act
    result = list(sliding_overlapping_window(data, step=step, max_len=max_len))

    # 1. Exact window contents
    assert result == expected

    # 2. No window exceeds max_len
    assert all(len(w) <= max_len for w in result)

    # 3. No empty windows
    assert all(len(w) > 0 for w in result)

    # 4. Windows are contiguous slices of data
    for w in result:
        start = data.index(w[0])
        assert data[start:start + len(w)] == w

    # 5. Start indices advance by step
    starts = [data.index(w[0]) for w in result]
    for a, b in zip(starts, starts[1:]):
        assert b - a == step