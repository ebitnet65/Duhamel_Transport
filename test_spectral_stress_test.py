from run_spectral_stress_test import _windows


def test_stress_windows_have_equal_pixel_budgets():
    first=_windows(5)
    second=_windows(5)
    names=("center_3x3","low_3x3","high_3x3","crosspeak_9")
    assert [int(first[name].sum()) for name in names]==[9,9,9,9]
    for name in names:
        assert (first[name]==second[name]).all()
    assert int(first["full"].sum())==25


def test_stress_windows_reject_undersampled_frequency_grid():
    try:
        _windows(4)
    except ValueError as error:
        assert "at least five" in str(error)
    else:
        raise AssertionError("undersampled grid should be rejected")
