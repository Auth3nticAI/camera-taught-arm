import numpy as np

from cta.filters import OneEuro


def test_one_euro_passes_constant():
    f = OneEuro()
    x = np.array([0.3, -1.0, 2.0])
    for i in range(10):
        y = f(x, i * 0.05)
    assert np.allclose(y, x)


def test_one_euro_reduces_noise():
    rng = np.random.default_rng(0)
    f = OneEuro(min_cutoff=1.0, beta=0.0)
    xs = 1.0 + 0.1 * rng.standard_normal((300, 2))
    ys = np.array([f(x, i / 30) for i, x in enumerate(xs)])
    assert ys[30:].std() < 0.5 * xs[30:].std()


def test_one_euro_first_call_and_repeated_timestamp():
    f = OneEuro()
    x = np.array([1.0, 2.0])
    assert np.array_equal(f(x, 0.0), x)
    y = f(np.array([5.0, 5.0]), 0.0)
    assert np.all(np.isfinite(y)) and np.array_equal(y, x)
