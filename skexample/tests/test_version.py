# Example test file.

from numpy.testing import assert_equal
import skexample

def test_version_good():
    assert_equal(skexample.__version__, "0.1")
