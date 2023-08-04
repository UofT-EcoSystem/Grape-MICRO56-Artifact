from quik_fix import ArgParseAdoptorToPyTestConfig


def test_parser(pytestconfig):
    assert pytestconfig.getoption("A") == 10


if __name__ == "__main__":
    import argparse
    from conftest import pytest_addoption

    parser = ArgParseAdoptorToPyTestConfig(argparse.ArgumentParser())

    pytest_addoption(parser)
    args = parser.parse_args()
    assert args.A == 10
