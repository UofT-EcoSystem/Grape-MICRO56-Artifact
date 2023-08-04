def pytest_addoption(parser):
    parser.addoption("-A", action="store", type=int, default=10)
