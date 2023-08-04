from types import MethodType


class ArgParseAdoptorToPyTestConfig:
    __slots__ = ("parser",)

    def __init__(self, parser):
        self.parser = parser

    def addoption(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def parse_args(self):  # pylint: disable=arguments-differ
        args = self.parser.parse_args()

        def getoption(self, option):
            return getattr(self, option)

        setattr(args, "getoption", MethodType(getoption, args))
        return args
