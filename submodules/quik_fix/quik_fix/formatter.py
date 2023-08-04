import sys

from .format import *  # pylint: disable=wildcard-import,unused-wildcard-import


if __name__ == "__main__":
    print(eval(sys.argv[1])(sys.argv[2]))  # pylint: disable=eval-used
