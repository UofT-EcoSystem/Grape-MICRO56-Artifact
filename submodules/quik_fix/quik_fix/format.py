def bold(str_or_obj):
    if not isinstance(str_or_obj, str):
        str_or_obj = f"{str_or_obj}"
    return "\033[1m" + str_or_obj + "\033[0m"


def emph(str_or_obj):
    if not isinstance(str_or_obj, str):
        str_or_obj = f"{str_or_obj}"
    return "\033[1;4m" + str_or_obj + "\033[0m"


def positive(str_or_obj):
    if not isinstance(str_or_obj, str):
        str_or_obj = f"{str_or_obj}"
    return "\033[1;32;7m" + str_or_obj + "\033[0m"


def negative(str_or_obj):
    if not isinstance(str_or_obj, str):
        str_or_obj = f"{str_or_obj}"
    return "\033[1;31;7m" + str_or_obj + "\033[0m"


def hl(str_or_obj):  # pylint: disable=invalid-name
    if not isinstance(str_or_obj, str):
        str_or_obj = f"{str_or_obj}"
    return "\033[1;33;7m" + str_or_obj + "\033[0m"
