from .format import bold
from .logger import logger


def add_after(old_text, new_text):
    def _add_before(module_data):
        whole_text = old_text + "\n" + new_text
        if whole_text in module_data:
            logger.warning(
                f"new_text={bold(new_text)} has already been added after old_text={bold(old_text)}"
            )
            return module_data
        if old_text not in module_data:
            logger.warning(f"Unable to locate old_text={bold(old_text)}")
            return module_data
        logger.info(f"{bold(old_text)} => {bold(whole_text)}")
        return module_data.replace(old_text, whole_text)

    return _add_before


def add_before(old_text, new_text):
    def _add_before(module_data):
        whole_text = new_text + "\n" + old_text
        if whole_text in module_data:
            logger.warning(
                f"new_text={bold(new_text)} has already been added before old_text={bold(old_text)}"
            )
            return module_data
        if old_text not in module_data:
            logger.warning(f"Unable to locate old_text={bold(old_text)}")
            return module_data
        logger.info(f"{bold(old_text)} => {bold(whole_text)}")
        return module_data.replace(old_text, whole_text)

    return _add_before


def replace(old_text, new_text):
    def _replace(module_data):
        if new_text in module_data:
            logger.warning(
                f"new_text={bold(new_text)} has already been "
                f"replaced with old_text={bold(old_text)}"
            )
            return module_data
        if old_text not in module_data:
            logger.warning(f"Unable to locate old_text={bold(old_text)}")
            return module_data
        logger.info(f"{bold(old_text)} => {bold(new_text)}")
        return module_data.replace(old_text, new_text)

    return _replace


def transform_module(module, transforms):
    module_file = module.__file__
    with open(module_file, "rt") as fin:
        module_data = fin.read()
    for transform in transforms:
        module_data = transform(module_data)
    with open(module_file, "wt") as fout:
        fout.write(module_data)
