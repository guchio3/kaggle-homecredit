# import sys

# sys.append('./')
from .converter import Converter


class HomeCreditConveter(Converter):
    def __init__(self, logger=None):
        super(HomeCreditConveter, self).__init__(logger=logger)
