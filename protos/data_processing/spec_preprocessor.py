# import sys

# sys.append('./')
from .preprocessor import Preprocessor


class HomeCreditPreprocessor(Preprocessor):
    def __init__(self, logger=None):
        super(HomeCreditPreprocessor, self).__init__(logger=logger)
