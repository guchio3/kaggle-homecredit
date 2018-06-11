from .analyzer import Analyzer


class HomeCreditAnalyzer(Analyzer):
    def __init__(self, logger=None):
        super(HomeCreditAnalyzer, self).__init__(logger=logger)
