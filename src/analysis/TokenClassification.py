# Basic class for analyzing results from Token Classification experiments
# Implemented by Grusha Prasad
# (https://github.com/grushaprasad)
# September 2024

from .Analysis import Analysis
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn' 

class TokenClassification(Analysis):
    def __init__(self, config: dict, 
                **kwargs):
        super().__init__(config, **kwargs)


    def analyze(self):
        raise NotImplementedError
