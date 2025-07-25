"""
Name: ionic-conductivity-predictor-app
Author: Navin
Description: Takes basic material information from user to predict lithium-ion conductivity
Date: 2025.07.24
Version: 0.1.0
Log:
- 0.1.0: Initial version
"""
import pandas as pd
from matminer



class ICPredApp:
    def __init__(self, input = pd.DataFrame):

        """
        :param input: Dataset containing user input of material paramaeters
        """

        if not isinstance(input, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame as input.")

        self.input = input

