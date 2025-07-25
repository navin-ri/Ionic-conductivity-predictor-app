# Functionalize the app

import pandas as pd
from matminer.featurizers.composition import (ElementProperty,
                                             Stoichiometry,
                                             IonProperty)
from matminer.featurizers.conversions import StrToComposition

def _convert_formula(data):
    stc = StrToComposition()
    comp_df = stc.featurize_dataframe(data.copy(), 'formula', ignore_errors = True)
    return comp_df

def _composition_features(comp_df):
    composition_featurizers = [
        ElementProperty.from_preset('magpie', impute_nan=True),
        Stoichiometry(),
        IonProperty(impute_nan=True),
    ]

    for feature in composition_featurizers:
        comp_ft = feature.featurize_dataframe(comp_df.copy(),
                                                  'composition',
                                                  ignore_errors=True)
    return comp_ft

def _data_clean(comp_ft):
    clean_temp = [

    ]