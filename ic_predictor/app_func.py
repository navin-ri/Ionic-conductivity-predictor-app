# Functionalize the app

import os
import json
import pickle

import pandas as pd
import numpy as np
from pymatgen.symmetry.groups import SpaceGroup
from matminer.featurizers.composition import ElementProperty, Stoichiometry
from matminer.featurizers.conversions import StrToComposition
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from xgboost import XGBRegressor

from model import DFTRegressor, InferenceDataset

# Paths relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
FORMAT_PATH = os.path.join(BASE_DIR, "saves", "format.csv" )
MODEL_PATH = os.path.join(BASE_DIR, "saves", "best_model.pth")
XGB_PATH = os.path.join(BASE_DIR, "saves", "xgb_model.json")
META_PATH = os.path.join(BASE_DIR, "saves", "meta.json")
SCALER_NN_PATH = os.path.join(BASE_DIR, "saves", "scaler_X.pkl")

# ----- Feature engineering ---- #
def _encode_input(data):

    # Converts formula to composition object
    stc = StrToComposition()
    comp_df = stc.featurize_dataframe(data.copy(), 'formula', ignore_errors = True)
    comp_df.drop('formula', axis = 1, inplace=True)

    # Converts space group category to group number labels
    # Step 1: Process only unique space group symbols
    unique_symbols = comp_df['space_group'].unique()

    # Step 2: Build mapping (symbol ➔ number)
    symbol_to_number = {}
    for symbol in unique_symbols:
        try:
            sg = SpaceGroup(symbol)
            symbol_to_number[symbol] = sg.int_number
        except:
            symbol_to_number[symbol] = None  # Handle bad symbols if needed

    # Step 3: Map numbers back to full dataset
    comp_df['space_group_number'] = comp_df['space_group'].map(symbol_to_number)
    return comp_df

def _composition_features(comp_df):
    composition_featurizers = [
        ElementProperty.from_preset('magpie', impute_nan=True),
        Stoichiometry(),
    ]
    comp_ft = comp_df.copy()
    for feature in composition_featurizers:
        comp_ft = feature.featurize_dataframe(comp_ft,
                                                  'composition',
                                                  ignore_errors=True)
    return comp_ft

def _clean_features(comp_ft):
    clean_format = pd.read_csv(FORMAT_PATH)
    clean_ft = comp_ft[clean_format.columns].copy()
    return clean_ft

# ----- Surrogate prediction ----- #
def _dft_predictor(clean_ft):

    # import the saved scaler function to scale the input
    with open(SCALER_NN_PATH, "rb") as f:
        scaler_nn = pickle.load(f)
    scaled_ft = scaler_nn.transform(clean_ft)
    scaled_ft_df = pd.DataFrame(scaled_ft, columns = clean_ft.columns)

    # Prepare Dataloader for inference dataset
    inference_dataset = InferenceDataset(scaled_ft)
    inference_loader = DataLoader(inference_dataset,
                                      batch_size= 32,
                                      shuffle =False,
                                      )

    # Predict DFT parameters
    ## Load model meta data
    with open(META_PATH, 'r') as f:
        meta = json.load(f)

    input_dim = meta['input_dim']
    output_dim = meta['output_dim']
    target_names = meta.get('target_names', [f'target_{i}' for i in range(output_dim)])

    # Load model
    model = DFTRegressor(input_dim=input_dim, output_dim=output_dim)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()

    # Run prediction
    all_preds = []

    with torch.no_grad():
        for X_batch in inference_loader:
            preds = model(X_batch)
            all_preds.append(preds.cpu().numpy())

    # Stack and concat dataframe
    y_pred = np.vstack(all_preds)
    y_pred_df = pd.DataFrame(y_pred, columns = target_names)

    ## Concat
    final_input = pd.concat([scaled_ft_df.reset_index(drop=True), y_pred_df.reset_index(drop = True)], axis = 1)

    return final_input

def _ic_predictor(final_input):

    # Recreate model instance
    xgb_model = XGBRegressor()
    xgb_model.load_model(XGB_PATH)

    ic_pred = xgb_model.predict(final_input)
    ic_pred_df = pd.DataFrame(ic_pred, columns= ['Ionic conductivity (Log10)'])
    return ic_pred_df

