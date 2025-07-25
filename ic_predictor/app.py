"""
Name: ionic-conductivity-predictor-app
Author: Navin
Description: Takes basic material information from user to predict lithium-ion conductivity
Date: 2025.07.24
Version: 0.1.0
Log:
- 0.1.0: Initial version
"""

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

class ICPredApp:
    def __init__(self, root):
        """
        :param root: Root directory of the app
        """

        self.root = root
        # Paths relative to project root
        self.FORMAT_PATH = os.path.join(self.root, "saves", "format.csv")
        self.MODEL_PATH = os.path.join(self.root, "saves", "best_model.pth")
        self.XGB_PATH = os.path.join(self.root, "saves", "xgb_model.json")
        self.META_PATH = os.path.join(self.root, "saves", "meta.json")
        self.SCALER_PATH = os.path.join(self.root, "saves", "scaler_X.pkl")

        # Load saved states
        self._load_scaler()
        self._load_meta()
        self._load_dft_model()
        self._load_xgb_model()
        self._load_clean_format()

    # ----- User methods ----- #
    def predict(self, df):
        """
        :param df: DataFrame containing the material input for prediction
        :return: Ionic conductivity in Log10-scale
        """
        comp_df = self._encode_input(df)
        feat_df = self._composition_features(comp_df)
        clean_df = self._clean_features(feat_df)
        final_input = self._predict_dft(clean_df)
        ic_pred_df = self._predict_ic(final_input)

        return ic_pred_df

    # ----- Private methods ----- #
    # Saved states
    def _load_scaler(self):
        with open(self.SCALER_PATH, "rb") as f:
            self.scaler_X = pickle.load(f)

    def _load_meta(self):
        with open(self.META_PATH, "r") as f:
            self.meta = json.load(f)
        self.input_dim = self.meta["input_dim"]
        self.output_dim = self.meta["output_dim"]
        self.target_names = self.meta.get("target_names", [f"target_{i}" for i in range(self.output_dim)])

    def _load_dft_model(self):
        self.dft_model = DFTRegressor(self.input_dim, self.output_dim)
        self.dft_model.load_state_dict(torch.load(self.MODEL_PATH, map_location="cpu"))
        self.dft_model.eval()

    def _load_xgb_model(self):
        self.xgb_model = XGBRegressor()
        self.xgb_model.load_model(self.XGB_PATH)

    def _load_clean_format(self):
        self.clean_format = pd.read_csv(self.FORMAT_PATH)

    # Feature engineering
    def _encode_input(self, data):

        # Converts formula to composition object
        stc = StrToComposition()
        comp_df = stc.featurize_dataframe(data.copy(), 'formula', ignore_errors=True)
        comp_df.drop('formula', axis=1, inplace=True)

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

    def _composition_features(self, comp_df):
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

    def _clean_features(self,comp_ft):
        clean_ft = comp_ft[self.clean_format.columns].copy()
        return clean_ft

    # Surrogate prediction
    def _predict_dft(self, clean_ft):

        scaled_ft = self.scaler_X.transform(clean_ft)
        scaled_ft_df = pd.DataFrame(scaled_ft, columns=clean_ft.columns)

        # Prepare Dataloader for inference dataset
        inference_dataset = InferenceDataset(scaled_ft)
        inference_loader = DataLoader(inference_dataset,
                                      batch_size=32,
                                      shuffle=False,
                                      )

        # Run prediction
        all_preds = []

        with torch.no_grad():
            for X_batch in inference_loader:
                preds = self.dft_model(X_batch)
                all_preds.append(preds.cpu().numpy())

        # Stack and concat dataframe
        y_pred = np.vstack(all_preds)
        y_pred_df = pd.DataFrame(y_pred, columns= self.target_names)

        ## Concat
        final_input = pd.concat([scaled_ft_df.reset_index(drop=True), y_pred_df.reset_index(drop=True)], axis=1)

        return final_input

    # Ionic conductivity prediction
    def _predict_ic(self, final_input):

        # Recreate model instance
        ic_pred = self.xgb_model.predict(final_input)
        ic_pred_df = pd.DataFrame(ic_pred, columns=['Ionic conductivity (Log10)'])
        return ic_pred_df

if __name__ == "__main__":
    root = "root"
    user_input_df = pd.read_csv("/test/test.csv")
    app = ICPredApp(root)
    prediction = app.predict(user_input_df)
    print(prediction)