import os
import pickle

import pandas as pd
import numpy as np
import xgboost as xgb

from typing import Tuple, Union, List
from datetime import datetime

MODEL_PATH = os.path.join(os.path.dirname(__file__), "trained_model.pkl")

TOP_10_FEATURES = [
    "OPERA_Latin American Wings",
    "MES_7",
    "MES_10",
    "OPERA_Grupo LATAM",
    "MES_12",
    "TIPOVUELO_I",
    "MES_4",
    "MES_11",
    "OPERA_Sky Airline",
    "OPERA_Copa Air",
]

class DelayModel:

    def __init__(
        self
    ):
        self._model = None # Model should be saved in this attribute.

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        features = pd.concat([
            pd.get_dummies(data["OPERA"], prefix="OPERA"),
            pd.get_dummies(data["TIPOVUELO"], prefix="TIPOVUELO"),
            pd.get_dummies(data["MES"], prefix="MES"),
        ], axis=1)

        features = features.reindex(columns=TOP_10_FEATURES, fill_value=0)

        if target_column is not None:
            if target_column not in data.columns:
                data["min_diff"] = data.apply(self._get_min_diff, axis=1)
                data["delay"] = np.where(data["min_diff"] > 15, 1, 0)
            target = data[[target_column]]
            return features, target

        return features

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        n_y0 = len(target[target.iloc[:, 0] == 0])
        n_y1 = len(target[target.iloc[:, 0] == 1])
        scale = n_y0 / n_y1

        self._model = xgb.XGBClassifier(
            random_state=1,
            learning_rate=0.01,
            scale_pos_weight=scale,
        )
        self._model.fit(features, target.iloc[:, 0])

        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self._model, f)

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.

        Returns:
            (List[int]): predicted targets.
        """
        if self._model is None:
            with open(MODEL_PATH, "rb") as f:
                self._model = pickle.load(f)

        predictions = self._model.predict(features)
        return [int(p) for p in predictions]

    @staticmethod
    def _get_min_diff(row):
        fecha_o = datetime.strptime(row["Fecha-O"], "%Y-%m-%d %H:%M:%S")
        fecha_i = datetime.strptime(row["Fecha-I"], "%Y-%m-%d %H:%M:%S")
        return (fecha_o - fecha_i).total_seconds() / 60
