import os

import pandas as pd
from darts import TimeSeries
from darts.models import LightGBMModel
from enfobench import AuthorInfo, ModelInfo, ForecasterType
from enfobench.evaluation.server import server_factory
from enfobench.evaluation.utils import periods_in_duration


class DartsLightGBMModel:
    def __init__(self, seasonality: str):
        self.seasonality = seasonality.upper()

    def info(self) -> ModelInfo:
        return ModelInfo(
            name=f"Darts.LightGBM.{self.seasonality}.Direct",
            authors=[
                AuthorInfo(name="Mohamad Khalil", email="coo17619@newcastle.ac.uk")
            ],
            type=ForecasterType.point,
            params={"seasonality": self.seasonality,},
        )

    def forecast(
        self,
        horizon: int,
        history: pd.DataFrame,
        past_covariates: pd.DataFrame | None = None,
        future_covariates: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        periods = periods_in_duration(history.index, duration=self.seasonality)
        model = LightGBMModel(
            lags=list(range(-periods, 0)),
            output_chunk_length=horizon,
            multi_models=False,
        )

        series = TimeSeries.from_dataframe(history, value_cols=["y"])
        model.fit(series)

        # Make forecast
        pred = model.predict(horizon)

        forecast = (
            pred.pd_dataframe()
            .rename(columns={"y": "yhat"})
            .fillna(history["y"].mean())
        )

        return forecast


# Load parameters
seasonality = os.getenv("ENFOBENCH_MODEL_SEASONALITY")

# Instantiate your model
model = DartsLightGBMModel(seasonality=seasonality)

# Create a forecast server by passing in your model
app = server_factory(model)
