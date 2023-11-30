import os

import pandas as pd
from darts import TimeSeries
from darts.models import RegressionModel
from sklearn.linear_model import LinearRegression
from enfobench import AuthorInfo, ModelInfo, ForecasterType
from enfobench.evaluation.server import server_factory
from enfobench.evaluation.utils import periods_in_duration


class DartsLinearRegressionModel:
    def __init__(self, seasonality: str):
        self.seasonality = seasonality.upper()

    def info(self) -> ModelInfo:
        return ModelInfo(
            name=f"Darts.LinearRegression.{self.seasonality}",
            authors=[
                AuthorInfo(name="Mohamad Khalil", email="coo17619@newcastle.ac.uk")
            ],
            type=ForecasterType.point,
            params={
                "seasonality": self.seasonality,
            },
        )

    def forecast(
        self,
        horizon: int,
        history: pd.DataFrame,
        past_covariates: pd.DataFrame | None = None,
        future_covariates: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        # Fill missing values
        history = history.fillna(history.y.mean())

        # Create model
        periods = periods_in_duration(history.index, duration=self.seasonality)
        model = RegressionModel(
            lags=list(range(-periods, 0)),
            output_chunk_length=horizon,
            model=LinearRegression(),
        )

        # Fit model
        series = TimeSeries.from_dataframe(history, value_cols=["y"])
        model.fit(series)

        # Make forecast
        pred = model.predict(horizon)

        # Postprocess forecast
        forecast = (
            pred.pd_dataframe().rename(columns={"y": "yhat"}).fillna(history.y.mean())
        )
        return forecast


# Load parameters
seasonality = os.getenv("ENFOBENCH_MODEL_SEASONALITY")

# Instantiate your model
model = DartsLinearRegressionModel(seasonality=seasonality)

# Create a forecast server by passing in your model
app = server_factory(model)
