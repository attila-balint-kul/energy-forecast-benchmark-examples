import os

import pandas as pd
from statsforecast.models import SeasonalWindowAverage
from enfobench import AuthorInfo, ModelInfo, ForecasterType
from enfobench.evaluation.server import server_factory
from enfobench.evaluation.utils import create_forecast_index, periods_in_duration


class SeasonalWindowAverageModel:
    def __init__(self, seasonality: str, window_size: int):
        self.seasonality = seasonality.upper()
        self.window_size = window_size

    def info(self) -> ModelInfo:
        return ModelInfo(
            name=f"Statsforecast.SeasonalWindowAverage.{self.seasonality}.W{self.window_size}",
            authors=[
                AuthorInfo(name="Attila Balint", email="attila.balint@kuleuven.be")
            ],
            type=ForecasterType.point,
            params={"seasonality": self.seasonality, "window_size": self.window_size,},
        )

    def forecast(
        self,
        horizon: int,
        history: pd.DataFrame,
        past_covariates: pd.DataFrame | None = None,
        future_covariates: pd.DataFrame | None = None,
        level: list[int] | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        # Create model using period length
        y = history.y
        periods = periods_in_duration(y.index, duration=self.seasonality)
        model = SeasonalWindowAverage(
            season_length=periods, window_size=self.window_size
        )

        # Make forecast
        pred = model.forecast(y=y.values, h=horizon, **kwargs)

        # Create index for forecast
        index = create_forecast_index(history=history, horizon=horizon)

        # Format forecast dataframe
        forecast = pd.DataFrame(
            index=index,
            data={
                "yhat": pred["mean"],
            },
        ).fillna(y.mean())
        return forecast


# Load parameters
seasonality = os.getenv("ENFOBENCH_MODEL_SEASONALITY")
window_size = int(os.getenv("ENFOBENCH_MODEL_WINDOW_SIZE"))

# Instantiate your model
model = SeasonalWindowAverageModel(seasonality, window_size)

# Create a forecast server by passing in your model
app = server_factory(model)
