import os

import pandas as pd
from statsforecast.models import SeasonalNaive
from enfobench import AuthorInfo, ModelInfo, ForecasterType
from enfobench.evaluation.server import server_factory
from enfobench.evaluation.utils import create_forecast_index, periods_in_duration


class NaiveSeasonal:

    def __init__(self, seasonality: str):
        self.seasonality = seasonality.upper()

    def info(self) -> ModelInfo:
        return ModelInfo(
            name=f"Statsforecast.SeasonalNaive.{self.seasonality}",
            authors=[
                AuthorInfo(
                    name="Attila Balint",
                    email="attila.balint@kuleuven.be"
                )
            ],
            type=ForecasterType.quantile,
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
        level: list[int] | None = None,
        **kwargs
    ) -> pd.DataFrame:
        # Create model using period length
        y = history.y
        periods = periods_in_duration(y.index, duration=self.seasonality)
        model = SeasonalNaive(season_length=periods)

        # Make forecast
        pred = model.forecast(
            y=y.values,
            h=horizon,
            level=level,
            **kwargs
        )

        # Create index for forecast
        index = create_forecast_index(history=history, horizon=horizon)

        # Format forecast dataframe
        forecast = (
            pd.DataFrame(
                index=index,
                data=pred
            )
            .rename(columns={"mean": "yhat"})
            .fillna(y.mean())
        )
        return forecast


# Load parameters
seasonality = os.getenv("ENFOBENCH_MODEL_SEASONALITY")

# Instantiate your model
model = NaiveSeasonal(seasonality)
# Create a forecast server by passing in your model
app = server_factory(model)
