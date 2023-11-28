import os

import pandas as pd
from statsforecast.models import SeasonalExponentialSmoothing
from enfobench import AuthorInfo, ModelInfo, ForecasterType
from enfobench.evaluation.server import server_factory
from enfobench.evaluation.utils import create_forecast_index, periods_in_duration


class ExponentialSmoothing:

    def __init__(self, seasonality: str, alpha: float):
        self.seasonality = seasonality

        if alpha < 0 or alpha > 1:
            msg = "Alpha parameter must be between 0 and 1"
            raise ValueError(msg)
        self._alpha = round(alpha, 3)

    def info(self) -> ModelInfo:
        return ModelInfo(
            name=f"Statsforecast.SeasonalExponentialSmoothing.{self.seasonality}.A{self._alpha:.3f}",
            authors=[
                AuthorInfo(
                    name="Attila Balint",
                    email="attila.balint@kuleuven.be"
                )
            ],
            type=ForecasterType.point,
            params={
                "season_length": self.seasonality,
                "alpha": self._alpha,
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
        model = SeasonalExponentialSmoothing(season_length=periods, alpha=self._alpha)

        # Make forecast
        pred = model.forecast(
            y=y.values,
            h=horizon,
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
alpha = float(os.getenv("ENFOBENCH_MODEL_ALPHA"))

# Instantiate your model
model = ExponentialSmoothing(seasonality, alpha)
# Create a forecast server by passing in your model
app = server_factory(model)
