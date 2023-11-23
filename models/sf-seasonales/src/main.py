import os

import pandas as pd
from datetime import timedelta
from statsforecast.models import SeasonalExponentialSmoothing
from pandas import Timedelta
from enfobench import AuthorInfo, ModelInfo, ForecasterType
from enfobench.evaluation.server import server_factory
from enfobench.evaluation.utils import create_forecast_index


def periods_in_duration(ts, duration) -> int:
    if isinstance(duration, timedelta):
        duration = Timedelta(duration)

    first_delta = ts[1] - ts[0]
    last_delta = ts[-1] - ts[-2]
    assert first_delta == last_delta, "Season length is not constant"

    periods = duration / first_delta
    assert periods.is_integer(), "Season length is not a multiple of the frequency"

    return int(periods)


class ExponentialSmoothing:

    def __init__(self, season_length: str, alpha: float):
        self.season_length = season_length

        if alpha < 0 or alpha > 1:
            msg = "Alpha parameter must be between 0 and 1"
            raise ValueError(msg)
        self._alpha = round(alpha, 3)

    def info(self) -> ModelInfo:
        return ModelInfo(
            name=f"Statsforecast.SeasonalExponentialSmoothing.{self.season_length}.A{self._alpha:.3f}",
            authors=[
                AuthorInfo(
                    name="Attila Balint",
                    email="attila.balint@kuleuven.be"
                )
            ],
            type=ForecasterType.point,
            params={
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

        periods = periods_in_duration(ts=y.index, duration=pd.Timedelta(self.season_length))
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
seasonality = str(os.getenv("ENFOBENCH_MODEL_SEASONALITY"))
alpha = float(os.getenv("ENFOBENCH_MODEL_ALPHA"))

# Instantiate your model
model = ExponentialSmoothing(seasonality, alpha)
# Create a forecast server by passing in your model
app = server_factory(model)
