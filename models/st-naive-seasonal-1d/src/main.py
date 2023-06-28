from __future__ import annotations

from datetime import timedelta

from enfobench.evaluation.server import server_factory
import pandas as pd
from enfobench.evaluation import ModelInfo, ForecasterType
from enfobench.evaluation.utils import create_forecast_index
from pandas import Timedelta
from statsforecast.models import SeasonalNaive


def periods_in_duration(ts, duration) -> int:
    if isinstance(duration, timedelta):
        duration = Timedelta(duration)

    first_delta = ts[1] - ts[0]
    last_delta = ts[-1] - ts[-2]
    assert first_delta == last_delta, "Season length is not constant"

    periods = duration / first_delta
    assert periods.is_integer(), "Season length is not a multiple of the frequency"

    return int(periods)


class NaiveSeasonal:

    def __init__(self, season_length: str):
        self.season_length = season_length

    def info(self) -> ModelInfo:
        return ModelInfo(
            name="statsforecast.models.SeasonalNaive",
            type=ForecasterType.quantile,
            params={
                "seasonality": self.season_length,
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
        y = history.set_index('ds').y
        periods = periods_in_duration(ts=y.index, duration=pd.Timedelta(self.season_length))
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
            .rename_axis("ds")
            .rename(columns={"mean": "yhat"})
            .reset_index()
            .fillna(y.mean())
        )
        return forecast


# Instantiate your model
model = NaiveSeasonal('1D')
# Create a forecast server by passing in your model
app = server_factory(model)
