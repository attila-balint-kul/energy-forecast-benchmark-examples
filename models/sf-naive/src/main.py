from __future__ import annotations

from enfobench.evaluation.server import server_factory
import pandas as pd
from enfobench.evaluation import ModelInfo, ForecasterType
from enfobench.evaluation.utils import create_forecast_index
from statsforecast.models import Naive


class NaiveModel:

    def info(self) -> ModelInfo:
        return ModelInfo(
            name="statsforecast.models.Naive",
            type=ForecasterType.quantile,
            params={},
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
        model = Naive()

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
model = NaiveModel()
# Create a forecast server by passing in your model
app = server_factory(model)
