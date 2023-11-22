import pandas as pd
from darts import TimeSeries

from darts.models.forecasting.baselines import NaiveMean

from enfobench import AuthorInfo, ModelInfo, ForecasterType
from enfobench.evaluation.server import server_factory


class NaiveModel:

    def info(self) -> ModelInfo:
        return ModelInfo(
            name="Darts.NaiveMean",
            authors=[
                AuthorInfo(
                    name="Attila Balint",
                    email="attila.balint@kuleuven.be"
                )
            ],
            type=ForecasterType.point,
            params={},
        )

    def forecast(
        self,
        horizon: int,
        history: pd.DataFrame,
        past_covariates: pd.DataFrame | None = None,
        future_covariates: pd.DataFrame | None = None,
        **kwargs
    ) -> pd.DataFrame:
        model = NaiveMean()

        series = TimeSeries.from_dataframe(history, value_cols=['y'])
        model.fit(series)

        # Make forecast
        pred = model.predict(horizon)

        forecast = (
            pred.pd_dataframe()
            .rename(columns={"y": "yhat"})
            .fillna(history['y'].mean())
        )
        return forecast


# Instantiate your model
model = NaiveModel()
# Create a forecast server by passing in your model
app = server_factory(model)
