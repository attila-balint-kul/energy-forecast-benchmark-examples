import pandas as pd

from statsforecast.models import Naive

from enfobench import AuthorInfo, ModelInfo, ForecasterType
from enfobench.evaluation.server import server_factory
from enfobench.evaluation.utils import create_forecast_index


class NaiveModel:
    def info(self) -> ModelInfo:
        return ModelInfo(
            name="Statsforecast.Naive",
            authors=[
                AuthorInfo(name="Attila Balint", email="attila.balint@kuleuven.be")
            ],
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
        y = history.y
        model = Naive()

        # Make forecast
        pred = model.forecast(y=y.values, h=horizon, level=level, **kwargs)

        # Create index for forecast
        index = create_forecast_index(history=history, horizon=horizon)

        # Format forecast dataframe
        forecast = (
            pd.DataFrame(index=index, data=pred)
            .rename(columns={"mean": "yhat"})
            .fillna(y.mean())
        )
        return forecast


# Instantiate your model
model = NaiveModel()

# Create a forecast server by passing in your model
app = server_factory(model)
