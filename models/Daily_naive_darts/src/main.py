import os
from enfobench.dataset import Dataset
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import NaiveSeasonal
from enfobench.evaluation import evaluate_metrics
from enfobench.evaluation.metrics import mean_absolute_error, mean_bias_error,root_mean_squared_error
from datetime import timedelta
from enfobench import AuthorInfo, ModelInfo, ForecasterType
from enfobench.evaluation.server import server_factory
from enfobench.evaluation.utils import create_forecast_index

class NaiveModel:

    def info(self) -> ModelInfo:
        
        return ModelInfo(
            name="Darts.dailyNaive",
            authors=[
                AuthorInfo(
                    name="Mohamad Khalil",
                    email="coo17619@newcastle.ac.uk"
                )
            ],
            type=ForecasterType.point,
            params={},
        )


    def periods_in_duration(self, ts, duration) -> int:
        
        if isinstance(duration, timedelta):
            duration = Timedelta(duration)
    
        first_delta = ts[1] - ts[0]
        last_delta = ts[-1] - ts[-2]
        assert first_delta == last_delta, "Season length is not constant"
    
        periods = duration / first_delta
        assert periods.is_integer(), "Season length is not a multiple of the frequency"

        return int(periods)

    
    def forecast(
        self,
        horizon: int,
        history: pd.DataFrame,
        past_covariates: pd.DataFrame | None = None,
        future_covariates: pd.DataFrame | None = None,
        **kwargs
        
    ) -> pd.DataFrame:
        
        periods = self.periods_in_duration(ts=history.index, duration=pd.Timedelta("1D"))
        model =  NaiveSeasonal(periods)
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

model = NaiveModel()

app = server_factory(model)
